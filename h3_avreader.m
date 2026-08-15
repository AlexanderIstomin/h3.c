#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <ImageIO/ImageIO.h>

#include "h3_avreader.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Decodes reference media with the system frameworks instead of spawning
 * FFmpeg, so the app carries no external tool requirement for the formats a
 * macOS user actually hands it. Output layout matches the FFmpeg readers
 * exactly — channel-major F32 in [0,1] for pixels, channel-major F32 at
 * 32 kHz for audio — because callers fall back between the two. */

enum { AUDIO_RATE = 32000, AUDIO_CHANNELS = 2 };
static const double kVideoFPS = 24.0;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

/* Draws into a BGRA bitmap at the requested size, then scatters to
 * channel-major planes. Cover crops centrally; stretch ignores the source
 * ratio, matching what the FFmpeg reader's scale filters do. */
static float *bitmap_to_planes(CGImageRef image, int width, int height,
                               h3_image_fit fit, char *error,
                               size_t error_size) {
    size_t row = (size_t)width * 4;
    uint8_t *bitmap = calloc(row * (size_t)height, 1);
    if (!bitmap) {
        fail(error, error_size, "out of memory rasterizing the reference");
        return NULL;
    }
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(
        bitmap, (size_t)width, (size_t)height, 8, row, space,
        (CGBitmapInfo)kCGImageAlphaNoneSkipFirst |
            kCGBitmapByteOrder32Little);
    CGColorSpaceRelease(space);
    if (!context) {
        free(bitmap);
        fail(error, error_size, "cannot rasterize the reference image");
        return NULL;
    }
    CGContextSetInterpolationQuality(context, kCGInterpolationHigh);
    CGRect target = CGRectMake(0, 0, width, height);
    if (fit == H3_IMAGE_FIT_COVER) {
        double source_w = (double)CGImageGetWidth(image);
        double source_h = (double)CGImageGetHeight(image);
        if (source_w > 0 && source_h > 0) {
            double scale = fmax(width / source_w, height / source_h);
            double draw_w = source_w * scale;
            double draw_h = source_h * scale;
            target = CGRectMake((width - draw_w) / 2, (height - draw_h) / 2,
                                draw_w, draw_h);
        }
    }
    CGContextDrawImage(context, target, image);
    CGContextRelease(context);

    size_t plane = (size_t)width * (size_t)height;
    float *pixels = malloc(plane * 3 * sizeof(*pixels));
    if (!pixels) {
        free(bitmap);
        fail(error, error_size, "out of memory converting the reference");
        return NULL;
    }
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            const uint8_t *sample = bitmap + (size_t)y * row + (size_t)x * 4;
            size_t at = (size_t)y * (size_t)width + (size_t)x;
            pixels[at] = sample[2] / 255.0f;
            pixels[plane + at] = sample[1] / 255.0f;
            pixels[2 * plane + at] = sample[0] / 255.0f;
        }
    free(bitmap);
    return pixels;
}

int h3_avreader_read_image_f32(const char *path, int width, int height,
                               h3_image_fit fit, float **pixels,
                               char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (!path || !*path || !pixels || width < 1 || height < 1) {
        fail(error, error_size, "invalid arguments for the system image reader");
        return 0;
    }
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        CGImageSourceRef source =
            CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
        if (!source) {
            fail(error, error_size, "cannot open %s as an image", path);
            return 0;
        }
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(source);
        if (!image) {
            fail(error, error_size, "cannot decode %s", path);
            return 0;
        }
        float *planes = bitmap_to_planes(image, width, height, fit, error,
                                         error_size);
        CGImageRelease(image);
        if (!planes) return 0;
        *pixels = planes;
    }
    return 1;
}

/* Copies one decoded frame into the channel-major destination. */
static int pixel_buffer_to_planes(CVPixelBufferRef buffer, float *destination,
                                  size_t plane, int width, int height) {
    CVPixelBufferLockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    const uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
    size_t row = CVPixelBufferGetBytesPerRow(buffer);
    if (!base) {
        CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
        return 0;
    }
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            const uint8_t *sample = base + (size_t)y * row + (size_t)x * 4;
            size_t at = (size_t)y * (size_t)width + (size_t)x;
            destination[at] = sample[2] / 255.0f;
            destination[plane + at] = sample[1] / 255.0f;
            destination[2 * plane + at] = sample[0] / 255.0f;
        }
    CVPixelBufferUnlockBaseAddress(buffer, kCVPixelBufferLock_ReadOnly);
    return 1;
}

int h3_avreader_read_video_f32(const char *path, int width, int height,
                               int max_frames, float **pixels, int *frames,
                               char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (frames) *frames = 0;
    if (!path || !*path || !pixels || !frames || width < 1 || height < 1 ||
        max_frames < 5) {
        fail(error, error_size, "invalid arguments for the system video reader");
        return 0;
    }
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
        /* The synchronous accessor is deprecated but the asynchronous
         * replacement would restructure this whole C-callable path; the file
         * is local and already on disk, so loading cannot block on a network. */
        NSArray<AVAssetTrack *> *tracks =
            [asset tracksWithMediaType:AVMediaTypeVideo];
        if (!tracks.count) {
            fail(error, error_size, "%s has no video the system can read", path);
            return 0;
        }
        NSError *readerError = nil;
        AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:asset
                                                              error:&readerError];
        if (!reader) {
            fail(error, error_size, "cannot read %s: %s", path,
                 readerError.localizedDescription.UTF8String);
            return 0;
        }
        AVAssetReaderTrackOutput *output = [AVAssetReaderTrackOutput
            assetReaderTrackOutputWithTrack:tracks.firstObject
                             outputSettings:@{
                (id)kCVPixelBufferPixelFormatTypeKey:
                    @(kCVPixelFormatType_32BGRA),
                (id)kCVPixelBufferWidthKey: @(width),
                (id)kCVPixelBufferHeightKey: @(height)
            }];
        output.alwaysCopiesSampleData = NO;
        if (![reader canAddOutput:output]) {
            fail(error, error_size, "the system decoder rejected %s", path);
            return 0;
        }
        [reader addOutput:output];
        if (![reader startReading]) {
            fail(error, error_size, "cannot start reading %s: %s", path,
                 reader.error.localizedDescription.UTF8String);
            return 0;
        }

        size_t plane = (size_t)width * (size_t)height;
        size_t frame_floats = plane * 3;
        float *decoded = malloc(frame_floats * (size_t)max_frames *
                                sizeof(*decoded));
        if (!decoded) {
            [reader cancelReading];
            fail(error, error_size, "out of memory decoding %s", path);
            return 0;
        }

        /* Sample the source on the 24 fps grid the model expects, so a clip
         * at any frame rate yields the same timeline. */
        int count = 0;
        int next = 0;
        CMSampleBufferRef sample = NULL;
        while (count < max_frames &&
               (sample = [output copyNextSampleBuffer])) {
            CMTime stamp = CMSampleBufferGetPresentationTimeStamp(sample);
            int slot = (int)floor(CMTimeGetSeconds(stamp) * kVideoFPS + 0.5);
            CVPixelBufferRef buffer = CMSampleBufferGetImageBuffer(sample);
            if (buffer && slot >= next) {
                if (pixel_buffer_to_planes(buffer, decoded + frame_floats *
                                           (size_t)count, plane, width,
                                           height))
                    count++;
                next = slot + 1;
            }
            CFRelease(sample);
        }
        [reader cancelReading];

        if (count < 5) {
            free(decoded);
            fail(error, error_size,
                 "%s decoded %d frames, fewer than the five-frame minimum",
                 path, count);
            return 0;
        }
        /* The released cadence: 5 + 17k frames. */
        while (count % 17 != 5) count--;
        *pixels = decoded;
        *frames = count;
    }
    return 1;
}

int h3_avreader_read_audio_f32(const char *path, int max_samples,
                               int truncate_at_limit, float **pcm,
                               int *samples, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pcm) *pcm = NULL;
    if (samples) *samples = 0;
    if (!path || !*path || !pcm || !samples || max_samples < 1) {
        fail(error, error_size, "invalid arguments for the system audio reader");
        return 0;
    }
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
        NSArray<AVAssetTrack *> *tracks =
            [asset tracksWithMediaType:AVMediaTypeAudio];
        if (!tracks.count) {
            fail(error, error_size, "%s has no audio the system can read", path);
            return 0;
        }
        NSError *readerError = nil;
        AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:asset
                                                              error:&readerError];
        if (!reader) {
            fail(error, error_size, "cannot read %s: %s", path,
                 readerError.localizedDescription.UTF8String);
            return 0;
        }
        AVAssetReaderTrackOutput *output = [AVAssetReaderTrackOutput
            assetReaderTrackOutputWithTrack:tracks.firstObject
                             outputSettings:@{
                AVFormatIDKey: @(kAudioFormatLinearPCM),
                AVSampleRateKey: @(AUDIO_RATE),
                AVNumberOfChannelsKey: @(AUDIO_CHANNELS),
                AVLinearPCMBitDepthKey: @32,
                AVLinearPCMIsFloatKey: @YES,
                AVLinearPCMIsNonInterleaved: @NO,
                AVLinearPCMIsBigEndianKey: @NO
            }];
        if (![reader canAddOutput:output]) {
            fail(error, error_size, "the system decoder rejected the audio in %s",
                 path);
            return 0;
        }
        [reader addOutput:output];
        if (![reader startReading]) {
            fail(error, error_size, "cannot start reading %s: %s", path,
                 reader.error.localizedDescription.UTF8String);
            return 0;
        }

        float *decoded = malloc((size_t)max_samples * AUDIO_CHANNELS *
                                sizeof(*decoded));
        if (!decoded) {
            [reader cancelReading];
            fail(error, error_size, "out of memory decoding %s", path);
            return 0;
        }
        int count = 0;
        int overflowed = 0;
        CMSampleBufferRef sample = NULL;
        while ((sample = [output copyNextSampleBuffer])) {
            CMBlockBufferRef block = CMSampleBufferGetDataBuffer(sample);
            size_t length = block ? CMBlockBufferGetDataLength(block) : 0;
            size_t available = length / (sizeof(float) * AUDIO_CHANNELS);
            if (available) {
                size_t room = (size_t)(max_samples - count);
                size_t take = available < room ? available : room;
                if (take < available) overflowed = 1;
                if (take) {
                    /* Interleaved LR arrives; the model wants planar. */
                    float *staging = malloc(take * AUDIO_CHANNELS *
                                            sizeof(*staging));
                    if (!staging) {
                        CFRelease(sample);
                        free(decoded);
                        [reader cancelReading];
                        fail(error, error_size, "out of memory staging audio");
                        return 0;
                    }
                    CMBlockBufferCopyDataBytes(block, 0,
                                               take * AUDIO_CHANNELS *
                                                   sizeof(float), staging);
                    for (size_t index = 0; index < take; index++) {
                        decoded[count + index] = staging[index * 2];
                        decoded[(size_t)max_samples + count + index] =
                            staging[index * 2 + 1];
                    }
                    free(staging);
                    count += (int)take;
                }
            }
            CFRelease(sample);
            if (count >= max_samples) break;
        }
        [reader cancelReading];

        if (overflowed && !truncate_at_limit) {
            free(decoded);
            fail(error, error_size,
                 "%s is longer than the %d samples this input accepts", path,
                 max_samples);
            return 0;
        }
        if (count < 1) {
            free(decoded);
            fail(error, error_size, "%s decoded no audio", path);
            return 0;
        }
        /* Channel-major with the right-channel plane packed against the left,
         * matching the FFmpeg reader's layout for the decoded length. */
        if (count != max_samples)
            memmove(decoded + count, decoded + max_samples,
                    (size_t)count * sizeof(*decoded));
        *pcm = decoded;
        *samples = count;
    }
    return 1;
}
