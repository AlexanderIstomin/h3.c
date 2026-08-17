#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <VideoToolbox/VideoToolbox.h>

#include "h3_avwriter.h"

#include <stdio.h>
#include <string.h>

/* Muxes generated frames and audio with AVFoundation instead of spawning
 * FFmpeg. The system frameworks are always present on macOS, so a fresh
 * install can produce video without the user first installing anything, and
 * the video encoder runs on the hardware encoder rather than in software.
 *
 * The frame buffer is tightly packed RGB24; AVFoundation wants BGRA, so each
 * frame is widened during the copy into the pixel buffer's own rowBytes,
 * which is padded to the encoder's alignment and is not the image width. */

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static NSString *codec_for(int width, int height) {
    /* HEVC keeps generated detail at a smaller size, but the hardware encoder
     * refuses odd dimensions and very small canvases on some machines; H.264
     * accepts everything H3 can produce. */
    return (width >= 640 && height >= 640) ? AVVideoCodecTypeHEVC
                                           : AVVideoCodecTypeH264;
}

static int append_video(AVAssetWriterInput *input,
                        AVAssetWriterInputPixelBufferAdaptor *adaptor,
                        const uint8_t *frames, int frame_count, int width,
                        int height, int fps, char *error, size_t error_size) {
    size_t row_source = (size_t)width * 3;
    for (int index = 0; index < frame_count; index++) {
        while (!input.readyForMoreMediaData) {
            [NSThread sleepForTimeInterval:0.002];
        }
        CVPixelBufferRef buffer = NULL;
        CVReturn status = CVPixelBufferPoolCreatePixelBuffer(
            NULL, adaptor.pixelBufferPool, &buffer);
        if (status != kCVReturnSuccess || !buffer) {
            fail(error, error_size,
                 "cannot allocate a pixel buffer for frame %d (%d)", index,
                 status);
            return 0;
        }
        CVPixelBufferLockBaseAddress(buffer, 0);
        uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
        size_t row_destination = CVPixelBufferGetBytesPerRow(buffer);
        const uint8_t *source = frames + (size_t)index * row_source * height;
        for (int y = 0; y < height; y++) {
            const uint8_t *in = source + (size_t)y * row_source;
            uint8_t *out = base + (size_t)y * row_destination;
            for (int x = 0; x < width; x++) {
                out[x * 4 + 0] = in[x * 3 + 2];
                out[x * 4 + 1] = in[x * 3 + 1];
                out[x * 4 + 2] = in[x * 3 + 0];
                out[x * 4 + 3] = 255;
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, 0);
        CMTime stamp = CMTimeMake(index, fps);
        BOOL ok = [adaptor appendPixelBuffer:buffer withPresentationTime:stamp];
        CVPixelBufferRelease(buffer);
        if (!ok) {
            fail(error, error_size, "the encoder rejected frame %d", index);
            return 0;
        }
    }
    [input markAsFinished];
    return 1;
}

static int append_audio(AVAssetWriterInput *input, const float *pcm,
                        int samples, int channels, int sample_rate,
                        char *error, size_t error_size) {
    AudioStreamBasicDescription description = {0};
    description.mSampleRate = sample_rate;
    description.mFormatID = kAudioFormatLinearPCM;
    description.mFormatFlags =
        kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    description.mBitsPerChannel = 32;
    description.mChannelsPerFrame = (UInt32)channels;
    description.mFramesPerPacket = 1;
    description.mBytesPerFrame = (UInt32)(4 * channels);
    description.mBytesPerPacket = description.mBytesPerFrame;

    CMAudioFormatDescriptionRef format = NULL;
    OSStatus status = CMAudioFormatDescriptionCreate(
        NULL, &description, 0, NULL, 0, NULL, NULL, &format);
    if (status != noErr || !format) {
        fail(error, error_size, "cannot describe the audio format (%d)",
             (int)status);
        return 0;
    }

    size_t bytes = (size_t)samples * channels * sizeof(float);
    CMBlockBufferRef block = NULL;
    status = CMBlockBufferCreateWithMemoryBlock(
        NULL, NULL, bytes, kCFAllocatorDefault, NULL, 0, bytes, 0, &block);
    if (status == noErr)
        status = CMBlockBufferReplaceDataBytes(pcm, block, 0, bytes);
    if (status != noErr || !block) {
        if (block) CFRelease(block);
        CFRelease(format);
        fail(error, error_size, "cannot stage the audio samples (%d)",
             (int)status);
        return 0;
    }

    CMSampleBufferRef sample = NULL;
    CMSampleTimingInfo timing = {
        .duration = CMTimeMake(1, sample_rate),
        .presentationTimeStamp = kCMTimeZero,
        .decodeTimeStamp = kCMTimeInvalid
    };
    status = CMSampleBufferCreate(NULL, block, TRUE, NULL, NULL, format,
                                  samples, 1, &timing, 0, NULL, &sample);
    CFRelease(block);
    CFRelease(format);
    if (status != noErr || !sample) {
        if (sample) CFRelease(sample);
        fail(error, error_size, "cannot build the audio sample buffer (%d)",
             (int)status);
        return 0;
    }

    while (!input.readyForMoreMediaData) {
        [NSThread sleepForTimeInterval:0.002];
    }
    BOOL ok = [input appendSampleBuffer:sample];
    CFRelease(sample);
    if (!ok) {
        fail(error, error_size, "the encoder rejected the soundtrack");
        return 0;
    }
    [input markAsFinished];
    return 1;
}

int h3_avwriter_available(void) {
    return 1;
}

int h3_avwriter_write_av_rgb24_f32(const char *path, const uint8_t *frames,
                                   int frame_count, int width, int height,
                                   int fps, const float *pcm, int samples,
                                   int channels, int sample_rate,
                                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !frames || frame_count < 1 || width < 1 ||
        height < 1 || fps < 1) {
        fail(error, error_size, "invalid arguments for the AVFoundation mux");
        return 0;
    }
    int with_audio = pcm && samples > 0 && channels > 0 && sample_rate > 0;

    @autoreleasepool {
        NSString *destination = [NSString stringWithUTF8String:path];
        NSURL *url = [NSURL fileURLWithPath:destination];
        [[NSFileManager defaultManager] removeItemAtURL:url error:nil];

        NSError *writerError = nil;
        AVAssetWriter *writer =
            [AVAssetWriter assetWriterWithURL:url
                                     fileType:AVFileTypeMPEG4
                                        error:&writerError];
        if (!writer) {
            fail(error, error_size, "cannot open %s for writing: %s", path,
                 writerError.localizedDescription.UTF8String);
            return 0;
        }

        NSDictionary *videoSettings = @{
            AVVideoCodecKey: codec_for(width, height),
            AVVideoWidthKey: @(width),
            AVVideoHeightKey: @(height)
        };
        AVAssetWriterInput *video = [AVAssetWriterInput
            assetWriterInputWithMediaType:AVMediaTypeVideo
                           outputSettings:videoSettings];
        video.expectsMediaDataInRealTime = NO;
        if (![writer canAddInput:video]) {
            fail(error, error_size,
                 "the system encoder rejected %dx%d video", width, height);
            return 0;
        }
        [writer addInput:video];

        AVAssetWriterInputPixelBufferAdaptor *adaptor =
            [AVAssetWriterInputPixelBufferAdaptor
                assetWriterInputPixelBufferAdaptorWithAssetWriterInput:video
                                           sourcePixelBufferAttributes:@{
                    (id)kCVPixelBufferPixelFormatTypeKey:
                        @(kCVPixelFormatType_32BGRA),
                    (id)kCVPixelBufferWidthKey: @(width),
                    (id)kCVPixelBufferHeightKey: @(height)
                }];

        AVAssetWriterInput *audio = nil;
        if (with_audio) {
            NSDictionary *audioSettings = @{
                AVFormatIDKey: @(kAudioFormatMPEG4AAC),
                AVNumberOfChannelsKey: @(channels),
                AVSampleRateKey: @(sample_rate),
                AVEncoderBitRateKey: @(192000)
            };
            audio = [AVAssetWriterInput
                assetWriterInputWithMediaType:AVMediaTypeAudio
                               outputSettings:audioSettings];
            audio.expectsMediaDataInRealTime = NO;
            if ([writer canAddInput:audio]) {
                [writer addInput:audio];
            } else {
                audio = nil;
                with_audio = 0;
            }
        }

        if (![writer startWriting]) {
            fail(error, error_size, "cannot start writing %s: %s", path,
                 writer.error.localizedDescription.UTF8String);
            return 0;
        }
        [writer startSessionAtSourceTime:kCMTimeZero];

        /* Soundtrack first, and it is one buffer.
         *
         * The writer interleaves, so it stops taking video once that track
         * runs far enough ahead of the audio — and with the video loop
         * spinning on readyForMoreMediaData until audio it has not been given
         * arrives, that is a deadlock rather than a wait. A 22-frame clip
         * stayed inside the tolerance and muxed; 73 frames hung for as long as
         * it was left, which is every clip over about a second.
         *
         * Appending the audio first and marking it finished leaves nothing for
         * the writer to wait for, so the video runs to the end. This works
         * because the whole soundtrack is a single sample buffer; were it
         * chunked, the two would have to be interleaved by presentation time
         * or driven from requestMediaDataWhenReady. */
        if (with_audio &&
            !append_audio(audio, pcm, samples, channels, sample_rate, error,
                          error_size)) {
            [writer cancelWriting];
            return 0;
        }
        if (!append_video(video, adaptor, frames, frame_count, width, height,
                          fps, error, error_size)) {
            [writer cancelWriting];
            return 0;
        }

        __block BOOL finished = NO;
        [writer finishWritingWithCompletionHandler:^{ finished = YES; }];
        while (!finished) {
            [NSThread sleepForTimeInterval:0.002];
        }
        if (writer.status != AVAssetWriterStatusCompleted) {
            fail(error, error_size, "cannot finish writing %s: %s", path,
                 writer.error.localizedDescription.UTF8String);
            return 0;
        }
    }
    return 1;
}
