#ifndef H3_TOKENIZER_H
#define H3_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define H3_PAD_TOKEN_ID UINT32_C(151643)

typedef struct h3_tokenizer h3_tokenizer;

h3_tokenizer *h3_tokenizer_load(const char *tokenizer_json,
                                char *error, size_t error_size);
/* The same, for a tokenizer that arrives as bytes rather than a file: LTX-2.5
 * ships Gemma's inside the text encoder checkpoint. The bytes are not copied
 * and need only outlive the call. */
h3_tokenizer *h3_tokenizer_load_json(const void *bytes, size_t length,
                                     char *error, size_t error_size);
void h3_tokenizer_free(h3_tokenizer *tokenizer);

/* The caller owns *ids and releases it with h3_tokenizer_ids_free(). */
int h3_tokenizer_encode(const h3_tokenizer *tokenizer, const char *utf8,
                        int pad_empty, uint32_t **ids, size_t *count,
                        char *error, size_t error_size);
void h3_tokenizer_ids_free(uint32_t *ids);

/* The caller owns the returned UTF-8 string and releases it with free(). */
char *h3_tokenizer_decode(const h3_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_size);

#endif
