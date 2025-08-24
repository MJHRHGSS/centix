#ifndef LLM_H
#define LLM_H
#include "tensor.h"

// LLM configuration
typedef struct {
    size_t vocab_size;
    size_t hidden_dim;
    size_t num_layers;
    size_t num_heads;
    size_t max_seq_len;
} llm_config_t;

// LLM model
typedef struct {
    llm_config_t config;
    tensor_t* token_embeddings;
    tensor_t* position_embeddings;
    tensor_t** layer_norms;
    tensor_t** attention_weights;
    tensor_t** feed_forward_weights;
    tensor_t* vocab_projection_weights;  // New: vocabulary projection
} llm_model_t;

// Initialize LLM model
llm_model_t* llm_init(const llm_config_t* config);

// Free LLM model
void llm_free(llm_model_t* model);

// Forward pass through transformer block
tensor_t* transformer_block_forward(const tensor_t* input, 
                                   const tensor_t* attn_weights,
                                   const tensor_t* ff_weights,
                                   const tensor_t* ln1,
                                   const tensor_t* ln2);

// Generate AI response (new function)
char* generate_ai_response(llm_model_t* model, const tensor_t* final_output, size_t max_length);

// Sample next token with temperature
int sample_next_token(tensor_t* logits, float temperature);

// Generate response
char* llm_generate(llm_model_t* model, const char* input, size_t max_length);

// Simple tokenization (character-level)
size_t tokenize(const char* text, int* tokens, size_t max_tokens);

// Detokenize
char* detokenize(const int* tokens, size_t num_tokens);

#endif

