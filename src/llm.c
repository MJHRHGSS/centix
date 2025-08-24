#include "llm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Initialize LLM model with random weights
llm_model_t* llm_init(const llm_config_t* config) {
    llm_model_t* model = malloc(sizeof(llm_model_t));
    if (!model) return NULL;
    
    model->config = *config;
    
    // Initialize embeddings
    model->token_embeddings = tensor_alloc(config->vocab_size, config->hidden_dim);
    model->position_embeddings = tensor_alloc(config->max_seq_len, config->hidden_dim);
    
    if (!model->token_embeddings || !model->position_embeddings) {
        llm_free(model);
        return NULL;
    }
    
    tensor_fill_random(model->token_embeddings);
    tensor_fill_random(model->position_embeddings);
    
    // Initialize vocabulary projection weights
    model->vocab_projection_weights = tensor_alloc(config->hidden_dim, config->vocab_size);
    if (!model->vocab_projection_weights) {
        llm_free(model);
        return NULL;
    }
    tensor_fill_random(model->vocab_projection_weights);
    
    // Initialize layer weights
    model->layer_norms = malloc(config->num_layers * sizeof(tensor_t*));
    model->attention_weights = malloc(config->num_layers * sizeof(tensor_t*));
    model->feed_forward_weights = malloc(config->num_layers * sizeof(tensor_t*));
    
    if (!model->layer_norms || !model->attention_weights || !model->feed_forward_weights) {
        llm_free(model);
        return NULL;
    }
    
    for (size_t i = 0; i < config->num_layers; i++) {
        // Layer norms (scale and bias)
        model->layer_norms[i] = tensor_alloc(2, config->hidden_dim);
        if (!model->layer_norms[i]) {
            llm_free(model);
            return NULL;
        }
        tensor_fill_ones(model->layer_norms[i]);
        
        // Attention weights (Q, K, V projections)
        model->attention_weights[i] = tensor_alloc(config->hidden_dim, config->hidden_dim * 3);
        if (!model->attention_weights[i]) {
            llm_free(model);
            return NULL;
        }
        tensor_fill_random(model->attention_weights[i]);
        
        // Feed-forward weights
        model->feed_forward_weights[i] = tensor_alloc(config->hidden_dim * 4, config->hidden_dim);
        if (!model->feed_forward_weights[i]) {
            llm_free(model);
            return NULL;
        }
        tensor_fill_random(model->feed_forward_weights[i]);
    }
    
    return model;
}

void llm_free(llm_model_t* model) {
    if (!model) return;
    
    if (model->token_embeddings) tensor_free(model->token_embeddings);
    if (model->position_embeddings) tensor_free(model->position_embeddings);
    if (model->vocab_projection_weights) tensor_free(model->vocab_projection_weights);
    
    if (model->layer_norms) {
        for (size_t i = 0; i < model->config.num_layers; i++) {
            if (model->layer_norms[i]) tensor_free(model->layer_norms[i]);
        }
        free(model->layer_norms);
    }
    
    if (model->attention_weights) {
        for (size_t i = 0; i < model->config.num_layers; i++) {
            if (model->attention_weights[i]) tensor_free(model->attention_weights[i]);
        }
        free(model->attention_weights);
    }
    
    if (model->feed_forward_weights) {
        for (size_t i = 0; i < model->config.num_layers; i++) {
            if (model->feed_forward_weights[i]) tensor_free(model->feed_forward_weights[i]);
        }
        free(model->feed_forward_weights);
    }
    
    free(model);
}

// Multi-head attention
tensor_t* multi_head_attention(const tensor_t* input, const tensor_t* weights) {
    size_t seq_len = input->rows;
    size_t hidden_dim = input->cols;
    size_t head_dim = hidden_dim / 4; // Simplified head dimension
    
    // Project to Q, K, V
    tensor_t* qkv = tensor_matmul(input, weights);
    if (!qkv) return NULL;
    
    // Apply attention
    tensor_t* attention_scores = tensor_alloc(seq_len, seq_len);
    if (!attention_scores) {
        tensor_free(qkv);
        return NULL;
    }
    
    // Simple attention computation
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < seq_len; j++) {
            float score = 0.0f;
            for (size_t k = 0; k < head_dim; k++) {
                score += qkv->data[i * qkv->cols + k] * qkv->data[j * qkv->cols + k];
            }
            attention_scores->data[i * seq_len + j] = score / sqrtf(head_dim);
        }
    }
    
    tensor_softmax(attention_scores);
    
    // Apply attention to values
    tensor_t* output = tensor_matmul(attention_scores, qkv);
    
    // Project back to hidden_dim
    tensor_t* proj_weights = tensor_alloc(hidden_dim * 3, hidden_dim);
    if (!proj_weights) {
        tensor_free(qkv);
        tensor_free(attention_scores);
        if (output) tensor_free(output);
        return NULL;
    }
    tensor_fill_random(proj_weights);
    
    tensor_t* final_output = tensor_matmul(output, proj_weights);
    
    tensor_free(qkv);
    tensor_free(attention_scores);
    tensor_free(output);
    tensor_free(proj_weights);
    
    return final_output;
}

// Feed-forward network
tensor_t* feed_forward(const tensor_t* input, const tensor_t* weights) {
    size_t seq_len = input->rows;
    size_t hidden_dim = input->cols;
    
    // First linear layer
    tensor_t* hidden = tensor_alloc(seq_len, hidden_dim * 4);
    if (!hidden) return NULL;
    
    // Simplified feed-forward computation
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < hidden_dim * 4; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < hidden_dim; k++) {
                sum += input->data[i * hidden_dim + k] * weights->data[k * (hidden_dim * 4) + j];
            }
            hidden->data[i * (hidden_dim * 4) + j] = sum;
        }
    }
    
    tensor_gelu(hidden);
    
    // Second linear layer (simplified)
    tensor_t* output = tensor_alloc(seq_len, hidden_dim);
    if (!output) {
        tensor_free(hidden);
        return NULL;
    }
    
    // Simplified projection back
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < hidden_dim; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < hidden_dim * 4; k++) {
                sum += hidden->data[i * (hidden_dim * 4) + k] * 0.25f; // Simplified weights
            }
            output->data[i * hidden_dim + j] = sum;
        }
    }
    
    tensor_free(hidden);
    return output;
}

// Transformer block forward pass
tensor_t* transformer_block_forward(const tensor_t* input, 
                                   const tensor_t* attn_weights,
                                   const tensor_t* ff_weights,
                                   const tensor_t* ln1,
                                   const tensor_t* ln2) {
    
    // Layer norm 1
    tensor_t* norm1 = tensor_alloc(input->rows, input->cols);
    if (!norm1) return NULL;
    tensor_copy(norm1, input);
    tensor_layer_norm(norm1, 1e-5f);
    
    // Attention
    tensor_t* attn_output = multi_head_attention(norm1, attn_weights);
    if (!attn_output) {
        tensor_free(norm1);
        return NULL;
    }
    
    // Residual connection
    tensor_add(attn_output, input);
    
    // Layer norm 2
    tensor_t* norm2 = tensor_alloc(attn_output->rows, attn_output->cols);
    if (!norm2) {
        tensor_free(norm1);
        tensor_free(attn_output);
        return NULL;
    }
    tensor_copy(norm2, attn_output);
    tensor_layer_norm(norm2, 1e-5f);
    
    // Feed-forward
    tensor_t* ff_output = feed_forward(norm2, ff_weights);
    if (!ff_output) {
        tensor_free(norm1);
        tensor_free(attn_output);
        tensor_free(norm2);
        return NULL;
    }
    
    // Final residual connection
    tensor_add(ff_output, attn_output);
    
    tensor_free(norm1);
    tensor_free(attn_output);
    tensor_free(norm2);
    
    return ff_output;
}

// Sample next token with temperature
int sample_next_token(tensor_t* logits, float temperature) {
    // Apply temperature scaling
    tensor_scale(logits, 1.0f / temperature);
    
    // Apply softmax to get probabilities
    tensor_softmax(logits);
    
    // Sample from distribution (roulette wheel)
    float rand_val = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    
    // Only sample from printable ASCII characters (32-126)
    for (int i = 32; i <= 126; i++) {
        cumsum += logits->data[i];
        if (rand_val <= cumsum) {
            return i;
        }
    }
    return 32;  // Fallback to space
}

// Generate AI response using language modeling
char* generate_ai_response(llm_model_t* model, const tensor_t* final_output, size_t max_length) {
    char* response = malloc(max_length + 1);
    if (!response) return NULL;
    
    size_t response_len = 0;
    
    // Start with a seed token (space)
    int current_token = ' ';
    
    for (size_t i = 0; i < max_length && response_len < max_length; i++) {
        // Use the final output to predict next token
        // Project to vocabulary space
        tensor_t* vocab_logits = tensor_vocab_projection(final_output, model->vocab_projection_weights);
        if (!vocab_logits) {
            free(response);
            return NULL;
        }
        
        // Sample next token with temperature
        int next_token = sample_next_token(vocab_logits, 0.7f);
        
        // Add to response
        response[response_len++] = (char)next_token;
        current_token = next_token;
        
        // Stop conditions
        if (next_token == '\n' || next_token == '.' || next_token == '!' || next_token == '?') {
            break;
        }
        
        // Limit response length
        if (response_len >= 50) break;
        
        tensor_free(vocab_logits);
    }
    
    response[response_len] = '\0';
    return response;
}

// Simple character-level tokenization
size_t tokenize(const char* text, int* tokens, size_t max_tokens) {
    size_t len = strlen(text);
    size_t token_count = 0;
    
    for (size_t i = 0; i < len && token_count < max_tokens; i++) {
        tokens[token_count++] = (int)text[i];
    }
    
    return token_count;
}

// Detokenize
char* detokenize(const int* tokens, size_t num_tokens) {
    char* text = malloc(num_tokens + 1);
    if (!text) return NULL;
    
    for (size_t i = 0; i < num_tokens; i++) {
        text[i] = (char)tokens[i];
    }
    text[num_tokens] = '\0';
    
    return text;
}

// Generate response
char* llm_generate(llm_model_t* model, const char* input, size_t max_length) {
    if (!model || !input) return NULL;
    
    // Tokenize input
    int* input_tokens = malloc(strlen(input) * sizeof(int));
    if (!input_tokens) return NULL;
    
    size_t input_len = tokenize(input, input_tokens, strlen(input));
    
    // Create input tensor
    tensor_t* input_tensor = tensor_alloc(input_len, model->config.hidden_dim);
    if (!input_tensor) {
        free(input_tokens);
        return NULL;
    }
    
    // Get embeddings
    for (size_t i = 0; i < input_len; i++) {
        int token = input_tokens[i];
        // Ensure token is within valid range
        if (token < 0) token = 0;
        if (token >= model->config.vocab_size) token = model->config.vocab_size - 1;
        
        for (size_t j = 0; j < model->config.hidden_dim; j++) {
            input_tensor->data[i * model->config.hidden_dim + j] = 
                model->token_embeddings->data[token * model->config.hidden_dim + j];
        }
    }
    
    // Add position embeddings
    for (size_t i = 0; i < input_len; i++) {
        for (size_t j = 0; j < model->config.hidden_dim; j++) {
            input_tensor->data[i * model->config.hidden_dim + j] += 
                model->position_embeddings->data[i * model->config.hidden_dim + j];
        }
    }
    
    // Forward pass through transformer layers
    tensor_t* current = input_tensor;
    for (size_t layer = 0; layer < model->config.num_layers; layer++) {
        tensor_t* next = transformer_block_forward(current, 
                                                 model->attention_weights[layer],
                                                 model->feed_forward_weights[layer],
                                                 model->layer_norms[layer],
                                                 model->layer_norms[layer]);
        if (!next) {
            tensor_free(current);
            free(input_tokens);
            return NULL;
        }
        
        if (current != input_tensor) {
            tensor_free(current);
        }
        current = next;
    }
    
    // Generate AI response instead of template selection
    char* response = generate_ai_response(model, current, max_length);
    
    // Cleanup
    free(input_tokens);
    tensor_free(current);
    
    return response;
}
