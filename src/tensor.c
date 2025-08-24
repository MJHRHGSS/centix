#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

// Mathematical constants
#define M_PI 3.14159265358979323846

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Generate random float in range [0, 1)
 * 
 * @return Random float value
 */
static inline float generate_random_float() {
    return (float)rand() / (float)RAND_MAX;
}

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

tensor_t* tensor_alloc(size_t rows, size_t cols) {
    // Allocate tensor structure
    tensor_t* tensor = malloc(sizeof(tensor_t));
    if (!tensor) {
        return NULL;
    }
    
    // Set dimensions
    tensor->rows = rows;
    tensor->cols = cols;
    
    // Allocate data array with 64-byte alignment for SIMD performance
    tensor->data = aligned_alloc(64, sizeof(float) * rows * cols);
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }
    
    return tensor;
}

void tensor_free(tensor_t* tensor) {
    if (!tensor) {
        return;
    }
    
    // Free data array first
    if (tensor->data) {
        free(tensor->data);
    }
    
    // Free tensor structure
    free(tensor);
}

// ============================================================================
// INITIALIZATION FUNCTIONS
// ============================================================================

void tensor_fill_random(tensor_t* tensor) {
    if (!tensor || !tensor->data) {
        return;
    }
    
    size_t total_elements = tensor->rows * tensor->cols;
    for (size_t i = 0; i < total_elements; i++) {
        tensor->data[i] = generate_random_float();
    }
}

void tensor_fill_zeros(tensor_t* tensor) {
    if (!tensor || !tensor->data) {
        return;
    }
    
    size_t total_elements = tensor->rows * tensor->cols;
    memset(tensor->data, 0, sizeof(float) * total_elements);
}

void tensor_fill_ones(tensor_t* tensor) {
    if (!tensor || !tensor->data) {
        return;
    }
    
    size_t total_elements = tensor->rows * tensor->cols;
    for (size_t i = 0; i < total_elements; i++) {
        tensor->data[i] = 1.0f;
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void tensor_copy(tensor_t* destination, const tensor_t* source) {
    if (!destination || !source || !destination->data || !source->data) {
        return;
    }
    
    // Check dimension compatibility
    if (destination->rows != source->rows || destination->cols != source->cols) {
        return;
    }
    
    size_t total_elements = source->rows * source->cols;
    memcpy(destination->data, source->data, sizeof(float) * total_elements);
}

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

tensor_t* tensor_matmul(const tensor_t* matrix_a, const tensor_t* matrix_b) {
    if (!matrix_a || !matrix_b || !matrix_a->data || !matrix_b->data) {
        return NULL;
    }
    
    // Check dimension compatibility for matrix multiplication
    if (matrix_a->cols != matrix_b->rows) {
        return NULL;
    }
    
    // Allocate result tensor
    tensor_t* result = tensor_alloc(matrix_a->rows, matrix_b->cols);
    if (!result) {
        return NULL;
    }
    
    // Initialize result to zero
    tensor_fill_zeros(result);
    
    // Perform matrix multiplication: C[i,j] = Σ A[i,k] × B[k,j]
    for (size_t row = 0; row < matrix_a->rows; row++) {
        for (size_t col = 0; col < matrix_b->cols; col++) {
            float sum = 0.0f;
            
            for (size_t k = 0; k < matrix_a->cols; k++) {
                sum += matrix_a->data[row * matrix_a->cols + k] * 
                       matrix_b->data[k * matrix_b->cols + col];
            }
            
            result->data[row * result->cols + col] = sum;
        }
    }
    
    return result;
}

void tensor_add(tensor_t* destination, const tensor_t* source) {
    if (!destination || !source || !destination->data || !source->data) {
        return;
    }
    
    // Check dimension compatibility
    if (destination->rows != source->rows || destination->cols != source->cols) {
        return;
    }
    
    size_t total_elements = destination->rows * destination->cols;
    for (size_t i = 0; i < total_elements; i++) {
        destination->data[i] += source->data[i];
    }
}

void tensor_scale(tensor_t* tensor, float scale_factor) {
    if (!tensor || !tensor->data) {
        return;
    }
    
    size_t total_elements = tensor->rows * tensor->cols;
    for (size_t i = 0; i < total_elements; i++) {
        tensor->data[i] *= scale_factor;
    }
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

void tensor_relu(tensor_t* tensor) {
    if (!tensor || !tensor->data) {
        return;
    }
    
    size_t total_elements = tensor->rows * tensor->cols;
    for (size_t i = 0; i < total_elements; i++) {
        if (tensor->data[i] < 0.0f) {
            tensor->data[i] = 0.0f;
        }
    }
}

void tensor_gelu(tensor_t* tensor) {
    if (!tensor || !tensor->data) {
        return;
    }
    
    size_t total_elements = tensor->rows * tensor->cols;
    for (size_t i = 0; i < total_elements; i++) {
        float x = tensor->data[i];
        
        // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        float x_cubed = x * x * x;
        float inner_term = x + 0.044715f * x_cubed;
        float scaling_factor = sqrtf(2.0f / M_PI);
        
        tensor->data[i] = 0.5f * x * (1.0f + tanhf(scaling_factor * inner_term));
    }
}

// ============================================================================
// NORMALIZATION FUNCTIONS
// ============================================================================

void tensor_softmax(tensor_t* tensor) {
    if (!tensor || !tensor->data) {
        return;
    }
    
    // Process each row independently
    for (size_t row = 0; row < tensor->rows; row++) {
        // Find maximum value for numerical stability
        float max_value = tensor->data[row * tensor->cols];
        for (size_t col = 1; col < tensor->cols; col++) {
            float current_value = tensor->data[row * tensor->cols + col];
            if (current_value > max_value) {
                max_value = current_value;
            }
        }
        
        // Compute exponentials with stability offset
        float sum = 0.0f;
        for (size_t col = 0; col < tensor->cols; col++) {
            float stable_exponent = tensor->data[row * tensor->cols + col] - max_value;
            tensor->data[row * tensor->cols + col] = expf(stable_exponent);
            sum += tensor->data[row * tensor->cols + col];
        }
        
        // Normalize to get probabilities
        for (size_t col = 0; col < tensor->cols; col++) {
            tensor->data[row * tensor->cols + col] /= sum;
        }
    }
}

void tensor_layer_norm(tensor_t* tensor, float epsilon) {
    if (!tensor || !tensor->data) {
        return;
    }
    
    // Process each row independently
    for (size_t row = 0; row < tensor->rows; row++) {
        // Compute mean across columns
        float mean = 0.0f;
        for (size_t col = 0; col < tensor->cols; col++) {
            mean += tensor->data[row * tensor->cols + col];
        }
        mean /= tensor->cols;
        
        // Compute variance across columns
        float variance = 0.0f;
        for (size_t col = 0; col < tensor->cols; col++) {
            float deviation = tensor->data[row * tensor->cols + col] - mean;
            variance += deviation * deviation;
        }
        variance /= tensor->cols;
        
        // Apply normalization: (x - μ) / √(σ² + ε)
        for (size_t col = 0; col < tensor->cols; col++) {
            float deviation = tensor->data[row * tensor->cols + col] - mean;
            tensor->data[row * tensor->cols + col] = deviation / sqrtf(variance + epsilon);
        }
    }
}

// ============================================================================
// VOCABULARY PROJECTION (for LLM training)
// ============================================================================

tensor_t* tensor_vocab_projection(const tensor_t* input, const tensor_t* weights) {
    if (!input || !weights || !input->data || !weights->data) {
        return NULL;
    }
    
    // Check dimension compatibility
    if (input->cols != weights->rows) {
        return NULL;
    }
    
    // Allocate result tensor: [input_rows, weights_cols]
    tensor_t* result = tensor_alloc(input->rows, weights->cols);
    if (!result) {
        return NULL;
    }
    
    // Initialize result to zero
    tensor_fill_zeros(result);
    
    // Perform projection: output[i,j] = Σ input[i,k] × weights[k,j]
    for (size_t row = 0; row < input->rows; row++) {
        for (size_t col = 0; col < weights->cols; col++) {
            float sum = 0.0f;
            
            for (size_t k = 0; k < input->cols; k++) {
                sum += input->data[row * input->cols + k] * 
                       weights->data[k * weights->cols + col];
            }
            
            result->data[row * result->cols + col] = sum;
        }
    }
    
    return result;
}
