#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

/**
 * @brief 2D tensor (matrix) structure for neural network operations
 * 
 * This structure represents a 2D tensor with float data stored in row-major order.
 * Memory layout: tensor->data[i * cols + j] = element at row i, column j
 */
typedef struct {
    float* data;        ///< Raw data array in row-major order
    size_t rows;        ///< Number of rows
    size_t cols;        ///< Number of columns
} tensor_t;

// ============================================================================
// MEMORY MANAGEMENT
// ============================================================================

/**
 * @brief Allocate a new tensor with specified dimensions
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to allocated tensor, or NULL on failure
 * 
 * @note Uses 64-byte aligned allocation for SIMD performance
 */
tensor_t* tensor_alloc(size_t rows, size_t cols);

/**
 * @brief Deallocate tensor and free memory
 * 
 * @param t Tensor to deallocate (can be NULL)
 * 
 * @note Handles NULL pointers gracefully
 */
void tensor_free(tensor_t* t);

// ============================================================================
// INITIALIZATION FUNCTIONS
// ============================================================================

/**
 * @brief Fill tensor with random values from uniform distribution [0, 1)
 * 
 * @param t Tensor to fill
 * 
 * @note Use for weight initialization in neural networks
 */
void tensor_fill_random(tensor_t* t);

/**
 * @brief Fill tensor with zeros
 * 
 * @param t Tensor to fill
 * 
 * @note Uses memset for efficiency
 */
void tensor_fill_zeros(tensor_t* t);

/**
 * @brief Fill tensor with ones
 * 
 * @param t Tensor to fill
 * 
 * @note Use for bias initialization and layer normalization parameters
 */
void tensor_fill_ones(tensor_t* t);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Copy data from source to destination tensor
 * 
 * @param dst Destination tensor
 * @param src Source tensor
 * 
 * @note Checks dimension compatibility before copying
 */
void tensor_copy(tensor_t* dst, const tensor_t* src);

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

/**
 * @brief Matrix multiplication C = A × B
 * 
 * @param a Left matrix
 * @param b Right matrix
 * @return Result matrix, or NULL on failure
 * 
 * @note Dimensions must match: A[rows_a, cols_a] × B[cols_a, cols_b] → C[rows_a, cols_b]
 * @note Returns NULL if a->cols != b->rows
 */
tensor_t* tensor_matmul(const tensor_t* a, const tensor_t* b);

/**
 * @brief Element-wise addition: dst += src
 * 
 * @param dst Destination tensor (modified in-place)
 * @param src Source tensor
 * 
 * @note Dimensions must match exactly
 */
void tensor_add(tensor_t* dst, const tensor_t* src);

/**
 * @brief Scale all elements by a factor
 * 
 * @param t Tensor to scale (modified in-place)
 * @param scale Scaling factor
 */
void tensor_scale(tensor_t* t, float scale);

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

/**
 * @brief Apply ReLU activation: f(x) = max(0, x)
 * 
 * @param t Tensor to modify (in-place)
 * 
 * @note Use for feed-forward network activations
 */
void tensor_relu(tensor_t* t);

/**
 * @brief Apply GELU activation: f(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
 * 
 * @param t Tensor to modify (in-place)
 * 
 * @note Modern transformer activations (better than ReLU)
 */
void tensor_gelu(tensor_t* t);

// ============================================================================
// NORMALIZATION FUNCTIONS
// ============================================================================

/**
 * @brief Apply softmax activation to convert logits to probabilities
 * 
 * @param t Tensor to modify (in-place)
 * 
 * @note Formula: softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
 * @note Uses two-pass algorithm for numerical stability
 */
void tensor_softmax(tensor_t* t);

/**
 * @brief Apply layer normalization across the last dimension
 * 
 * @param t Tensor to normalize (in-place)
 * @param epsilon Small value for numerical stability
 * 
 * @note Formula: (x - μ) / √(σ² + ε)
 * @note Use to stabilize transformer training
 */
void tensor_layer_norm(tensor_t* t, float epsilon);

// ============================================================================
// VOCABULARY PROJECTION (for LLM training)
// ============================================================================

/**
 * @brief Project hidden dimensions to vocabulary space
 * 
 * @param input Input tensor [seq_len, hidden_dim]
 * @param weights Projection weights [hidden_dim, vocab_size]
 * @return Projected tensor [seq_len, vocab_size], or NULL on failure
 * 
 * @note Use for language modeling and token prediction
 */
tensor_t* tensor_vocab_projection(const tensor_t* input, const tensor_t* weights);

#endif // TENSOR_H
