# Tensor Operations Guide

## Overview

The tensor library provides the mathematical foundation for the LLM. This document explains all the tensor operations, their implementations, and how they're used in the transformer architecture.

## Tensor Structure

**File**: `include/tensor.h` (lines 4-8)

```c
typedef struct {
    float* data;        // Raw data array
    size_t rows;        // Number of rows
    size_t cols;        // Number of columns
} tensor_t;
```

**Memory Layout**: Row-major order
```
tensor->data[i * cols + j] = element at row i, column j
```

## Core Operations

### 1. Memory Management

#### `tensor_alloc(rows, cols)`
- **Purpose**: Allocate a new tensor with specified dimensions
- **Memory**: Uses aligned allocation (64-byte alignment) for performance
- **Returns**: Pointer to allocated tensor or NULL on failure

```c
tensor_t* tensor_alloc(size_t rows, size_t cols) {
    tensor_t* t = malloc(sizeof(tensor_t));
    if (!t) return NULL;
    t->rows = rows;
    t->cols = cols;
    t->data = aligned_alloc(64, sizeof(float) * rows * cols);
    if (!t->data) { free(t); return NULL; }
    return t;
}
```

#### `tensor_free(tensor)`
- **Purpose**: Deallocate tensor and free memory
- **Safety**: Handles NULL pointers gracefully

### 2. Initialization Functions

#### `tensor_fill_random(tensor)`
- **Purpose**: Fill tensor with random values from [0, 1)
- **Use Case**: Weight initialization for neural networks

#### `tensor_fill_zeros(tensor)`
- **Purpose**: Fill tensor with zeros
- **Implementation**: Uses `memset` for efficiency
- **Use Case**: Output initialization before accumulation

#### `tensor_fill_ones(tensor)`
- **Purpose**: Fill tensor with ones
- **Use Case**: Bias initialization, layer normalization parameters

### 3. Matrix Operations

#### `tensor_matmul(a, b)`
- **Purpose**: Matrix multiplication C = A × B
- **Dimensions**: A[rows_a, cols_a] × B[cols_a, cols_b] → C[rows_a, cols_b]
- **Implementation**: Triple-nested loop with accumulation

```c
tensor_t* tensor_matmul(const tensor_t* a, const tensor_t* b) {
    if (a->cols != b->rows) return NULL;  // Dimension check
    
    tensor_t* out = tensor_alloc(a->rows, b->cols);
    if (!out) return NULL;
    
    tensor_fill_zeros(out);  // Initialize output
    
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            for (size_t k = 0; k < a->cols; k++) {
                out->data[i * out->cols + j] += 
                    a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }
    return out;
}
```

**Performance**: O(rows × cols_a × cols_b) time complexity

#### `tensor_add(dst, src)`
- **Purpose**: Element-wise addition: dst += src
- **Dimensions**: Must match exactly
- **Implementation**: Direct element addition

#### `tensor_scale(tensor, scale)`
- **Purpose**: Scale all elements by a factor
- **Use Case**: Attention scaling, gradient scaling

### 4. Activation Functions

#### `tensor_relu(tensor)`
- **Formula**: f(x) = max(0, x)
- **Use Case**: Feed-forward network activations
- **Implementation**: Simple conditional assignment

```c
void tensor_relu(tensor_t* t) {
    for (size_t i = 0; i < t->rows * t->cols; i++) {
        if (t->data[i] < 0) t->data[i] = 0;
    }
}
```

#### `tensor_gelu(tensor)`
- **Formula**: f(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
- **Use Case**: Modern transformer activations (better than ReLU)
- **Implementation**: Direct mathematical formula

```c
void tensor_gelu(tensor_t* t) {
    for (size_t i = 0; i < t->rows * t->cols; i++) {
        float x = t->data[i];
        t->data[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f/M_PI) * (x + 0.044715f*x*x*x)));
    }
}
```

### 5. Normalization Functions

#### `tensor_softmax(tensor)`
- **Purpose**: Convert logits to probability distribution
- **Formula**: softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
- **Implementation**: Two-pass algorithm for numerical stability

```c
void tensor_softmax(tensor_t* t) {
    for (size_t i = 0; i < t->rows; i++) {
        // Find maximum value for numerical stability
        float max_val = t->data[i * t->cols];
        for (size_t j = 1; j < t->cols; j++) {
            if (t->data[i * t->cols + j] > max_val) 
                max_val = t->data[i * t->cols + j];
        }
        
        // Compute exponentials
        float sum = 0.0f;
        for (size_t j = 0; j < t->cols; j++) {
            t->data[i * t->cols + j] = expf(t->data[i * t->cols + j] - max_val);
            sum += t->data[i * t->cols + j];
        }
        
        // Normalize to probabilities
        for (size_t j = 0; j < t->cols; j++) {
            t->data[i * t->cols + j] /= sum;
        }
    }
}
```

#### `tensor_layer_norm(tensor, epsilon)`
- **Purpose**: Normalize across the last dimension (hidden dimension)
- **Formula**: (x - μ) / √(σ² + ε)
- **Use Case**: Stabilize transformer training

```c
void tensor_layer_norm(tensor_t* t, float epsilon) {
    for (size_t i = 0; i < t->rows; i++) {
        // Compute mean
        float mean = 0.0f;
        for (size_t j = 0; j < t->cols; j++) {
            mean += t->data[i * t->cols + j];
        }
        mean /= t->cols;
        
        // Compute variance
        float var = 0.0f;
        for (size_t j = 0; j < t->cols; j++) {
            float diff = t->data[i * t->cols + j] - mean;
            var += diff * diff;
        }
        var /= t->cols;
        
        // Normalize
        for (size_t j = 0; j < t->cols; j++) {
            t->data[i * t->cols + j] = (t->data[i * t->cols + j] - mean) / sqrtf(var + epsilon);
        }
    }
}
```

### 6. Utility Functions

#### `tensor_copy(dst, src)`
- **Purpose**: Copy data from source to destination
- **Safety**: Checks dimension compatibility
- **Implementation**: Uses `memcpy` for efficiency

## Usage in LLM

### 1. Attention Mechanism
```c
// QKV projection
tensor_t* qkv = tensor_matmul(input, attention_weights);

// Attention scores
tensor_t* scores = tensor_alloc(seq_len, seq_len);
// ... compute scores ...
tensor_softmax(scores);

// Apply attention
tensor_t* output = tensor_matmul(scores, qkv);
```

### 2. Feed-Forward Network
```c
// First linear layer
tensor_t* hidden = tensor_matmul(input, ff_weights);
tensor_gelu(hidden);

// Second linear layer
tensor_t* output = tensor_matmul(hidden, proj_weights);
```

### 3. Residual Connections
```c
// Add residual connection
tensor_add(attn_output, input);
```

## Performance Optimizations

### 1. Memory Alignment
- 64-byte alignment for SIMD operations
- Cache-friendly memory access patterns

### 2. Compiler Optimizations
- `-O3 -Ofast` for aggressive optimization
- `-march=native` for CPU-specific optimizations
- `-funroll-loops` for loop unrolling

### 3. OpenMP Support
- Matrix operations can be parallelized
- `#pragma omp parallel for` directives ready

## Error Handling

All functions return NULL or handle errors gracefully:
- **Memory allocation failures**: Return NULL
- **Dimension mismatches**: Return NULL for matmul
- **NULL inputs**: Safe handling in most functions

## Future Enhancements

1. **BLAS Integration**: Use optimized BLAS libraries
2. **GPU Support**: CUDA/OpenCL implementations
3. **Sparse Operations**: Support for sparse matrices
4. **Quantization**: INT8/FP16 support for efficiency
5. **Automatic Differentiation**: Gradient computation for training

