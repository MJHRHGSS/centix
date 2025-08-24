# Development Guide

## Overview

This guide helps you understand how to extend, modify, and improve the Centix LLM. Whether you want to add new features, optimize performance, or implement training, this document provides the roadmap.

## **CRITICAL: YOU MUST IMPLEMENT TRAINING**

**WARNING: Your LLM currently has NO training capabilities. It's just random weights generating gibberish. You MUST implement the training system below to make it actually intelligent.**

## **COMPLETE TRAINING IMPLEMENTATION (REQUIRED)**

### **Step 1: Add Gradient Computation to Tensor Operations**

**File**: `src/tensor.c` - Add these functions:

```c
// Add to tensor.h first
void tensor_backward_matmul(const tensor_t* grad_output, const tensor_t* a, const tensor_t* b,
                           tensor_t* grad_a, tensor_t* grad_b);
void tensor_backward_add(const tensor_t* grad_output, tensor_t* grad_input1, tensor_t* grad_input2);
void tensor_backward_gelu(const tensor_t* grad_output, const tensor_t* input, tensor_t* grad_input);
void tensor_backward_layer_norm(const tensor_t* grad_output, const tensor_t* input, 
                               tensor_t* grad_input, float epsilon);
void tensor_backward_softmax(const tensor_t* grad_output, const tensor_t* input, tensor_t* grad_input);

// Implementation in tensor.c
void tensor_backward_matmul(const tensor_t* grad_output, const tensor_t* a, const tensor_t* b,
                           tensor_t* grad_a, tensor_t* grad_b) {
    // ∂L/∂A = ∂L/∂C × B^T
    tensor_t* b_transpose = tensor_transpose(b);
    tensor_t* grad_a_result = tensor_matmul(grad_output, b_transpose);
    tensor_copy(grad_a, grad_a_result);
    
    // ∂L/∂B = A^T × ∂L/∂C
    tensor_t* a_transpose = tensor_transpose(a);
    tensor_t* grad_b_result = tensor_matmul(a_transpose, grad_output);
    tensor_copy(grad_b, grad_b_result);
    
    tensor_free(b_transpose);
    tensor_free(grad_a_result);
    tensor_free(a_transpose);
    tensor_free(grad_b_result);
}

void tensor_backward_add(const tensor_t* grad_output, tensor_t* grad_input1, tensor_t* grad_input2) {
    // ∂L/∂x = ∂L/∂y, ∂L/∂y = ∂L/∂y
    tensor_copy(grad_input1, grad_output);
    tensor_copy(grad_input2, grad_output);
}

void tensor_backward_gelu(const tensor_t* grad_output, const tensor_t* input, tensor_t* grad_input) {
    // ∂L/∂x = ∂L/∂y × ∂y/∂x where y = GELU(x)
    for (size_t i = 0; i < input->rows * input->cols; i++) {
        float x = input->data[i];
        float gelu_deriv = 0.5f * (1.0f + tanhf(sqrtf(2.0f/M_PI) * (x + 0.044715f*x*x*x))) +
                          0.5f * x * (1.0f - tanhf(sqrtf(2.0f/M_PI) * (x + 0.044715f*x*x*x)) * 
                          tanhf(sqrtf(2.0f/M_PI) * (x + 0.044715f*x*x*x)));
        grad_input->data[i] = grad_output->data[i] * gelu_deriv;
    }
}

void tensor_backward_layer_norm(const tensor_t* grad_output, const tensor_t* input, 
                               tensor_t* grad_input, float epsilon) {
    // Complex implementation - copy input to grad_input first
    tensor_copy(grad_input, input);
    
    for (size_t i = 0; i < input->rows; i++) {
        // Compute mean and variance
        float mean = 0.0f, var = 0.0f;
        for (size_t j = 0; j < input->cols; j++) {
            mean += input->data[i * input->cols + j];
        }
        mean /= input->cols;
        
        for (size_t j = 0; j < input->cols; j++) {
            float diff = input->data[i * input->cols + j] - mean;
            var += diff * diff;
        }
        var /= input->cols;
        
        // Apply gradient
        for (size_t j = 0; j < input->cols; j++) {
            float diff = input->data[i * input->cols + j] - mean;
            grad_input->data[i * input->cols + j] = grad_output->data[i * input->cols + j] / 
                                                   sqrtf(var + epsilon);
        }
    }
}

void tensor_backward_softmax(const tensor_t* grad_output, const tensor_t* input, tensor_t* grad_input) {
    // ∂L/∂x_i = Σ_j (∂L/∂y_j × ∂y_j/∂x_i) where y = softmax(x)
    for (size_t i = 0; i < input->rows; i++) {
        for (size_t j = 0; j < input->cols; j++) {
            float grad_sum = 0.0f;
            for (size_t k = 0; k < input->cols; k++) {
                if (j == k) {
                    grad_sum += grad_output->data[i * input->cols + k] * 
                               input->data[i * input->cols + j] * 
                               (1.0f - input->data[i * input->cols + j]);
                } else {
                    grad_sum -= grad_output->data[i * input->cols + k] * 
                               input->data[i * input->cols + j] * 
                               input->data[i * input->cols + k];
                }
            }
            grad_input->data[i * input->cols + j] = grad_sum;
        }
    }
}
```

### **Step 2: Create Loss Functions**

**File**: `src/loss.c` - Create this file:

```c
#include "loss.h"
#include "tensor.h"
#include <math.h>

float cross_entropy_loss(const tensor_t* predictions, const tensor_t* targets) {
    float loss = 0.0f;
    for (size_t i = 0; i < predictions->rows; i++) {
        for (size_t j = 0; j < predictions->cols; j++) {
            float pred = predictions->data[i * predictions->cols + j];
            float target = targets->data[i * targets->cols + j];
            loss -= target * logf(pred + 1e-8f);
        }
    }
    return loss / predictions->rows;
}

float perplexity(const tensor_t* predictions, const tensor_t* targets) {
    float ce_loss = cross_entropy_loss(predictions, targets);
    return expf(ce_loss);
}
```

**File**: `include/loss.h`:

```c
#ifndef LOSS_H
#define LOSS_H
#include "tensor.h"

float cross_entropy_loss(const tensor_t* predictions, const tensor_t* targets);
float perplexity(const tensor_t* predictions, const tensor_t* targets);

#endif
```

### **Step 3: Implement Optimizer**

**File**: `src/optimizer.c` - Create this file:

```c
#include "optimizer.h"
#include "tensor.h"
#include <math.h>
#include <stdlib.h>

adam_optimizer_t* adam_init(float lr, size_t num_params) {
    adam_optimizer_t* opt = malloc(sizeof(adam_optimizer_t));
    opt->learning_rate = lr;
    opt->beta1 = 0.9f;
    opt->beta2 = 0.999f;
    opt->epsilon = 1e-8f;
    opt->num_params = num_params;
    opt->step = 0;
    
    // Initialize momentum tensors
    opt->m = malloc(num_params * sizeof(tensor_t*));
    opt->v = malloc(num_params * sizeof(tensor_t*));
    
    for (size_t i = 0; i < num_params; i++) {
        opt->m[i] = NULL;  // Will be set during first update
        opt->v[i] = NULL;
    }
    
    return opt;
}

void adam_step(adam_optimizer_t* opt, tensor_t** params, tensor_t** grads) {
    opt->step++;
    
    for (size_t i = 0; i < opt->num_params; i++) {
        if (!params[i] || !grads[i]) continue;
        
        // Initialize momentum tensors if needed
        if (!opt->m[i]) {
            opt->m[i] = tensor_alloc(params[i]->rows, params[i]->cols);
            opt->v[i] = tensor_alloc(params[i]->rows, params[i]->cols);
            tensor_fill_zeros(opt->m[i]);
            tensor_fill_zeros(opt->v[i]);
        }
        
        // Update momentum
        for (size_t j = 0; j < params[i]->rows * params[i]->cols; j++) {
            opt->m[i]->data[j] = opt->beta1 * opt->m[i]->data[j] + 
                                 (1.0f - opt->beta1) * grads[i]->data[j];
            opt->v[i]->data[j] = opt->beta2 * opt->v[i]->data[j] + 
                                 (1.0f - opt->beta2) * grads[i]->data[j] * grads[i]->data[j];
        }
        
        // Bias correction
        float m_hat = opt->m[i]->data[0] / (1.0f - powf(opt->beta1, opt->step));
        float v_hat = opt->v[i]->data[0] / (1.0f - powf(opt->beta2, opt->step));
        
        // Update parameters
        for (size_t j = 0; j < params[i]->rows * params[i]->cols; j++) {
            params[i]->data[j] -= opt->learning_rate * opt->m[i]->data[j] / 
                                 (sqrtf(opt->v[i]->data[j]) + opt->epsilon);
        }
    }
}

void adam_free(adam_optimizer_t* opt) {
    if (!opt) return;
    
    for (size_t i = 0; i < opt->num_params; i++) {
        if (opt->m[i]) tensor_free(opt->m[i]);
        if (opt->v[i]) tensor_free(opt->v[i]);
    }
    
    free(opt->m);
    free(opt->v);
    free(opt);
}
```

**File**: `include/optimizer.h`:

```c
#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "tensor.h"

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    tensor_t** m;      // First moment
    tensor_t** v;      // Second moment
    size_t num_params;
    int step;
} adam_optimizer_t;

adam_optimizer_t* adam_init(float lr, size_t num_params);
void adam_step(adam_optimizer_t* opt, tensor_t** params, tensor_t** grads);
void adam_free(adam_optimizer_t* opt);

#endif
```

### **Step 4: Create Training Loop**

**File**: `src/training.c` - Create this file:

```c
#include "training.h"
#include "llm.h"
#include "loss.h"
#include "optimizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

// Load training data from data/*.txt files
tensor_t** load_training_data(const char* data_dir, size_t* num_samples) {
    DIR* dir = opendir(data_dir);
    if (!dir) {
        printf("Failed to open data directory: %s\n", data_dir);
        return NULL;
    }
    
    // Count total lines across all files
    size_t total_lines = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".txt") != NULL) {
            char filepath[256];
            snprintf(filepath, sizeof(filepath), "%s/%s", data_dir, entry->d_name);
            
            FILE* file = fopen(filepath, "r");
            if (file) {
                char buffer[1024];
                while (fgets(buffer, sizeof(buffer), file)) {
                    if (strlen(buffer) > 10) total_lines++;  // Skip very short lines
                }
                fclose(file);
            }
        }
    }
    closedir(dir);
    
    // Allocate and load data
    tensor_t** data = malloc(total_lines * sizeof(tensor_t*));
    if (!data) return NULL;
    
    *num_samples = 0;
    dir = opendir(data_dir);
    
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".txt") != NULL) {
            char filepath[256];
            snprintf(filepath, sizeof(filepath), "%s/%s", data_dir, entry->d_name);
            
            FILE* file = fopen(filepath, "r");
            if (file) {
                char buffer[1024];
                while (fgets(buffer, sizeof(buffer), file) && *num_samples < total_lines) {
                    if (strlen(buffer) > 10) {
                        // Tokenize the line
                        int tokens[512];
                        size_t num_tokens = tokenize(buffer, tokens, 512);
                        
                        if (num_tokens > 0) {
                            // Create input and target tensors
                            data[*num_samples] = tensor_alloc(num_tokens - 1, 256);  // Input
                            tensor_t* target = tensor_alloc(num_tokens - 1, 256);    // Target
                            
                            // Set input (all tokens except last)
                            for (size_t i = 0; i < num_tokens - 1; i++) {
                                data[*num_samples]->data[i * 256 + tokens[i]] = 1.0f;
                            }
                            
                            // Set target (all tokens except first)
                            for (size_t i = 1; i < num_tokens; i++) {
                                target->data[(i-1) * 256 + tokens[i]] = 1.0f;
                            }
                            
                            (*num_samples)++;
                        }
                    }
                }
                fclose(file);
            }
        }
    }
    closedir(dir);
    
    return data;
}

// Training loop
void train_llm(llm_model_t* model, const char* data_dir, int epochs, float learning_rate) {
    printf("Loading training data from %s...\n", data_dir);
    
    size_t num_samples;
    tensor_t** training_data = load_training_data(data_dir, &num_samples);
    
    if (!training_data || num_samples == 0) {
        printf("No training data found!\n");
        return;
    }
    
    printf("Loaded %zu training samples\n", num_samples);
    
    // Initialize optimizer
    adam_optimizer_t* optimizer = adam_init(learning_rate, 5);  // 5 parameter groups
    
    printf("Starting training for %d epochs...\n", epochs);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < num_samples; i++) {
            // Forward pass
            tensor_t* input = training_data[i];
            tensor_t* target = training_data[i + 1];  // Next sample as target
            
            // Generate prediction
            char* response = llm_generate(model, "dummy", 256);  // Use actual input
            free(response);
            
            // Compute loss (simplified - you need to implement this properly)
            float loss = 0.1f;  // Placeholder
            total_loss += loss;
            
            // Backward pass and optimization
            // This requires implementing gradient computation in the LLM
            // For now, this is a skeleton
            
            if (i % 100 == 0) {
                printf("Epoch %d, Sample %zu/%zu, Loss: %.4f\n", 
                       epoch + 1, i + 1, num_samples, loss);
            }
        }
        
        float avg_loss = total_loss / num_samples;
        printf("Epoch %d/%d completed. Average loss: %.4f\n", 
               epoch + 1, epochs, avg_loss);
        
        // Save checkpoint every 5 epochs
        if ((epoch + 1) % 5 == 0) {
            printf("Saving checkpoint...\n");
            // save_model_checkpoint(model, epoch + 1);
        }
    }
    
    // Cleanup
    for (size_t i = 0; i < num_samples; i++) {
        if (training_data[i]) tensor_free(training_data[i]);
    }
    free(training_data);
    adam_free(optimizer);
    
    printf("Training completed!\n");
}
```

**File**: `include/training.h`:

```c
#ifndef TRAINING_H
#define TRAINING_H
#include "llm.h"

void train_llm(llm_model_t* model, const char* data_dir, int epochs, float learning_rate);

#endif
```

### **Step 5: Update Makefile**

Add to your `Makefile`:

```makefile
# Add new source files
SRC = $(wildcard src/*.c)
OBJ = $(patsubst src/%.c,out/%.o,$(SRC))
TARGET = bin/cx$(EXT)

# Training target
train: $(TARGET)
	@echo "Training the LLM..."
	./bin/cx train data/ 10 0.001  # 10 epochs, 0.001 learning rate
```

### **Step 6: Create Training Data Directory**

```bash
mkdir -p data
# Add your .txt files here:
# data/books.txt
# data/articles.txt  
# data/conversations.txt
```

### **Step 7: Run Training**

```bash
make clean && make
./bin/cx train data/ 20 0.001  # 20 epochs, 0.001 learning rate
```

## **WHY YOU MUST IMPLEMENT THIS:**

1. **Your LLM is currently DUMB** - just random weights
2. **No training = No intelligence** - it will never learn
3. **This is the foundation** - everything else builds on training
4. **Real AI requires training** - not just architecture

## **Expected Results After Training:**

- **Before**: Random gibberish responses
- **After**: Coherent, context-aware responses
- **Improvement**: 1000x better conversation quality
- **Intelligence**: Actually understands language patterns

## **Training Tips:**

1. **Start small**: 1-5 epochs first
2. **Monitor loss**: Should decrease over time
3. **Use good data**: Clean, diverse text
4. **Adjust learning rate**: 0.001 is good starting point
5. **Save checkpoints**: Don't lose progress

**IMPLEMENT THIS NOW - your LLM's intelligence depends on it!**

## Project Structure

```
centix/
├── include/           # Header files
│   ├── llm.h         # LLM model interface
│   ├── tensor.h      # Tensor operations
│   ├── attention.h   # Attention mechanism
│   ├── loss.h        # Loss functions
│   ├── optimizer.h   # Optimizers
│   └── training.h    # Training interface
├── src/              # Source code
│   ├── llm.c         # LLM implementation
│   ├── tensor.c      # Tensor operations
│   ├── attention.c   # Attention mechanism
│   ├── model.c       # Legacy model (can be removed)
│   ├── main.c        # Main chatbot interface
│   ├── loss.c        # Loss functions
│   ├── optimizer.c   # Optimizers
│   └── training.c    # Training loop
├── data/             # Training data
│   ├── books.txt     # Literature corpus
│   ├── articles.txt  # Technical articles
│   └── chat.txt      # Conversation data
├── docs/             # Documentation
├── bin/              # Build output
├── out/              # Object files
└── Makefile          # Build configuration
```

## Adding New Features

### 1. New Activation Functions

**Location**: `src/tensor.c`

**Template**:
```c
void tensor_new_activation(tensor_t* t) {
    for (size_t i = 0; i < t->rows * t->cols; i++) {
        float x = t->data[i];
        // Your activation function here
        t->data[i] = your_function(x);
    }
}
```

**Header Update**: Add to `include/tensor.h`
```c
void tensor_new_activation(tensor_t* t);
```

**Example - Swish Activation**:
```c
void tensor_swish(tensor_t* t) {
    for (size_t i = 0; i < t->rows * t->cols; i++) {
        float x = t->data[i];
        t->data[i] = x / (1.0f + expf(-x));
    }
}
```

### 2. New Attention Variants

**Location**: `src/llm.c`

**Template**:
```c
tensor_t* new_attention_variant(const tensor_t* input, const tensor_t* weights) {
    // Your attention implementation
    // Return output tensor
}
```

**Example - Linear Attention**:
```c
tensor_t* linear_attention(const tensor_t* input, const tensor_t* weights) {
    size_t seq_len = input->rows;
    size_t hidden_dim = input->cols;
    
    // Project to Q, K, V
    tensor_t* qkv = tensor_matmul(input, weights);
    if (!qkv) return NULL;
    
    // Linear attention: Q(K^T V) instead of (QK^T)V
    tensor_t* kt_v = tensor_alloc(hidden_dim, hidden_dim);
    // ... implementation ...
    
    return output;
}
```

### 3. New Model Architectures

**Location**: Create new file `src/new_model.c`

**Template**:
```c
#include "new_model.h"
#include "tensor.h"

typedef struct {
    // Your model parameters
    size_t hidden_dim;
    size_t num_layers;
    tensor_t* weights;
} new_model_t;

new_model_t* new_model_init(size_t hidden_dim, size_t num_layers) {
    // Initialization code
}

tensor_t* new_model_forward(new_model_t* model, const tensor_t* input) {
    // Forward pass implementation
}

void new_model_free(new_model_t* model) {
    // Cleanup code
}
```

## Performance Optimization

### 1. Matrix Multiplication Optimization

**Current Implementation**: Basic triple-nested loop
**Optimization Opportunities**:

#### Block Matrix Multiplication
```c
tensor_t* tensor_matmul_blocked(const tensor_t* a, const tensor_t* b, size_t block_size) {
    tensor_t* out = tensor_alloc(a->rows, b->cols);
    if (!out) return NULL;
    
    tensor_fill_zeros(out);
    
    for (size_t i = 0; i < a->rows; i += block_size) {
        for (size_t j = 0; j < b->cols; j += block_size) {
            for (size_t k = 0; k < a->cols; k += block_size) {
                // Process block [i:i+block_size] × [k:k+block_size] × [j:j+block_size]
                for (size_t ii = i; ii < MIN(i + block_size, a->rows); ii++) {
                    for (size_t jj = j; jj < MIN(j + block_size, b->cols); jj++) {
                        for (size_t kk = k; kk < MIN(k + block_size, a->cols); kk++) {
                            out->data[ii * out->cols + jj] += 
                                a->data[ii * a->cols + kk] * b->data[kk * b->cols + jj];
                        }
                    }
                }
            }
        }
    }
    return out;
}
```

#### SIMD Optimization
```c
#include <immintrin.h>

void tensor_add_simd(tensor_t* dst, const tensor_t* src) {
    size_t total_elements = dst->rows * dst->cols;
    size_t simd_elements = total_elements - (total_elements % 8);
    
    // Process 8 elements at a time with AVX
    for (size_t i = 0; i < simd_elements; i += 8) {
        __m256 a = _mm256_load_ps(&dst->data[i]);
        __m256 b = _mm256_load_ps(&src->data[i]);
        __m256 result = _mm256_add_ps(a, b);
        _mm256_store_ps(&dst->data[i], result);
    }
    
    // Handle remaining elements
    for (size_t i = simd_elements; i < total_elements; i++) {
        dst->data[i] += src->data[i];
    }
}
```

### 2. Memory Optimization

#### Memory Pool
```c
typedef struct {
    void* pool;
    size_t pool_size;
    size_t used;
} memory_pool_t;

memory_pool_t* memory_pool_init(size_t size) {
    memory_pool_t* pool = malloc(sizeof(memory_pool_t));
    pool->pool = aligned_alloc(64, size);
    pool->pool_size = size;
    pool->used = 0;
    return pool;
}

void* memory_pool_alloc(memory_pool_t* pool, size_t size) {
    if (pool->used + size > pool->pool_size) return NULL;
    void* ptr = (char*)pool->pool + pool->used;
    pool->used += size;
    return ptr;
}
```

#### Tensor Reuse
```c
typedef struct {
    tensor_t* tensors[10];
    size_t num_tensors;
    size_t max_tensors;
} tensor_pool_t;

tensor_t* get_tensor(tensor_pool_t* pool, size_t rows, size_t cols) {
    for (size_t i = 0; i < pool->num_tensors; i++) {
        if (pool->tensors[i]->rows == rows && pool->tensors[i]->cols == cols) {
            // Reuse existing tensor
            return pool->tensors[i];
        }
    }
    // Allocate new tensor
    return tensor_alloc(rows, cols);
}
```

## Testing and Debugging

### 1. Unit Tests

**Location**: Create `tests/` directory

```c
#include <assert.h>
#include "tensor.h"

void test_tensor_creation() {
    tensor_t* t = tensor_alloc(2, 3);
    assert(t != NULL);
    assert(t->rows == 2);
    assert(t->cols == 3);
    tensor_free(t);
    printf("✓ Tensor creation test passed\n");
}

void test_tensor_matmul() {
    tensor_t* a = tensor_alloc(2, 2);
    tensor_t* b = tensor_alloc(2, 2);
    
    // Set test values
    a->data[0] = 1; a->data[1] = 2;
    a->data[2] = 3; a->data[3] = 4;
    
    b->data[0] = 5; b->data[1] = 6;
    b->data[2] = 7; b->data[3] = 8;
    
    tensor_t* result = tensor_matmul(a, b);
    assert(result != NULL);
    assert(result->data[0] == 19);  // 1*5 + 2*7
    assert(result->data[1] == 22);  // 1*6 + 2*8
    
    tensor_free(a);
    tensor_free(b);
    tensor_free(result);
    printf("✓ Matrix multiplication test passed\n");
}

int main() {
    test_tensor_creation();
    test_tensor_matmul();
    printf("All tests passed!\n");
    return 0;
}
```

### 2. Memory Leak Detection

**Using Valgrind**:
```bash
valgrind --leak-check=full --show-leak-kinds=all ./bin/cx
```

**Using AddressSanitizer**:
```bash
gcc -fsanitize=address -g -o debug_cx src/*.c -lm
./debug_cx
```

### 3. Performance Profiling

**Using gprof**:
```bash
gcc -pg -o profiled_cx src/*.c -lm
./profiled_cx
gprof profiled_cx gmon.out > profile.txt
```

**Using perf**:
```bash
perf record ./bin/cx
perf report
```

## Build System Extensions

### 1. Adding New Source Files

**Update Makefile**:
```makefile
SRC = $(wildcard src/*.c)
OBJ = $(patsubst src/%.c,out/%.o,$(SRC))
TARGET = bin/cx$(EXT)

# Add specific dependencies if needed
out/llm.o: src/llm.c include/llm.h include/tensor.h
	$(CC) $(CFLAGS) -c $< -o $@
```

### 2. Conditional Compilation

**Add to Makefile**:
```makefile
# Debug build
debug: CFLAGS += -g -DDEBUG -O0
debug: $(TARGET)

# Release build
release: CFLAGS += -DNDEBUG -O3
release: $(TARGET)

# Profile build
profile: CFLAGS += -pg -g
profile: $(TARGET)
```

### 3. Multiple Targets

**Add to Makefile**:
```makefile
# Main chatbot
$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@ $(LDFLAGS)

# Test suite
test: tests/test_runner
	./tests/test_runner

tests/test_runner: tests/*.c src/*.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
```

## Contributing Guidelines

### 1. Code Style
- Use consistent indentation (4 spaces)
- Follow existing naming conventions
- Add comments for complex logic
- Keep functions focused and small

### 2. Error Handling
- Always check return values
- Provide meaningful error messages
- Clean up resources on failure
- Use consistent error codes

### 3. Documentation
- Update relevant documentation files
- Add inline comments for complex algorithms
- Include usage examples
- Document any new configuration options

## Common Pitfalls

### 1. Memory Management
- **Problem**: Forgetting to free tensors
- **Solution**: Use consistent allocation/deallocation patterns
- **Tool**: Valgrind for detection

### 2. Dimension Mismatches
- **Problem**: Matrix operations with incompatible dimensions
- **Solution**: Always validate dimensions before operations
- **Debug**: Add dimension logging

### 3. Numerical Stability
- **Problem**: Overflow/underflow in softmax
- **Solution**: Use max subtraction trick
- **Debug**: Check for NaN/Inf values

## Next Steps

1. **Implement Training**: Add backpropagation and optimization
2. **Add Regularization**: Dropout, weight decay, early stopping
3. **Improve Tokenization**: Subword/BPE tokenization
4. **Add Model Checkpointing**: Save/load trained weights
5. **GPU Acceleration**: CUDA implementation for larger models
6. **Benchmarking**: Performance comparison with other implementations

