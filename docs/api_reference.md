# API Reference

Complete reference for all public APIs in the Centix LLM system.

## **CRITICAL: IMPLEMENT TRAINING TO MAKE IT INTELLIGENT**

**Your LLM currently has NO training capabilities. It's just random weights generating gibberish.**

**To make it actually intelligent, you MUST implement the training system from [Development Guide](development_guide.md).**

## **Table of Contents**

- [Data Structures](#data-structures)
- [Tensor Operations](#tensor-operations)
- [LLM Model Functions](#llm-model-functions)
- [Tokenization Functions](#tokenization-functions)
- [Transformer Block Functions](#transformer-block-functions)
- [Error Handling](#error-handling)
- [Memory Management](#memory-management)
- [Performance Considerations](#performance-considerations)

## **Data Structures**

### `tensor_t` - 2D Tensor Structure

```c
typedef struct {
    float* data;        ///< Raw data array in row-major order
    size_t rows;        ///< Number of rows
    size_t cols;        ///< Number of columns
} tensor_t;
```

**Memory Layout**: `tensor->data[i * cols + j]` = element at row `i`, column `j`

**Example Usage**:
```c
tensor_t* matrix = tensor_alloc(3, 4);
matrix->data[1 * 4 + 2] = 5.0f;  // Set element at row 1, column 2
```

### `llm_config_t` - LLM Configuration

```c
typedef struct {
    size_t vocab_size;      ///< Vocabulary size (256 for ASCII)
    size_t hidden_dim;      ///< Hidden dimension (128)
    size_t num_layers;      ///< Number of transformer layers (4)
    size_t num_heads;       ///< Number of attention heads (8)
    size_t max_seq_len;     ///< Maximum sequence length (512)
} llm_config_t;
```

**Default Configuration**:
```c
llm_config_t config = {
    .vocab_size = 256,      // ASCII characters
    .hidden_dim = 128,      // Hidden dimension
    .num_layers = 4,        // Transformer layers
    .num_heads = 8,         // Attention heads
    .max_seq_len = 512      // Max sequence length
};
```

### `llm_model_t` - LLM Model State

```c
typedef struct {
    llm_config_t config;                    ///< Model configuration
    tensor_t* token_embeddings;             ///< Token embedding weights
    tensor_t* position_embeddings;          ///< Position embedding weights
    tensor_t* layer_norm_weights;           ///< Layer normalization weights
    tensor_t* attention_weights;            ///< Attention projection weights
    tensor_t* feed_forward_weights;        ///< Feed-forward network weights
    tensor_t* vocab_projection_weights;    ///< Vocabulary projection weights
} llm_model_t;
```

**Memory Layout**: All weights are stored as 2D tensors for efficient matrix operations.

## **Tensor Operations**

### Memory Management

#### `tensor_alloc(size_t rows, size_t cols)`

**Purpose**: Allocate a new tensor with specified dimensions

**Parameters**:
- `rows`: Number of rows
- `cols`: Number of columns

**Returns**: Pointer to allocated tensor, or `NULL` on failure

**Example**:
```c
tensor_t* matrix = tensor_alloc(10, 20);
if (!matrix) {
    printf("Failed to allocate tensor\n");
    return;
}
// Use matrix...
tensor_free(matrix);
```

**Notes**:
- Uses 64-byte aligned allocation for SIMD performance
- Always check return value for `NULL`

#### `tensor_free(tensor_t* tensor)`

**Purpose**: Deallocate tensor and free memory

**Parameters**:
- `tensor`: Tensor to deallocate (can be `NULL`)

**Example**:
```c
tensor_t* matrix = tensor_alloc(5, 5);
// ... use matrix ...
tensor_free(matrix);  // Safe to call even if matrix is NULL
```

**Notes**:
- Handles `NULL` pointers gracefully
- Frees both data array and tensor structure

### Initialization Functions

#### `tensor_fill_random(tensor_t* tensor)`

**Purpose**: Fill tensor with random values from uniform distribution [0, 1)

**Parameters**:
- `tensor`: Tensor to fill

**Example**:
```c
tensor_t* weights = tensor_alloc(128, 128);
tensor_fill_random(weights);  // Initialize with random weights
```

**Use Case**: Weight initialization in neural networks

#### `tensor_fill_zeros(tensor_t* tensor)`

**Purpose**: Fill tensor with zeros

**Parameters**:
- `tensor`: Tensor to fill

**Example**:
```c
tensor_t* bias = tensor_alloc(128, 1);
tensor_fill_zeros(bias);  // Initialize bias to zero
```

**Notes**: Uses `memset` for efficiency

#### `tensor_fill_ones(tensor_t* tensor)`

**Purpose**: Fill tensor with ones

**Parameters**:
- `tensor`: Tensor to fill

**Example**:
```c
tensor_t* scale = tensor_alloc(128, 1);
tensor_fill_ones(scale);  // Initialize scale to one
```

**Use Case**: Bias initialization and layer normalization parameters

### Utility Functions

#### `tensor_copy(tensor_t* dst, const tensor_t* src)`

**Purpose**: Copy data from source to destination tensor

**Parameters**:
- `dst`: Destination tensor
- `src`: Source tensor

**Example**:
```c
tensor_t* original = tensor_alloc(10, 10);
tensor_t* copy = tensor_alloc(10, 10);
tensor_fill_random(original);
tensor_copy(copy, original);  // Copy data
```

**Notes**:
- Checks dimension compatibility before copying
- Both tensors must have same dimensions

### Matrix Operations

#### `tensor_matmul(const tensor_t* a, const tensor_t* b)`

**Purpose**: Matrix multiplication C = A × B

**Parameters**:
- `a`: Left matrix
- `b`: Right matrix

**Returns**: Result matrix, or `NULL` on failure

**Example**:
```c
tensor_t* A = tensor_alloc(3, 4);
tensor_t* B = tensor_alloc(4, 2);
tensor_fill_random(A);
tensor_fill_random(B);

tensor_t* C = tensor_matmul(A, B);  // C = A × B
if (C) {
    // C has dimensions [3, 2]
    tensor_free(C);
}
```

**Dimension Requirements**: `a->cols == b->rows`

**Algorithm**: Standard triple-nested loop matrix multiplication

#### `tensor_add(tensor_t* dst, const tensor_t* src)`

**Purpose**: Element-wise addition: `dst += src`

**Parameters**:
- `dst`: Destination tensor (modified in-place)
- `src`: Source tensor

**Example**:
```c
tensor_t* result = tensor_alloc(3, 3);
tensor_t* increment = tensor_alloc(3, 3);
tensor_fill_ones(increment);

tensor_add(result, increment);  // result += increment
```

**Notes**:
- Dimensions must match exactly
- Modifies destination tensor in-place

#### `tensor_scale(tensor_t* tensor, float scale)`

**Purpose**: Scale all elements by a factor

**Parameters**:
- `tensor`: Tensor to scale (modified in-place)
- `scale`: Scaling factor

**Example**:
```c
tensor_t* weights = tensor_alloc(128, 128);
tensor_fill_random(weights);
tensor_scale(weights, 0.1f);  // Scale weights by 0.1
```

**Use Case**: Learning rate scaling, weight decay

### Activation Functions

#### `tensor_relu(tensor_t* tensor)`

**Purpose**: Apply ReLU activation: f(x) = max(0, x)

**Parameters**:
- `tensor`: Tensor to modify (in-place)

**Example**:
```c
tensor_t* activations = tensor_alloc(128, 1);
// ... compute activations ...
tensor_relu(activations);  // Apply ReLU
```

**Use Case**: Feed-forward network activations

#### `tensor_gelu(tensor_t* tensor)`

**Purpose**: Apply GELU activation: f(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))

**Parameters**:
- `tensor`: Tensor to modify (in-place)

**Example**:
```c
tensor_t* hidden = tensor_alloc(128, 128);
// ... compute hidden state ...
tensor_gelu(hidden);  // Apply GELU activation
```

**Use Case**: Modern transformer activations (better than ReLU)

### Normalization Functions

#### `tensor_softmax(tensor_t* tensor)`

**Purpose**: Apply softmax activation to convert logits to probabilities

**Parameters**:
- `tensor`: Tensor to modify (in-place)

**Example**:
```c
tensor_t* logits = tensor_alloc(1, 256);
// ... compute logits ...
tensor_softmax(logits);  // Convert to probabilities
```

**Algorithm**: Two-pass algorithm for numerical stability
1. Find maximum value: `max_val = max(x_i)`
2. Compute exponentials: `exp(x_i - max_val)`
3. Normalize: `softmax(x_i) = exp(x_i - max_val) / Σ exp(x_j - max_val)`

#### `tensor_layer_norm(tensor_t* tensor, float epsilon)`

**Purpose**: Apply layer normalization across the last dimension

**Parameters**:
- `tensor`: Tensor to normalize (in-place)
- `epsilon`: Small value for numerical stability

**Example**:
```c
tensor_t* hidden = tensor_alloc(10, 128);
// ... compute hidden state ...
tensor_layer_norm(hidden, 1e-5f);  // Normalize each row
```

**Algorithm**: For each row:
1. Compute mean: `μ = (1/n) × Σ x_i`
2. Compute variance: `σ² = (1/n) × Σ (x_i - μ)²`
3. Normalize: `(x_i - μ) / √(σ² + ε)`

### Vocabulary Projection

#### `tensor_vocab_projection(const tensor_t* input, const tensor_t* weights)`

**Purpose**: Project hidden dimensions to vocabulary space

**Parameters**:
- `input`: Input tensor [seq_len, hidden_dim]
- `weights`: Projection weights [hidden_dim, vocab_size]

**Returns**: Projected tensor [seq_len, vocab_size], or `NULL` on failure

**Example**:
```c
tensor_t* hidden = tensor_alloc(10, 128);      // 10 tokens, 128 dims
tensor_t* proj_weights = tensor_alloc(128, 256);  // 128 dims → 256 vocab
tensor_fill_random(proj_weights);

tensor_t* logits = tensor_vocab_projection(hidden, proj_weights);
// logits has dimensions [10, 256]
```

**Use Case**: Language modeling and token prediction

## **LLM Model Functions**

### Model Management

#### `llm_init(const llm_config_t* config)`

**Purpose**: Initialize LLM model with specified configuration

**Parameters**:
- `config`: Model configuration structure

**Returns**: Initialized model, or `NULL` on failure

**Example**:
```c
llm_config_t config = {
    .vocab_size = 256,
    .hidden_dim = 128,
    .num_layers = 4,
    .num_heads = 8,
    .max_seq_len = 512
};

llm_model_t* model = llm_init(&config);
if (!model) {
    printf("Failed to initialize model\n");
    return;
}
```

**Initialization Process**:
1. Allocate all weight tensors
2. Initialize with random values
3. Set up layer normalization parameters

#### `llm_free(llm_model_t* model)`

**Purpose**: Deallocate model and free all memory

**Parameters**:
- `model`: Model to deallocate

**Example**:
```c
llm_model_t* model = llm_init(&config);
// ... use model ...
llm_free(model);  // Clean up all memory
```

**Notes**: Frees all weight tensors and model structure

### Generation

#### `llm_generate(llm_model_t* model, const char* input, size_t max_length)`

**Purpose**: Generate AI response to input text

**Parameters**:
- `model`: LLM model
- `input`: Input text string
- `max_length`: Maximum response length

**Returns**: Generated response string (must be freed by caller)

**Example**:
```c
char* response = llm_generate(model, "Hello, how are you?", 100);
if (response) {
    printf("Bot: %s\n", response);
    free(response);
}
```

**Generation Process**:
1. Tokenize input text
2. Forward pass through transformer layers
3. Sample next tokens using temperature scaling
4. Convert tokens back to text

## **Tokenization Functions**

### `tokenize(const char* text, int* tokens, size_t max_tokens)`

**Purpose**: Convert text to token IDs

**Parameters**:
- `text`: Input text string
- `tokens`: Output token array
- `max_tokens`: Maximum number of tokens

**Returns**: Number of tokens generated

**Example**:
```c
int tokens[512];
size_t num_tokens = tokenize("Hello world!", tokens, 512);
printf("Generated %zu tokens\n", num_tokens);
```

**Tokenization**: Character-level (ASCII 0-255)

### `detokenize(const int* tokens, size_t num_tokens)`

**Purpose**: Convert token IDs back to text

**Parameters**:
- `tokens`: Input token array
- `num_tokens`: Number of tokens

**Returns**: Detokenized text string (must be freed by caller)

**Example**:
```c
int tokens[] = {72, 101, 108, 108, 111};  // "Hello"
char* text = detokenize(tokens, 5);
printf("Text: %s\n", text);  // Prints: "Hello"
free(text);
```

## **Transformer Block Functions**

### `transformer_block_forward(tensor_t* input, const llm_model_t* model, size_t layer_idx)`

**Purpose**: Forward pass through a single transformer layer

**Parameters**:
- `input`: Input tensor [seq_len, hidden_dim]
- `model`: LLM model
- `layer_idx`: Layer index (0-based)

**Returns**: Output tensor [seq_len, hidden_dim]

**Example**:
```c
tensor_t* input = tensor_alloc(10, 128);
tensor_t* output = transformer_block_forward(input, model, 0);
// Process output...
tensor_free(input);
tensor_free(output);
```

**Layer Operations**:
1. Layer normalization
2. Multi-head attention
3. Residual connection
4. Layer normalization
5. Feed-forward network
6. Residual connection

## **Error Handling**

### Return Values

**Tensor Functions**:
- `tensor_alloc`: Returns `NULL` on allocation failure
- `tensor_matmul`: Returns `NULL` on dimension mismatch
- `tensor_copy`: Silent failure on dimension mismatch

**LLM Functions**:
- `llm_init`: Returns `NULL` on initialization failure
- `llm_generate`: Returns `NULL` on generation failure

### Error Checking

**Always Check**:
```c
tensor_t* tensor = tensor_alloc(10, 10);
if (!tensor) {
    printf("Allocation failed\n");
    return;
}
```

**Dimension Validation**:
```c
if (a->cols != b->rows) {
    printf("Matrix dimensions incompatible: %zu != %zu\n", a->cols, b->rows);
    return NULL;
}
```

## **Memory Management**

### Allocation Patterns

**Single Tensor**:
```c
tensor_t* tensor = tensor_alloc(rows, cols);
// ... use tensor ...
tensor_free(tensor);
```

**Multiple Tensors**:
```c
tensor_t* tensors[3];
for (int i = 0; i < 3; i++) {
    tensors[i] = tensor_alloc(10, 10);
}

// ... use tensors ...

for (int i = 0; i < 3; i++) {
    tensor_free(tensors[i]);
}
```

**Early Return Cleanup**:
```c
tensor_t* tensor = tensor_alloc(10, 10);
if (!tensor) return;

if (some_error_condition) {
    tensor_free(tensor);
    return;
}

// ... use tensor ...
tensor_free(tensor);
```

### Memory Safety

**NULL Pointer Handling**:
- All functions handle `NULL` inputs gracefully
- `tensor_free` is safe to call on `NULL`

**Resource Cleanup**:
- Always free allocated tensors
- Use consistent allocation/deallocation patterns
- Consider using RAII patterns in C++ wrappers

## 🚀 **Performance Considerations**

### Optimization Tips

**Memory Access Patterns**:
- Row-major order: `tensor->data[i * cols + j]`
- Cache-friendly iteration patterns
- 64-byte aligned allocation for SIMD

**Matrix Multiplication**:
- Current: O(n³) complexity
- Future: Block matrix multiplication, SIMD optimization

**Activation Functions**:
- ReLU: Fast, simple
- GELU: Better gradients, slightly slower
- Softmax: Two-pass for numerical stability

### Profiling

**Memory Usage**:
```bash
valgrind --leak-check=full ./bin/cx
```

**Performance Profiling**:
```bash
perf record ./bin/cx
perf report
```

**CPU Profiling**:
```bash
gcc -pg -o profiled_cx src/*.c -lm
./profiled_cx
gprof profiled_cx gmon.out > profile.txt
```

## 🔧 **Common Usage Patterns**

### Model Training Loop

```c
// Initialize model
llm_model_t* model = llm_init(&config);

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (size_t i = 0; i < num_samples; i++) {
        // Forward pass
        char* response = llm_generate(model, training_data[i], 100);
        
        // Compute loss (implement this)
        float loss = compute_loss(response, target[i]);
        
        // Backward pass (implement this)
        // update_weights(model, gradients);
        
        free(response);
    }
}

llm_free(model);
```

### Tensor Operations Chain

```c
// Create input
tensor_t* input = tensor_alloc(10, 128);
tensor_fill_random(input);

// Apply transformations
tensor_t* hidden = tensor_matmul(input, weights);
tensor_add(hidden, bias);
tensor_gelu(hidden);
tensor_layer_norm(hidden, 1e-5f);

// Cleanup
tensor_free(input);
tensor_free(hidden);
```

## 📝 **Best Practices**

### Code Organization

1. **Check Return Values**: Always verify allocation success
2. **Consistent Naming**: Use descriptive variable names
3. **Error Handling**: Provide meaningful error messages
4. **Resource Management**: Free resources in reverse allocation order

### Performance

1. **Minimize Allocations**: Reuse tensors when possible
2. **Batch Operations**: Process multiple inputs together
3. **Memory Layout**: Use cache-friendly access patterns
4. **Compiler Flags**: Enable optimizations (`-O3`, `-march=native`)

### Debugging

1. **Print Dimensions**: Log tensor shapes during development
2. **Memory Checks**: Use Valgrind for memory issues
3. **Gradient Checks**: Verify numerical gradients during training
4. **Unit Tests**: Test individual functions in isolation

---

*This API reference covers all public functions. For implementation details, see the source code and architecture documentation.*

