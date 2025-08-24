# LLM Architecture Deep Dive

## Overview

This document explains the internal architecture of the Centix LLM, a 4-layer transformer model that processes text through attention mechanisms and generates responses.

## High-Level Architecture

```
Input Text → Tokenization → Embeddings → Transformer Layers → Output Generation
     ↓              ↓           ↓              ↓               ↓
  "Hello"    [72, 101, ...]   [0.1, ...]   [0.3, ...]   "Hi there!"
```

## Core Components

### 1. Tokenization System

**File**: `src/llm.c` (lines 280-290)

The LLM uses **character-level tokenization**:
- Each ASCII character becomes a token ID (0-255)
- Simple but effective for demonstration
- No vocabulary file needed

```c
size_t tokenize(const char* text, int* tokens, size_t max_tokens) {
    size_t len = strlen(text);
    size_t token_count = 0;
    
    for (size_t i = 0; i < len && token_count < max_tokens; i++) {
        tokens[token_count++] = (int)text[i];
    }
    
    return token_count;
}
```

### 2. Embedding Layers

**File**: `src/llm.c` (lines 320-340)

Two types of embeddings are learned:

#### Token Embeddings
- **Shape**: `[vocab_size, hidden_dim]` = `[256, 128]`
- **Purpose**: Convert token IDs to dense vectors
- **Initialization**: Random values from uniform distribution

#### Position Embeddings
- **Shape**: `[max_seq_len, hidden_dim]` = `[512, 128]`
- **Purpose**: Give the model information about token positions
- **Initialization**: Random values from uniform distribution

```c
// Get embeddings for input tokens
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
```

### 3. Transformer Block Architecture

**File**: `src/llm.c` (lines 180-250)

Each transformer block consists of:

```
Input → Layer Norm 1 → Multi-Head Attention → Residual → Layer Norm 2 → Feed-Forward → Residual → Output
```

#### Layer Normalization
- **Purpose**: Stabilize training and improve convergence
- **Implementation**: Normalize across the hidden dimension
- **Epsilon**: 1e-5 for numerical stability

#### Multi-Head Attention
- **Heads**: 8 attention heads
- **Head Dimension**: 128/8 = 16 dimensions per head
- **Process**:
  1. Project input to Q, K, V (Query, Key, Value)
  2. Compute attention scores: `Q × K^T / √(head_dim)`
  3. Apply softmax to get attention weights
  4. Weighted sum of values using attention weights

#### Feed-Forward Network
- **Hidden Size**: 4 × hidden_dim = 512
- **Activation**: GELU (Gaussian Error Linear Unit)
- **Structure**: Linear → GELU → Linear

### 4. Attention Mechanism Details

**File**: `src/llm.c` (lines 90-150)

The attention mechanism follows the standard scaled dot-product attention:

```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Implementation Steps**:
1. **QKV Projection**: `input × weights` → `[seq_len, hidden_dim*3]`
2. **Attention Scores**: Compute dot products between all positions
3. **Scaling**: Divide by `√(head_dim)` for stable gradients
4. **Softmax**: Convert scores to probabilities
5. **Value Weighting**: Apply attention weights to values
6. **Output Projection**: Project back to `hidden_dim`

### 5. Output Generation

**File**: `src/llm.c` (lines 350-380)

Currently uses a simple response selection system:
- 5 predefined responses
- Selection based on first output value
- Future: Can be replaced with proper language modeling head

## Data Flow Through the Model

### Forward Pass Example

For input "Hi" (2 characters):

1. **Tokenization**: `['H', 'i']` → `[72, 105]`
2. **Embedding Lookup**: 
   - Token embeddings: `[72, 105]` → `[2, 128]` tensor
   - Position embeddings: `[0, 1]` → `[2, 128]` tensor
   - Combined: `[2, 128]` tensor
3. **Transformer Layers** (4 iterations):
   - Layer 0: `[2, 128]` → `[2, 128]`
   - Layer 1: `[2, 128]` → `[2, 128]`
   - Layer 2: `[2, 128]` → `[2, 128]`
   - Layer 3: `[2, 128]` → `[2, 128]`
4. **Output Selection**: Use first output value to select response

### Memory Layout

Tensors use **row-major** memory layout:
```
tensor->data[i * cols + j] = value at row i, column j
```

## Configuration Parameters

**File**: `src/main.c` (lines 15-21)

```c
llm_config_t config = {
    .vocab_size = 256,        // ASCII character set
    .hidden_dim = 128,        // Hidden dimension
    .num_layers = 4,          // Number of transformer layers
    .num_heads = 8,           // Number of attention heads
    .max_seq_len = 512        // Maximum sequence length
};
```

## Performance Characteristics

- **Model Size**: ~2.5MB (mostly random weights)
- **Memory Usage**: Scales with sequence length and hidden dimension
- **Computational Complexity**: O(n² × d) per layer where n=seq_len, d=hidden_dim
- **Parallelization**: Matrix operations can be parallelized (OpenMP enabled)

## Future Improvements

1. **Training Infrastructure**: Add backpropagation and optimization
2. **Better Tokenization**: Implement subword/BPE tokenization
3. **Model Checkpointing**: Save/load trained weights
4. **GPU Acceleration**: CUDA implementation for larger models
5. **Attention Optimizations**: Sparse attention, linear attention

