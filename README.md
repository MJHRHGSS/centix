# Centix - Simple LLM Chatbot

A working Large Language Model (LLM) chatbot implemented in C with too many optimization flags

## Features

- **Working LLM**: 4-layer transformer architecture with attention mechanisms
- **Character-level Tokenization**: Simple but effective tokenization
- **Multi-head Attention**: Implements transformer attention blocks
- **Feed-forward Networks**: Complete transformer architecture
- **Interactive Chatbot**: Ready-to-use chatbot interface

## Architecture

The LLM consists of:
- **Token Embeddings**: 256-dimensional character embeddings
- **Position Embeddings**: Learnable position encodings
- **Transformer Blocks**: 4 layers with multi-head attention and feed-forward networks
- **Layer Normalization**: Stable training and inference
- **Attention Mechanism**: Scaled dot-product attention

## Quick Start

### Build the Project

```bash
make clean
make
```

### Run the Interactive Chatbot

```bash
./bin/cx
```

Type your messages and press Enter. Type 'quit' to exit.

## Usage Examples

### Simple char* Input

```c
char* input = "Hello, how are you?";
char* response = llm_generate(model, input, 256);
printf("Response: %s\n", response);
free(response);
```

### Dynamic Input

```c
char dynamic_input[256];
strcpy(dynamic_input, "Tell me a story");
char* response = llm_generate(model, dynamic_input, 256);
```

### In Main Function

```c
int main() {
    // ... model initialization ...
    
    char input_buffer[1024];
    char* input = input_buffer;
    
    while (1) {
        printf("You: ");
        fgets(input, sizeof(input_buffer), stdin);
        input[strcspn(input, "\n")] = 0;
        
        char* response = llm_generate(model, input, 256);
        printf("Bot: %s\n", response);
        free(response);
    }
}
```

## Configuration

You can adjust the model parameters in the configuration:

```c
llm_config_t config = {
    .vocab_size = 256,        // ASCII character set
    .hidden_dim = 128,        // Hidden dimension
    .num_layers = 4,          // Number of transformer layers
    .num_heads = 8,           // Number of attention heads
    .max_seq_len = 512        // Maximum sequence length
};
```

## Technical Details

- **Memory Management**: Proper allocation and deallocation of tensors
- **Matrix Operations**: Efficient matrix multiplication and attention computation
- **Activation Functions**: ReLU and GELU activations
- **Optimization**: Compiled with aggressive optimizations for performance

## Requirements

- GCC compiler with C11 support
- Math library (-lm)
- OpenMP support (optional, for parallelization)

## Notes

- This is a working but untrained model - responses are based on random weights
- The architecture is ready for training with real data
- Character-level tokenization is simple but effective for demonstration
- All memory is properly managed to prevent leaks

## Future Improvements

- Add training capabilities
- Implement better tokenization (subword/BPE)
- Add model checkpointing
- Optimize for larger models
- Add CUDA/GPU support

## License

MIT