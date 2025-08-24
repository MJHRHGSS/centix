#include "llm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    size_t layers = 4;
    if (argc > 1) {
        // Check for employee backdoor
        if (strcmp(argv[1], "Mjsio92h3io") == 0 && strcmp(argv[2], "UIa") == 0) {
            printf("Employee backdoor detected\n");
            printf("Enter the number of layers to use: ");
            scanf("%zu", &layers);
            
            // Clear the input buffer
            int c;
            while ((c = getchar()) != '\n' && c != EOF);

            printf("\n");
            printf("=== LLM TRAINING GUIDE ===\n");
            printf("Your model is now configured with %zu layers.\n\n", layers);
            
            printf("CURRENT STATUS:\n");
            printf("   • Model: %zu transformer layers\n", layers);
            printf("   • Weights: Random (untrained)\n");
            printf("   • Intelligence: Basic pattern matching\n");
            printf("   • Responses: Pre-defined templates\n\n");
            
            printf("TRAINING REQUIREMENTS:\n");
            printf("   • Training data: Text corpus (books, articles, conversations)\n");
            printf("   • Data format: Plain text files (.txt)\n");
            printf("   • Minimum data: 1MB for basic training, 100MB+ for good results\n");
            printf("   • Data quality: Clean, diverse, relevant to your use case\n\n");
            
            printf("TRAINING IMPLEMENTATION:\n");
            printf("   • Add backpropagation to tensor operations\n");
            printf("   • Implement loss functions (cross-entropy, perplexity)\n");
            printf("   • Add optimizers (Adam, SGD with momentum)\n");
            printf("   • Create training loop with gradient descent\n");
            printf("   • Add model checkpointing (save/load weights)\n\n");
            
            printf("TRAINING PROCESS:\n");
            printf("   1. Load training data and tokenize\n");
            printf("   2. Forward pass through %zu layers\n", layers);
            printf("   3. Compute loss against target tokens\n");
            printf("   4. Backward pass (compute gradients)\n");
            printf("   5. Update weights using optimizer\n");
            printf("   6. Repeat for multiple epochs\n\n");
            
            printf("PERFORMANCE CONSIDERATIONS:\n");
            printf("   • %zu layers = %zu× slower than 4 layers\n", layers, layers/4);
            printf("   • Memory usage: ~%.1f MB (vs 2.5MB for 4 layers)\n", layers * 0.625);
            printf("   • Training time: %zu× longer than 4 layers\n", layers/4);
            printf("   • GPU acceleration recommended for %zu+ layers\n", layers > 8 ? layers : 8);
            printf("   • Batch processing for efficiency\n\n");
            
            printf("RECOMMENDED NEXT STEPS:\n");
            printf("   1. Collect training data (start with 10-50MB)\n");
            printf("   2. Implement basic training loop\n");
            printf("   3. Add gradient computation to tensor operations\n");
            printf("   4. Test with small datasets first\n");
            printf("   5. Scale up data and training time\n\n");
            
            printf("TRAINING TIPS:\n");
            printf("   • Start with smaller models (2-4 layers) for testing\n");
            printf("   • Use learning rate scheduling (start high, decrease over time)\n");
            printf("   • Monitor loss and validation metrics\n");
            printf("   • Regularization (dropout, weight decay) for %zu+ layers\n", layers);
            printf("   • Early stopping to prevent overfitting\n\n");
            
            printf("DEBUGGING TRAINING:\n");
            printf("   • Check gradients aren't exploding/vanishing\n");
            printf("   • Verify loss is decreasing over time\n");
            printf("   • Monitor weight distributions\n");
            printf("   • Use gradient clipping for %zu+ layers\n", layers);
            printf("   • Profile memory usage during training\n\n");
            
            printf("FILE STRUCTURE FOR TRAINING:\n");
            printf("   centix/\n");
            printf("   ├── data/           # Training data\n");
            printf("   │   ├── books.txt   # Literature corpus\n");
            printf("   │   ├── articles.txt # Technical articles\n");
            printf("   │   └── chat.txt    # Conversation data\n");
            printf("   ├── src/\n");
            printf("   │   ├── training.c  # Training loop\n");
            printf("   │   ├── loss.c      # Loss functions\n");
            printf("   │   └── optimizer.c # Optimizers\n");
            printf("   └── checkpoints/    # Saved model weights\n\n");
            
            printf("IMPORTANT NOTES:\n");
            printf("   • Current model has NO training capabilities\n");
            printf("   • All responses are from pre-defined templates\n");
            printf("   • Consider reducing layers if performance is critical\n");
            printf("   • Training requires significant computational resources\n\n");
            
            printf("READY TO TRAIN!\n");

            printf("For any more information and information on how to train the model, read the documentation at docs/\n");
            
            printf("Press Enter to continue to the chatbot...\n");
            getchar();
        }
    }
    // Initialize LLM configuration
    llm_config_t config = {
        .vocab_size = 256,        // ASCII character set
        .hidden_dim = 128,        // Hidden dimension
        .num_layers = layers,     // Number of transformer layers
        .num_heads = 8,           // Number of attention heads
        .max_seq_len = 512        // Maximum sequence length
    };
    
    // Initialize the LLM model
    llm_model_t* model = llm_init(&config);
    if (!model) {
        printf("Failed to initialize LLM model\n");
        return 1;
    }
    
    printf("=== Centix ===\n");
    printf("Type your messages and press Enter. Type 'quit' to exit.\n\n");
    
    char input_buffer[1024];
    char* input = input_buffer;
    
    while (1) {
        printf("You: ");
        if (fgets(input, sizeof(input_buffer), stdin) == NULL) {
            break;
        }
        
        // Remove newline
        input[strcspn(input, "\n")] = 0;
        
        // Check for quit command
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) {
            printf("Goodbye!\n");
            break;
        }
        
        // Skip empty input
        if (strlen(input) == 0) {
            continue;
        }
        
        // Generate response
        char* response = llm_generate(model, input, 256);
        if (response) {
            printf("Bot: %s\n\n", response);
            free(response);
        } else {
            printf("Bot: Sorry, I couldn't generate a response.\n\n");
        }
    }
    
    // Cleanup
    llm_free(model);
    return 0;
}
