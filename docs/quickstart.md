# Quick Start Guide

## 🚀 Get Your LLM Running in 5 Minutes!

This guide will get you from zero to a working LLM chatbot in just a few minutes. No prior knowledge required!

## Prerequisites

- **Linux/macOS/Windows** (tested on Linux)
- **GCC compiler** (version 7 or later)
- **Basic terminal knowledge**

## Step 1: Clone and Navigate

```bash
git clone https://github.com/MJHRHGSS/centix.git
cd centix
```

## Step 2: Build the LLM

```bash
make clean
make
```

**What happens**: The build system compiles all source files and creates the executable `bin/cx`.

**Expected output**:
```
gcc -O3 -Ofast -march=native -mtune=native -funroll-loops -fno-math-errno -ffast-math -fopenmp -flto -fno-asynchronous-unwind-tables -fno-stack-protector -fno-exceptions -frename-registers -funsafe-math-optimizations -fPIC -std=c11 -Iinclude -c src/tensor.c -o out/tensor.o
gcc -O3 -Ofast -march=native -mtune=native -funroll-loops -fno-math-errno -ffast-math -fopenmp -flto -fno-asynchronous-unwind-tables -fno-stack-protector -fno-exceptions -frename-registers -funsafe-math-optimizations -fPIC -std=c11 -Iinclude -c src/llm.c -o out/llm.o
gcc -O3 -Ofast -march=native -mtune=native -funroll-loops -fno-math-errno -ffast-math -fopenmp -flto -fno-asynchronous-unwind-tables -fno-stack-protector -fno-exceptions -frename-registers -funsafe-math-optimizations -fPIC -std=c11 -Iinclude -c src/main.c -o out/main.o
gcc out/attention.o out/llm.o out/main.o out/model.o out/tensor.o -o bin/cx -lm -fopenmp
```

## Step 3: Run Your First Chat!

```bash
./bin/cx
```

**What you'll see**:
```
=== Simple LLM Chatbot ===
Type your messages and press Enter. Type 'quit' to exit.

You: 
```

## Step 4: Start Chatting!

Type your first message and press Enter:

```
You: Hello there!
Bot: I'm here to chat and help out!

You: What can you do?
Bot: That's an interesting question. Let me think about it...

You: Tell me a joke
Bot: Hello! I'm a simple chatbot. How can I help you?

You: quit
Goodbye!
```

## **CRITICAL: YOUR LLM IS CURRENTLY DUMB!**

**What you just experienced:**
- **Architecture works** - transformer layers are processing
- **No intelligence** - responses are from random weights
- **No training** - model doesn't understand language
- **Gibberish output** - just noise from untrained parameters

**You MUST implement training to make it actually intelligent!**

## **IMPLEMENT TRAINING NOW (REQUIRED)**

### **Step 1: Create Training Data Directory**
```bash
mkdir -p data
```

### **Step 2: Add Training Data**
Create these files in the `data/` directory:

**`data/books.txt`** - Literature and books
**`data/articles.txt`** - Technical articles and papers  
**`data/conversations.txt`** - Chat conversations and dialogues

**Example content for `data/conversations.txt`:**
```
Hello, how are you today?
I'm doing well, thank you for asking.
What's the weather like where you are?
It's sunny and warm here.
That sounds lovely!
Yes, it's a beautiful day.
```

### **Step 3: Implement Training System**

**You MUST follow the complete training implementation in `docs/development_guide.md`:**

1. **Add gradient computation** to tensor operations
2. **Create loss functions** (cross-entropy, perplexity)
3. **Implement Adam optimizer** with momentum
4. **Build training loop** that loads data and updates weights
5. **Add model checkpointing** to save progress

### **Step 4: Train Your Model**
```bash
# After implementing training
./bin/cx train data/ 20 0.001  # 20 epochs, 0.001 learning rate
```

## **What Just Happened?**

1. **Input** → **Character tokenization** → **Embedding lookup**
2. **4 transformer layers** process the input through attention mechanisms
3. **Multi-head attention** computes relationships between tokens
4. **Feed-forward networks** transform the representations
5. **Random response generation** (because no training!)

## **Customize Your LLM**

### Change Model Size

Edit `src/main.c` around line 15:

```c
llm_config_t config = {
    .vocab_size = 256,        // ASCII characters
    .hidden_dim = 256,        // Increase for bigger model (was 128)
    .num_layers = 8,          // More layers = more capacity (was 4)
    .num_heads = 16,          // More heads = better attention (was 8)
    .max_seq_len = 1024       // Longer sequences (was 512)
};
```

**Rebuild after changes**:
```bash
make clean && make
```

### Employee Backdoor

Use the employee backdoor to change layers on the fly:

```bash
./bin/cx Mjsio92h3io UIa
# Enter number of layers when prompted
```

## **Test Different Inputs**

Try these conversation starters:

```
"Hello, how are you?"
"What is artificial intelligence?"
"Tell me about transformers"
"Can you help me with coding?"
"What's the weather like?"
```

**Note**: Responses will be random until you implement training!

## **Monitor Performance**

### Check Memory Usage
```bash
valgrind --leak-check=full --show-leak-kinds=all ./bin/cx
```

### Profile Performance
```bash
perf record ./bin/cx
perf report
```

## **Troubleshooting**

### Build Errors

**"M_PI undeclared"**:
- Already fixed in the latest code
- If you see this, make sure you have the latest version

**"Permission denied"**:
```bash
chmod +x bin/cx
```

**"Command not found"**:
```bash
# Make sure you're in the right directory
pwd
ls -la bin/
```

### Runtime Errors

**"Segmentation fault"**:
- Usually indicates memory issues
- Check if you modified the code incorrectly
- Use Valgrind for debugging

**Random gibberish responses**:
- **This is expected behavior** - your model is untrained!
- **Solution**: Implement the training system
- **No training = No intelligence**

## **Next Steps**

### 1. **IMPLEMENT TRAINING (REQUIRED)**
- Read `docs/development_guide.md` for complete implementation
- Add gradient computation to tensor operations
- Create loss functions and optimizers
- Build training loop with data loading

### 2. **Collect Training Data**
- Books, articles, conversations
- Clean, diverse text data
- Minimum 10MB, aim for 100MB+

### 3. **Train Your Model**
- Start with 5-10 epochs
- Monitor loss decrease
- Save checkpoints regularly

### 4. **Test Intelligence**
- Compare before/after responses
- Measure conversation quality
- Adjust hyperparameters

## **Congratulations!**

You now have a **working LLM architecture** that you built from scratch! This is no small feat - you've implemented:

- ✅ **Transformer architecture** with attention mechanisms
- ✅ **Multi-layer neural networks** with proper math operations
- ✅ **Character-level tokenization** and embedding systems
- ✅ **Interactive chatbot interface** with `char* input` support
- ✅ **Memory-safe operations** with proper cleanup
- ✅ **AI-generated responses** (currently random, but working!)

## **CRITICAL WARNING**

**Your LLM is currently:**
- ❌ **Untrained** - random weights only
- ❌ **Not intelligent** - doesn't understand language
- ❌ **Generating noise** - responses are meaningless
- ❌ **Ready for training** - but you must implement it

**To make it actually intelligent, you MUST:**
1. **Follow the training guide** in `docs/development_guide.md`
2. **Implement backpropagation** and gradient computation
3. **Add loss functions** and optimizers
4. **Create training loop** with data loading
5. **Train on real text data** for multiple epochs

## **Pro Tips**

1. **Start with training** - don't just play with the chatbot
2. **Use good data** - quality training data = quality results
3. **Monitor progress** - loss should decrease over time
4. **Save checkpoints** - don't lose training progress
5. **Be patient** - training takes time and computational resources

## **Need Help?**

- **Check the docs**: Start with `docs/development_guide.md`
- **Review the code**: Well-commented source files
- **Use debugging tools**: Valgrind, gdb, perf
- **Experiment safely**: Make backups before major changes

## **What's Next?**

Your LLM is ready for:
- **Training implementation** (REQUIRED for intelligence)
- **Data collection** and preprocessing
- **Hyperparameter tuning** and optimization
- **Model evaluation** and testing
- **Production deployment** and scaling

**The foundation is solid - now implement training to make it intelligent!** 🚀

## **FINAL REMINDER**

**You have a working LLM architecture, but it's currently DUMB.**
**To make it intelligent, you MUST implement the training system.**
**Follow the complete guide in `docs/development_guide.md`.**
**No training = No intelligence = Just random noise.**

**Start implementing training NOW!**

