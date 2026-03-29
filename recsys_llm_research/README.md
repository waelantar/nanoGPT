# RecSys-LLM Research: Applying Recommender Systems Principles to Large Language Models

## Overview

This research project explores the transfer of architectural efficiency principles from Recommender Systems (RecSys) to Large Language Models (LLMs). The core hypothesis is that LLMs can benefit from the "Wide & Deep" architecture paradigm that has proven effective in RecSys for balancing memorization and generalization.

## Background

### The Efficiency Gap

- **Recommender Systems**: Have dealt with "massive vocabulary" (millions of items) and "efficiency constraints" (ms latency) for decades, using sparse and hybrid architectures.
- **Large Language Models**: Have brute-forced their way through with massive compute, using dense attention mechanisms that scale quadratically with sequence length.

### The "Dense" Trap

- **RecSys** retained efficiency by keeping distinct paths for **Memorization** (Linear/Wide models, sparse features) and **Generalization** (Deep/Dense models).
- **LLMs** abandoned Memorization components, forcing the dense Transformer to memorize facts in its weights, which is inefficient.

## Research Approach

### Core Theoretical Insight

The goal is to re-introduce explicit "Wide" (Sparse/Retrieval) components to the "Deep" (Transformer) LLM architecture, inspired by the success of Wide & Deep models in RecSys.

### Key Experiments

#### 1. Wide & Deep Embedding Layer (Implemented)

This is the first and most impactful experiment, combining:
- **Static (Wide)**: Frozen embeddings (like Word2Vec/GloVe) for fixed semantics and rote memorization
- **Dynamic (Deep)**: Learnable embeddings that adapt to context

The combination is learned during training:
```
E_total = wide_weight * E_static + deep_weight * MLP(E_dynamic)
```

#### 2. Interaction-Based Sparse Attention (Future Work)

Instead of full quadratic attention, use pre-computed "co-occurrence matrices" to create sparse attention patterns based on PMI (Pointwise Mutual Information).

#### 3. Retrieval-Augmented Generation (Token Level) (Future Work)

Use a simple statistical model (like bigram/trigram) as a "draft model" to predict next tokens, then verify with the full transformer - similar to speculative decoding but with RecSys-inspired drafter models.

## Implementation

### Files

1. **`wide_deep_embedding.py`**: Implements the core Wide & Deep embedding layer
2. **`wide_deep_nanogpt.py`**: Integrates the Wide & Deep principle into a GPT-like architecture
3. **`README.md`**: This documentation

### Key Features

- **Modular Design**: Easy to integrate into existing codebases like NanoGPT
- **Flexible Static Embeddings**: Can load pre-trained embeddings (Word2Vec, GloVe) or use learnable static components
- **Learnable Combination Weights**: The model learns the optimal balance between wide and deep components
- **Proper Weight Initialization**: Follows standard transformer initialization practices

## Results

The implementation has been tested and verified to work correctly:
- Forward pass completes successfully
- Proper tensor shapes maintained
- Learnable parameters for combining wide and deep components
- Reasonable parameter count for testing

## Next Steps

1. **Benchmarking**: Compare against standard embeddings on language modeling tasks
2. **Efficiency Analysis**: Measure memory usage and latency improvements
3. **Additional Experiments**: Implement sparse attention and retrieval-augmented generation
4. **Real Static Embeddings**: Load actual pre-trained embeddings instead of random ones
5. **Evaluation**: Test on tasks that require memorization vs. reasoning separately

## Literature Foundation

- **The Inspiration**: *Wide & Deep Learning for Recommender Systems* (Cheng et al., 2016)
- **The Mechanism**: *Generating Long Sequences with Sparse Transformers* (Child et al., 2019)  
- **The Optimization**: *Fast Inference from Transformers via Speculative Decoding* (Leviathan et al., 2023)

## Novel Contributions

1. **Explicit RecSys-LLM Transfer**: Framing the architecture changes as "borrowing from RecSys history" is a strong narrative that hasn't been fully exploited
2. **PMI-based Drafters**: Using RecSys-inspired statistical models as drafters in speculative decoding
3. **Complexity Routing**: Routing "easy" tokens through simple layers and "hard" tokens through full transformer (related to Mixture of Depths)

## Usage

```python
from wide_deep_nanogpt import create_wide_deep_model

model = create_wide_deep_model(
    vocab_size=50257,  # GPT-2 vocab size
    n_embd=768,        # Embedding dimension
    n_head=12,         # Number of attention heads
    n_layer=12,        # Number of transformer layers
    block_size=1024    # Maximum sequence length
)
```

## Conclusion

This implementation provides a solid foundation for exploring RecSys-inspired architectures in LLMs. The Wide & Deep embedding layer is the first step toward more efficient and principled LLM architectures that better balance memorization and generalization.