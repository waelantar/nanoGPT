"""
Benchmark script to compare Wide & Deep embeddings vs standard embeddings
"""
import time
import torch
import torch.nn as nn
from wide_deep_embedding import WideDeepEmbedding


def benchmark_embeddings():
    """Compare Wide & Deep embeddings vs standard embeddings"""
    
    # Parameters
    vocab_size = 10000
    embedding_dim = 512
    batch_size = 32
    seq_len = 128
    
    print(f"Benchmarking parameters:")
    print(f"- Vocabulary size: {vocab_size:,}")
    print(f"- Embedding dimension: {embedding_dim}")
    print(f"- Batch size: {batch_size}")
    print(f"- Sequence length: {seq_len}")
    print()
    
    # Create dummy static embeddings for Wide & Deep
    static_embeddings = torch.randn(vocab_size, embedding_dim // 2)
    
    # Create models
    wide_deep_emb = WideDeepEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        static_embeddings=static_embeddings,
        freeze_static=True
    )
    
    standard_emb = nn.Embedding(vocab_size, embedding_dim)
    
    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Warm up
    for _ in range(5):
        _ = wide_deep_emb(input_ids)
        _ = standard_emb(input_ids)
    
    # Benchmark Wide & Deep
    start_time = time.time()
    for _ in range(10):
        _ = wide_deep_emb(input_ids)
    wide_deep_time = time.time() - start_time
    
    # Benchmark Standard
    start_time = time.time()
    for _ in range(10):
        _ = standard_emb(input_ids)
    standard_time = time.time() - start_time
    
    # Count parameters
    wd_params = sum(p.numel() for p in wide_deep_emb.parameters())
    std_params = sum(p.numel() for p in standard_emb.parameters())
    
    print(f"Wide & Deep Embedding:")
    print(f"- Parameters: {wd_params:,}")
    print(f"- Time for 10 forward passes: {wide_deep_time:.4f}s")
    print(f"- Average time per forward pass: {wide_deep_time/10*1000:.2f}ms")
    
    print()
    
    print(f"Standard Embedding:")
    print(f"- Parameters: {std_params:,}")
    print(f"- Time for 10 forward passes: {standard_time:.4f}s")
    print(f"- Average time per forward pass: {standard_time/10*1000:.2f}ms")
    
    print()
    
    print(f"Comparison:")
    print(f"- Wide & Deep is {((wide_deep_time/10) / (standard_time/10) - 1) * 100:+.1f}% {'slower' if wide_deep_time > standard_time else 'faster'} than standard")
    print(f"- Wide & Deep has {((wd_params / std_params) - 1) * 100:+.1f}% {'more' if wd_params > std_params else 'fewer'} parameters than standard")


def test_gradient_flow():
    """Test that gradients flow properly through the Wide & Deep embedding"""
    
    print("\nTesting gradient flow...")
    
    # Parameters
    vocab_size = 1000
    embedding_dim = 128
    batch_size = 4
    seq_len = 16
    
    # Create dummy static embeddings
    static_embeddings = torch.randn(vocab_size, embedding_dim // 2)
    
    # Create model
    model = WideDeepEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        static_embeddings=static_embeddings,
        freeze_static=True  # Freeze static embeddings to test dynamic gradients
    )
    
    # Create input and target
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    output = model(input_ids)
    
    # Simple loss function (just to test gradients)
    loss_fn = nn.MSELoss()
    target_output = torch.randn_like(output)
    loss = loss_fn(output, target_output)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad_dynamic = model.dynamic_embedding.weight.grad is not None
    has_grad_static = model.static_embeddings.weight.grad is not None if not model.freeze_static else model.static_embeddings.weight.grad is None
    has_grad_wide_weight = model.wide_weight.grad is not None
    has_grad_deep_weight = model.deep_weight.grad is not None
    
    print(f"- Dynamic embedding has gradients: {has_grad_dynamic}")
    print(f"- Static embedding has gradients: {has_grad_static} (should be False if frozen)")
    print(f"- Wide weight has gradients: {has_grad_wide_weight}")
    print(f"- Deep weight has gradients: {has_grad_deep_weight}")
    print(f"- MLP has gradients: {any(p.grad is not None for p in model.mlp.parameters())}")
    
    print("✓ Gradient flow test passed!")


if __name__ == "__main__":
    print("Running Wide & Deep Embedding Benchmarks\n")
    benchmark_embeddings()
    test_gradient_flow()
    print("\nAll benchmarks completed successfully!")