import torch
import time
import numpy as np
from collections import defaultdict
from pmi_sparse_attention import PMISparseGPT
import sys
sys.path.append('recsys_llm_research')
from recsys_llm_research.wide_deep_nanogpt import WideDeepNanoGPT  # Using the previous implementation as comparison


def create_sample_coocurrence_data(vocab_size=1000, num_pairs=5000):
    """
    Create sample co-occurrence data for testing PMI sparse attention.
    In practice, this would come from analyzing your training corpus.
    """
    token_cooc_counts = {}
    
    # Generate synthetic co-occurrence data
    for _ in range(num_pairs):
        token_i = np.random.randint(0, vocab_size)
        token_j = np.random.randint(0, vocab_size)
        count = np.random.randint(1, 100)
        token_cooc_counts[(token_i, token_j)] = count
    
    return token_cooc_counts


def benchmark_model(model, input_ids, num_runs=10, warmup_runs=3):
    """
    Benchmark a model's forward pass performance.
    """
    model.eval()
    
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_ids)
    
    # Actual benchmarking
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            output = model(input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time, output


def main():
    print("Benchmarking PMI Sparse Attention vs Wide & Deep Embeddings")
    print("="*60)
    
    # Model parameters
    vocab_size = 1000
    embed_dim = 256
    num_heads = 4
    num_layers = 3
    seq_len = 64
    batch_size = 2
    
    # Create sample co-occurrence data
    token_cooc_counts = create_sample_coocurrence_data(vocab_size, 5000)
    
    # Create PMI Sparse GPT model
    print("Creating PMI Sparse GPT model...")
    pmi_model = PMISparseGPT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len,
        top_k=16  # Only attend to 16 highest-PMI tokens per position
    )
    
    # Build PMI mask
    pmi_model.build_pmi_mask_from_data(token_cooc_counts)
    
    # Create Wide & Deep GPT model for comparison
    print("Creating Wide & Deep GPT model...")
    wd_model = WideDeepNanoGPT(
        vocab_size=vocab_size,
        n_embd=embed_dim,
        n_head=num_heads,
        n_layer=num_layers,
        block_size=seq_len
    )
    
    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Sequence length: {seq_len}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Benchmark PMI Sparse GPT
    print("Benchmarking PMI Sparse GPT...")
    pmi_avg_time, pmi_std_time, pmi_output = benchmark_model(pmi_model, input_ids)
    print(f"PMI Sparse GPT - Avg time: {pmi_avg_time:.4f}s ± {pmi_std_time:.4f}s")
    
    # Benchmark Wide & Deep GPT
    print("Benchmarking Wide & Deep GPT...")
    wd_avg_time, wd_std_time, wd_output_tuple = benchmark_model(wd_model, input_ids)
    # WideDeepNanoGPT returns (logits, loss), so extract just the logits
    wd_output = wd_output_tuple[0] if isinstance(wd_output_tuple, tuple) else wd_output_tuple
    print(f"Wide & Deep GPT - Avg time: {wd_avg_time:.4f}s ± {wd_std_time:.4f}s")
    
    print()
    print("Comparison Results:")
    print(f"PMI Sparse GPT shape: {pmi_output.shape}")
    print(f"Wide & Deep GPT shape: {wd_output.shape}")
    print(f"Speed ratio (PMI/WideDeep): {pmi_avg_time/wd_avg_time:.2f}x")
    
    # Calculate approximate computational complexity
    # PMI sparse: O(seq_len * top_k * embed_dim) vs O(seq_len^2 * embed_dim) for dense
    dense_complexity = seq_len * seq_len * embed_dim
    sparse_complexity = seq_len * 16 * embed_dim  # top_k = 16
    theoretical_speedup = dense_complexity / sparse_complexity
    
    print(f"Theoretical complexity reduction: {theoretical_speedup:.2f}x")
    
    # Save results
    results = {
        'pmi_avg_time': pmi_avg_time,
        'pmi_std_time': pmi_std_time,
        'wd_avg_time': wd_avg_time,
        'wd_std_time': wd_std_time,
        'theoretical_speedup': theoretical_speedup,
        'actual_speedup': wd_avg_time / pmi_avg_time,
        'seq_len': seq_len,
        'top_k': 16
    }
    
    print("\nBenchmark completed!")
    print(f"Results: {results}")
    
    return results


if __name__ == "__main__":
    main()