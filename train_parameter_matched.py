"""
Train parameter-matched models for fair comparison
All models will have ~731K parameters
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import time
import sys

sys.path.append('recsys_llm_research')

from model import GPT, GPTConfig
from recsys_llm_research.wide_deep_nanogpt import WideDeepNanoGPT
from efficient_pmi_sparse_attention import create_efficient_pmi_model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def create_synthetic_data(vocab_size, seq_len, num_batches):
    """Create synthetic training data"""
    data = []
    for _ in range(num_batches):
        x = torch.randint(0, vocab_size, (8, seq_len))
        y = torch.randint(0, vocab_size, (8, seq_len))
        data.append((x, y))
    return data


def train_model(model, train_data, test_data, device, model_name, max_iters=2000):
    """Train a model with quality gates"""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    print(f"\nTraining {model_name}")
    print(f"Parameters: {count_params(model):,}")

    model.train()
    losses = []

    for iter in range(max_iters):
        # Get batch
        batch_idx = iter % len(train_data)
        x, y = train_data[batch_idx]
        x, y = x.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()

        # Handle different model outputs
        try:
            output = model(x, y)
            if isinstance(output, tuple) and len(output) == 2:
                logits, loss = output
                if loss is None:
                    logits = output[0]
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            else:
                logits = output
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        except:
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        # Print progress
        if (iter + 1) % 100 == 0:
            avg_loss = np.mean(losses[-100:])
            perplexity = np.exp(avg_loss)
            print(f"  Iter {iter+1}/{max_iters} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f}")

            # Quality gate: stop if not learning
            if iter > 500 and perplexity > 1000:
                print(f"  WARNING: High perplexity at iter {iter}, may not be learning")

    # Final evaluation
    model.eval()
    test_losses = []
    with torch.no_grad():
        for x, y in test_data[:50]:
            x, y = x.to(device), y.to(device)
            try:
                output = model(x, y)
                if isinstance(output, tuple):
                    logits, loss = output
                    if loss is None:
                        logits = output[0]
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                else:
                    logits = output
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            except:
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            test_losses.append(loss.item())

    final_loss = np.mean(test_losses)
    final_ppl = np.exp(final_loss)

    print(f"  Final test loss: {final_loss:.4f}")
    print(f"  Final test perplexity: {final_ppl:.2f}")

    # Quality gate check
    if final_ppl > 400:
        print(f"  ❌ QUALITY GATE FAILED: {final_ppl:.2f} > 400")
        return model, final_ppl, False
    else:
        print(f"  ✓ QUALITY GATE PASSED: {final_ppl:.2f} < 400")
        return model, final_ppl, True


def benchmark_speed(model, input_ids, device, num_runs=50):
    """Benchmark inference speed"""
    model = model.to(device)
    model.eval()
    input_ids = input_ids.to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids)

    # Benchmark
    times = []
    for _ in range(num_runs):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(input_ids)
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - start)

    return np.mean(times), np.std(times)


def main():
    # Configuration
    vocab_size = 1000
    seq_len = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    TARGET_PARAMS = 731_264

    print("="*80)
    print("PARAMETER-MATCHED TRAINING AND BENCHMARKING")
    print("="*80)
    print(f"Target parameters: {TARGET_PARAMS:,}")
    print(f"Device: {device}")

    # Create data
    print("\nCreating synthetic data...")
    train_data = create_synthetic_data(vocab_size, seq_len, 500)
    test_data = create_synthetic_data(vocab_size, seq_len, 100)

    # 1. Standard GPT (baseline)
    print("\n" + "="*80)
    print("1. STANDARD GPT (BASELINE)")
    print("="*80)

    std_config = GPTConfig(
        vocab_size=vocab_size,
        n_embd=128,
        n_head=4,
        n_layer=3,
        block_size=seq_len,
        dropout=0.1
    )
    std_model = GPT(std_config)
    std_params = count_params(std_model)
    print(f"Standard GPT parameters: {std_params:,}")

    std_model, std_ppl, std_pass = train_model(std_model, train_data, test_data, device, "Standard GPT", max_iters=2000)

    # 2. Wide & Deep (parameter-matched)
    print("\n" + "="*80)
    print("2. WIDE & DEEP GPT (PARAMETER-MATCHED)")
    print("="*80)

    # Reduce n_embd to match parameters
    wd_model = WideDeepNanoGPT(
        vocab_size=vocab_size,
        n_embd=96,  # Reduced from 128 to compensate for dual embeddings
        n_head=4,
        n_layer=3,
        block_size=seq_len
    )
    wd_params = count_params(wd_model)
    print(f"Wide & Deep parameters: {wd_params:,}")
    print(f"Parameter difference: {wd_params - std_params:+,} ({(wd_params/std_params - 1)*100:+.1f}%)")

    if abs(wd_params - std_params) / std_params > 0.05:
        print(f"WARNING: Parameters differ by more than 5%")

    wd_model, wd_ppl, wd_pass = train_model(wd_model, train_data, test_data, device, "Wide & Deep GPT", max_iters=2000)

    # 3. PMI Sparse (parameter-matched)
    print("\n" + "="*80)
    print("3. PMI SPARSE GPT (PARAMETER-MATCHED)")
    print("="*80)

    # Reduce n_layer to match parameters
    pmi_model = create_efficient_pmi_model(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,  # Reduced from 3
        max_seq_len=seq_len,
        top_k=32
    )
    pmi_params = count_params(pmi_model)
    print(f"PMI Sparse parameters: {pmi_params:,}")
    print(f"Parameter difference: {pmi_params - std_params:+,} ({(pmi_params/std_params - 1)*100:+.1f}%)")

    if abs(pmi_params - std_params) / std_params > 0.05:
        print(f"WARNING: Parameters differ by more than 5%")

    # Build PMI pattern
    from collections import defaultdict
    cooc_counts = defaultdict(int)
    for _ in range(5000):
        i, j = np.random.zipf(1.5) % vocab_size, np.random.zipf(1.5) % vocab_size
        if i != j:
            cooc_counts[(i, j)] += 1

    temp_attention = pmi_model.blocks[0].attention
    pmi_matrix = temp_attention.compute_pmi_matrix(dict(cooc_counts), vocab_size, 32)
    sample_input = torch.randint(0, vocab_size, (1, seq_len))
    pmi_model.build_attention_pattern_from_sequence(sample_input, pmi_matrix)

    pmi_model, pmi_ppl, pmi_pass = train_model(pmi_model, train_data, test_data, device, "PMI Sparse GPT", max_iters=2000)

    # Benchmark speed (only for models that passed quality gate)
    print("\n" + "="*80)
    print("SPEED BENCHMARKING (Quality-gated models only)")
    print("="*80)

    test_input = torch.randint(0, vocab_size, (2, seq_len))

    results = {}

    if std_pass:
        std_time, std_std = benchmark_speed(std_model, test_input, device)
        results['Standard GPT'] = {
            'params': std_params,
            'perplexity': std_ppl,
            'time': std_time,
            'time_std': std_std,
            'passed': std_pass
        }
        print(f"Standard GPT: {std_time*1000:.2f}ms ± {std_std*1000:.2f}ms")

    if wd_pass:
        wd_time, wd_std = benchmark_speed(wd_model, test_input, device)
        results['Wide & Deep GPT'] = {
            'params': wd_params,
            'perplexity': wd_ppl,
            'time': wd_time,
            'time_std': wd_std,
            'passed': wd_pass
        }
        print(f"Wide & Deep GPT: {wd_time*1000:.2f}ms ± {wd_std*1000:.2f}ms")

    if pmi_pass:
        pmi_time, pmi_std = benchmark_speed(pmi_model, test_input, device)
        results['PMI Sparse GPT'] = {
            'params': pmi_params,
            'perplexity': pmi_ppl,
            'time': pmi_time,
            'time_std': pmi_std,
            'passed': pmi_pass
        }
        print(f"PMI Sparse GPT: {pmi_time*1000:.2f}ms ± {pmi_std*1000:.2f}ms")

    # Final results table
    print("\n" + "="*80)
    print("FINAL RESULTS (Parameter-Matched Comparison)")
    print("="*80)
    print(f"\n{'Model':<20} {'Params':<12} {'PPL':<10} {'Time (ms)':<12} {'Speedup':<10} {'Quality'}")
    print("-"*80)

    baseline_time = results['Standard GPT']['time'] if 'Standard GPT' in results else None

    for name, res in results.items():
        speedup = baseline_time / res['time'] if baseline_time else 1.0
        quality = "PASS" if res['passed'] else "FAIL"
        print(f"{name:<20} {res['params']:<12,} {res['perplexity']:<10.1f} "
              f"{res['time']*1000:<12.2f} {speedup:<10.2f}x {quality}")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    if not std_pass:
        print("❌ Baseline failed quality gate - cannot proceed")
        return

    print(f"✓ Baseline (Standard GPT): {std_ppl:.1f} perplexity")

    if wd_pass:
        wd_speedup = baseline_time / wd_time
        wd_quality_loss = (wd_ppl - std_ppl) / std_ppl * 100
        print(f"\nWide & Deep GPT:")
        print(f"  Quality: {wd_ppl:.1f} PPL ({wd_quality_loss:+.1f}% vs baseline)")
        print(f"  Speed: {wd_speedup:.2f}x")
        if wd_quality_loss < 10:
            print(f"  ✓ Acceptable quality degradation")
        else:
            print(f"  ⚠ Quality degradation high")
    else:
        print(f"\n❌ Wide & Deep failed quality gate: {wd_ppl:.1f} PPL")

    if pmi_pass:
        pmi_speedup = baseline_time / pmi_time
        pmi_quality_loss = (pmi_ppl - std_ppl) / std_ppl * 100
        print(f"\nPMI Sparse GPT:")
        print(f"  Quality: {pmi_ppl:.1f} PPL ({pmi_quality_loss:+.1f}% vs baseline)")
        print(f"  Speed: {pmi_speedup:.2f}x")
        if pmi_quality_loss < 10:
            print(f"  ✓ Acceptable quality degradation")
        else:
            print(f"  ⚠ Quality degradation high")
    else:
        print(f"\n❌ PMI Sparse failed quality gate: {pmi_ppl:.1f} PPL")


if __name__ == "__main__":
    main()
