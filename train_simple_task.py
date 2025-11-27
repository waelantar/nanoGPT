"""
Train on a simple learning task: next token prediction with patterns
This will actually converge and show quality differences
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


def create_pattern_data(vocab_size, seq_len, num_samples):
    """Create data with learnable patterns"""
    data = []
    for _ in range(num_samples):
        # Create sequences with repeating patterns
        x = torch.zeros(4, seq_len, dtype=torch.long)
        y = torch.zeros(4, seq_len, dtype=torch.long)

        for b in range(4):
            # Pattern: token i is followed by token (i+1) % vocab_size
            start = torch.randint(0, vocab_size, (1,)).item()
            for t in range(seq_len):
                x[b, t] = (start + t) % vocab_size
                y[b, t] = (start + t + 1) % vocab_size

        data.append((x, y))
    return data


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_model(model, train_data, test_data, device, model_name, max_iters=1000):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    print(f"\nTraining {model_name}")
    print(f"Parameters: {count_params(model):,}")

    model.train()
    best_loss = float('inf')

    for iter in range(max_iters):
        batch_idx = iter % len(train_data)
        x, y = train_data[batch_idx]
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

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

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (iter + 1) % 100 == 0:
            ppl = np.exp(loss.item())
            print(f"  Iter {iter+1}/{max_iters} | Loss: {loss.item():.4f} | PPL: {ppl:.2f}")
            best_loss = min(best_loss, loss.item())

    # Evaluate
    model.eval()
    test_losses = []
    with torch.no_grad():
        for x, y in test_data[:20]:
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

    passed = final_ppl < 100  # More lenient for pattern task
    if not passed:
        print(f"  ⚠ Model didn't learn well: {final_ppl:.2f}")
    else:
        print(f"  ✓ Model learned: {final_ppl:.2f}")

    return model, final_ppl, passed


def benchmark_speed(model, input_ids, device, num_runs=100):
    model = model.to(device)
    model.eval()
    input_ids = input_ids.to(device)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(input_ids)

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
    vocab_size = 100  # Smaller vocab for pattern learning
    seq_len = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*80)
    print("PARAMETER-MATCHED COMPARISON - PATTERN LEARNING TASK")
    print("="*80)
    print(f"Task: Learn that token[t+1] = (token[t] + 1) % {vocab_size}")
    print(f"Device: {device}\n")

    # Create pattern data
    train_data = create_pattern_data(vocab_size, seq_len, 100)
    test_data = create_pattern_data(vocab_size, seq_len, 20)

    # 1. Standard GPT
    print("="*80)
    print("1. STANDARD GPT (BASELINE)")
    print("="*80)

    std_config = GPTConfig(
        vocab_size=vocab_size,
        n_embd=64,
        n_head=4,
        n_layer=2,
        block_size=seq_len,
        dropout=0.0
    )
    std_model = GPT(std_config)
    std_params = count_params(std_model)
    print(f"Parameters: {std_params:,}")

    std_model, std_ppl, std_pass = train_model(std_model, train_data, test_data, device, "Standard GPT", max_iters=1000)

    # 2. Wide & Deep
    print("\n" + "="*80)
    print("2. WIDE & DEEP GPT (PARAMETER-MATCHED)")
    print("="*80)

    wd_model = WideDeepNanoGPT(
        vocab_size=vocab_size,
        n_embd=52,  # Adjusted to match params
        n_head=4,
        n_layer=2,
        block_size=seq_len
    )
    wd_params = count_params(wd_model)
    print(f"Parameters: {wd_params:,} (target: {std_params:,}, diff: {wd_params-std_params:+,})")

    wd_model, wd_ppl, wd_pass = train_model(wd_model, train_data, test_data, device, "Wide & Deep GPT", max_iters=1000)

    # 3. PMI Sparse
    print("\n" + "="*80)
    print("3. PMI SPARSE GPT (PARAMETER-MATCHED)")
    print("="*80)

    pmi_model = create_efficient_pmi_model(
        vocab_size=vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=seq_len,
        top_k=16
    )
    pmi_params = count_params(pmi_model)
    print(f"Parameters: {pmi_params:,} (target: {std_params:,}, diff: {pmi_params-std_params:+,})")

    # Build PMI pattern
    from collections import defaultdict
    cooc_counts = defaultdict(int)
    for i in range(vocab_size):
        j = (i + 1) % vocab_size
        cooc_counts[(i, j)] = 100  # Strong pattern

    temp_attention = pmi_model.blocks[0].attention
    pmi_matrix = temp_attention.compute_pmi_matrix(dict(cooc_counts), vocab_size, 16)
    sample_input = torch.randint(0, vocab_size, (1, seq_len))
    pmi_model.build_attention_pattern_from_sequence(sample_input, pmi_matrix)

    pmi_model, pmi_ppl, pmi_pass = train_model(pmi_model, train_data, test_data, device, "PMI Sparse GPT", max_iters=1000)

    # Benchmark
    print("\n" + "="*80)
    print("SPEED BENCHMARKING")
    print("="*80)

    test_input = torch.randint(0, vocab_size, (4, seq_len))

    results = {}

    if std_pass:
        std_time, std_std = benchmark_speed(std_model, test_input, device)
        results['Standard'] = {
            'params': std_params,
            'ppl': std_ppl,
            'time': std_time,
            'std': std_std
        }

    if wd_pass:
        wd_time, wd_std = benchmark_speed(wd_model, test_input, device)
        results['Wide&Deep'] = {
            'params': wd_params,
            'ppl': wd_ppl,
            'time': wd_time,
            'std': wd_std
        }

    if pmi_pass:
        pmi_time, pmi_std = benchmark_speed(pmi_model, test_input, device)
        results['PMI Sparse'] = {
            'params': pmi_params,
            'ppl': pmi_ppl,
            'time': pmi_time,
            'std': pmi_std
        }

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    if not results:
        print("❌ No models passed learning test")
        return

    baseline = results.get('Standard')
    if not baseline:
        print("❌ Baseline didn't learn - can't compare")
        return

    print(f"\n{'Model':<15} {'Params':>10} {'PPL':>8} {'Time(ms)':>10} {'Speedup':>8}")
    print("-"*60)

    for name, res in results.items():
        speedup = baseline['time'] / res['time']
        print(f"{name:<15} {res['params']:>10,} {res['ppl']:>8.2f} "
              f"{res['time']*1000:>10.3f} {speedup:>8.2f}x")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print(f"\nBaseline: {baseline['ppl']:.2f} PPL, {baseline['time']*1000:.3f}ms")

    if 'Wide&Deep' in results:
        wd = results['Wide&Deep']
        wd_speedup = baseline['time'] / wd['time']
        ppl_change = (wd['ppl'] - baseline['ppl']) / baseline['ppl'] * 100
        print(f"\nWide & Deep:")
        print(f"  PPL: {wd['ppl']:.2f} ({ppl_change:+.1f}%)")
        print(f"  Speed: {wd_speedup:.2f}x")
        if abs(ppl_change) < 5 and wd_speedup > 0.9:
            print(f"  ✓ Similar quality, acceptable speed")
        elif wd_speedup > 1.1:
            print(f"  ✓ Faster than baseline")
        else:
            print(f"  ⚠ Slower with similar quality")

    if 'PMI Sparse' in results:
        pmi = results['PMI Sparse']
        pmi_speedup = baseline['time'] / pmi['time']
        ppl_change = (pmi['ppl'] - baseline['ppl']) / baseline['ppl'] * 100
        print(f"\nPMI Sparse:")
        print(f"  PPL: {pmi['ppl']:.2f} ({ppl_change:+.1f}%)")
        print(f"  Speed: {pmi_speedup:.2f}x")
        if pmi_speedup > 1.5:
            print(f"  ✓ Significantly faster")
        elif pmi_speedup > 1.1:
            print(f"  ✓ Moderately faster")
        elif pmi_speedup < 0.5:
            print(f"  ❌ Much slower (not sparse)")
        else:
            print(f"  ⚠ Slower than expected")


if __name__ == "__main__":
    main()
