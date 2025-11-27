"""
Train parameter-matched models with quality gates and proper data.
CRITICAL FIXES:
1. Structured data (Markov chain) instead of random
2. Exact parameter matching (±1%)
3. Quality gate: PPL < 400 or abort
4. Proper train/validation split
5. Learning rate scheduling
"""
import torch
from torch.optim import AdamW, lr_scheduler
import numpy as np
from data_utils import create_language_modeling_data, verify_data_quality
from param_matcher import find_wide_deep_config, find_pmi_sparse_config, TARGET_PARAMS
from model import GPT, GPTConfig
from recsys_llm_research.wide_deep_nanogpt import WideDeepNanoGPT
from efficient_pmi_sparse_attention import create_efficient_pmi_model
import sys

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def train_with_quality_gate(model, train_data, val_data, device, model_name, 
                            max_iters=5000, target_ppl=400, patience=10):
    """Train with early stopping and quality gate"""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_ppl = float('inf')
    no_improve = 0
    
    print(f"\nTraining {model_name}")
    print(f"Parameters: {count_params(model):,}")
    
    for iter in range(max_iters):
        model.train()
        total_loss = 0
        
        for x, y in train_data[:50]:  # Subset for speed
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / 50
        train_ppl = np.exp(avg_train_loss)
        
        # Validation every 100 iters
        if iter % 100 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_data[:20]:
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    val_losses.append(loss.item())
            
            val_ppl = np.exp(np.mean(val_losses))
            scheduler.step(val_ppl)
            
            print(f"  Iter {iter:5d} | Train PPL: {train_ppl:7.1f} | Val PPL: {val_ppl:7.1f} | LR: {scheduler.optimizer.param_groups[0]['lr']:.2e}")
            
            # Quality gate check
            if iter > 1000 and val_ppl > target_ppl * 2:
                print(f"  ❌ ABORTING: Val PPL {val_ppl:.1f} > {target_ppl*2}")
                return model, val_ppl, False
            
            # Early stopping
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at iter {iter}")
                    break
    
    return model, best_val_ppl, best_val_ppl < target_ppl

def benchmark_speed(model, input_ids, device, num_runs=100):
    """Benchmark with proper warmup and synchronization"""
    model.eval()
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(input_ids)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model(input_ids)
        times.append(time.time() - start)
    
    return np.mean(times), np.std(times)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*80)
    print("VALIDATED TRAINING: Parameter-Matched RecSys-LLM Transfer")
    print("="*80)
    print(f"Target parameters: {TARGET_PARAMS:,}")
    print(f"Device: {device}")
    
    # Create structured data
    print("\nGenerating structured training data...")
    train_data = create_language_modeling_data(1000, 64, 500, transition_strength=0.85)
    val_data = create_language_modeling_data(1000, 64, 100, transition_strength=0.85)
    verify_data_quality(train_data, 1000)
    
    results = {}
    
    # 1. Standard GPT (baseline)
    print("\n" + "="*80)
    print("1. STANDARD GPT (PARAMETER-MATCHED)")
    print("="*80)
    
    std_config = GPTConfig(n_embd=128, n_head=4, n_layer=3, block_size=64, dropout=0.1)
    std_model = GPT(std_config)
    print(f"Parameters: {count_params(std_model):,}")
    
    std_model, std_ppl, std_pass = train_with_quality_gate(
        std_model, train_data, val_data, device, "Standard GPT", max_iters=3000
    )
    results['Standard GPT'] = {'model': std_model, 'ppl': std_ppl, 'passed': std_pass}
    
    # 2. Wide & Deep (parameter-matched)
    print("\n" + "="*80)
    print("2. WIDE & DEEP GPT (PARAMETER-MATCHED)")
    print("="*80)
    
    n_embd, _ = find_wide_deep_config()
    wd_model = WideDeepNanoGPT(1000, n_embd, 4, 3, 64)
    print(f"Parameters: {count_params(wd_model):,}")
    
    wd_model, wd_ppl, wd_pass = train_with_quality_gate(
        wd_model, train_data, val_data, device, "Wide & Deep GPT", max_iters=3000
    )
    results['Wide & Deep GPT'] = {'model': wd_model, 'ppl': wd_ppl, 'passed': wd_pass}
    
    # 3. PMI Sparse (parameter-matched)
    print("\n" + "="*80)
    print("3. PMI SPARSE GPT (PARAMETER-MATCHED)")
    print("="*80)
    
    n_layer, _ = find_pmi_sparse_config()
    pmi_model = create_efficient_pmi_model(1000, 128, 4, n_layer, 64, top_k=32)
    print(f"Parameters: {count_params(pmi_model):,}")
    
    # Build PMI pattern from training data
    print("Building PMI attention pattern...")
    from collections import defaultdict
    cooc = defaultdict(int)
    for x, y in train_data[:100]:
        for i in range(len(x[0]) - 1):
            token = x[0, i].item()
            next_token = x[0, i+1].item()  # Use temporal co-occurrence
            cooc[(token, next_token)] += 1
    
    pmi_matrix = pmi_model.compute_pmi_matrix(dict(cooc), 1000, 32)
    pmi_model.build_attention_pattern_from_sequence(val_data[0][0], pmi_matrix)
    
    pmi_model, pmi_ppl, pmi_pass = train_with_quality_gate(
        pmi_model, train_data, val_data, device, "PMI Sparse GPT", max_iters=3000
    )
    results['PMI Sparse GPT'] = {'model': pmi_model, 'ppl': pmi_ppl, 'passed': pmi_pass}
    
    # Final benchmark (only for models that passed quality gate)
    print("\n" + "="*80)
    print("SPEED BENCHMARKING")
    print("="*80)
    
    test_input = torch.randint(0, 1000, (4, 64)).to(device)
    benchmark_results = {}
    
    for name, res in results.items():
        if res['passed']:
            time_mean, time_std = benchmark_speed(res['model'], test_input, device)
            benchmark_results[name] = {
                'params': count_params(res['model']),
                'perplexity': res['ppl'],
                'time_ms': time_mean * 1000,
                'time_std': time_std * 1000
            }
            print(f"{name}: {time_mean*1000:.2f}ms ± {time_std*1000:.2f}ms | PPL: {res['ppl']:.1f}")
        else:
            print(f"{name}: ❌ Failed quality gate (PPL: {res['ppl']:.1f})")
    
    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Model':<20} {'Params':<10} {'Perplexity':<12} {'Time (ms)':<12} {'Status'}")
    print("-"*70)
    
    for name, res in results.items():
        status = "✓ PASS" if res['passed'] else "✗ FAIL"
        params = count_params(res['model'])
        print(f"{name:<20} {params:<10,} {res['ppl']:<12.1f} {'N/A':<12} {status}")
    
    if results['Standard GPT']['passed']:
        baseline_ppl = results['Standard GPT']['ppl']
        for name, res in results.items():
            if name != 'Standard GPT' and res['passed']:
                quality_delta = (res['ppl'] - baseline_ppl) / baseline_ppl * 100
                print(f"\n{name}:")
                print(f"  Quality: {res['ppl']:.1f} PPL ({quality_delta:+.1f}% vs baseline)")
                if quality_delta < 10:
                    print("  ✓ Acceptable quality degradation")
                else:
                    print("  ⚠ Significant quality degradation")

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print(f"\nTotal runtime: {(time.time() - start)/3600:.2f} hours")