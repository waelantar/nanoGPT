import torch
import time
import numpy as np
from data_utils import create_language_modeling_data, verify_data_quality
from param_matcher import find_wide_deep_config, find_pmi_sparse_config, TARGET_PARAMS
from model import GPT, GPTConfig
from recsys_llm_research.wide_deep_nanogpt import WideDeepNanoGPT
from efficient_pmi_sparse_attention import create_efficient_pmi_model

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def create_standard_gpt(vocab_size, target_params):
    """Create GPT with exact parameter target"""
    # START with the correct dimensions
    config = GPTConfig(vocab_size=vocab_size)
    model = GPT(config)
    
    # Verify parameter count BEFORE training
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"  GPT parameters: {actual_params:,} (target: {target_params:,})")
    
    if abs(actual_params - target_params) / target_params > 0.05:
        raise ValueError(f"Parameter mismatch! {actual_params:,} vs {target_params:,}")
    
    return model

def train_validated(model, train_data, val_data, device, model_name, target_ppl=150, max_iters=500):
    """Train with aggressive quality gates"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    print(f"\nTraining {model_name} | Params: {count_params(model):,}")
    
    for iter in range(max_iters):
        model.train()
        total_loss = 0
        
        # Use smaller subset to save memory
        for x, y in train_data[:20]:  # Reduced from 100 to 20
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / 20
        ppl = np.exp(avg_loss)
        
        if iter % 50 == 0:  # Check more frequently
            print(f"  Iter {iter:4d} | Train PPL: {ppl:6.1f}")
            
            # CRITICAL: Abort if not learning
            if iter > 50 and ppl > 500:
                print(f"  ❌ ABORT: Model not learning (PPL={ppl:.1f})")
                return None, False
        
        # Early stopping if converged
        if ppl < target_ppl:
            print(f"  ✓ Early convergence at iter {iter} (PPL={ppl:.1f})")
            break
    
    # Final validation on smaller subset
    model.eval()
    val_ppl = evaluate_model(model, val_data[:10], device)  # Reduced from 50 to 10
    print(f"  Final Val PPL: {val_ppl:.1f}")
    
    return model, val_ppl < target_ppl * 1.5

def evaluate_model(model, data, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in data[:10]:  # Reduced from 50 to 10
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    return np.exp(np.mean(losses))

def benchmark_final(model, input_ids, device):
    model.eval()
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    # Warmup
    for _ in range(30):
        with torch.no_grad():
            _ = model(input_ids)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    times = []
    for _ in range(100):
        start = time.time()
        with torch.no_grad():
            _ = model(input_ids)
        times.append(time.time() - start)
    
    return np.mean(times) * 1000  # Return ms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*80)
    print("FINAL VALIDATED TRAINING")
    print("="*80)
    print(f"Target: {TARGET_PARAMS:,} params | Device: {device}")
    
    # Create and verify data
    train_data = create_language_modeling_data(1000, 64, 100, batch_size=4)  # Reduced size
    val_data = create_language_modeling_data(1000, 64, 20, batch_size=4)    # Reduced size
    
    if not verify_data_quality(train_data, 1000):
        print("❌ Data quality check failed. Exiting.")
        return
    
    results = {}
    
    # 1. Standard GPT
    print("\n1. STANDARD GPT")
    std_model = create_standard_gpt(1000, TARGET_PARAMS)
    std_model, std_pass = train_validated(std_model, train_data, val_data, device, "Standard GPT")
    results['Standard GPT'] = {'model': std_model, 'passed': std_pass}
    
    # 2. Wide & Deep (parameter-matched)
    print("\n2. WIDE & DEEP")
    n_embd, _ = find_wide_deep_config()
    wd_model = WideDeepNanoGPT(1000, n_embd, 4, 3, 64)
    print(f"  Wide & Deep parameters: {count_params(wd_model):,}")
    wd_model, wd_pass = train_validated(wd_model, train_data, val_data, device, "Wide & Deep")
    results['Wide & Deep'] = {'model': wd_model, 'passed': wd_pass}
    
    # 3. PMI Sparse
    print("\n3. PMI SPARSE")
    n_layer, _ = find_pmi_sparse_config()
    pmi_model = create_efficient_pmi_model(1000, 128, 4, n_layer, 64, top_k=32)
    print(f"  PMI Sparse parameters: {count_params(pmi_model):,}")
    pmi_model, pmi_pass = train_validated(pmi_model, train_data, val_data, device, "PMI Sparse")
    results['PMI Sparse'] = {'model': pmi_model, 'passed': pmi_pass}
    
    # Benchmark
    print("\n" + "="*80)
    print("BENCHMARKING")
    print("="*80)
    
    test_input = torch.randint(0, 1000, (8, 64)).to(device)
    for name, res in results.items():
        if res['passed'] and res['model'] is not None:
            time_ms = benchmark_final(res['model'], test_input, device)
            print(f"{name}: {time_ms:.2f}ms")

if __name__ == "__main__":
    main()