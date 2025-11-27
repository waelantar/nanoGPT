# Fix Parameter Matching - Immediate Action Required

## The Core Problem

**Current (INVALID):**
- Standard GPT: 731,264 params
- Wide & Deep: 997,506 params (+36%)
- PMI Sparse: 859,264 params (+18%)

**This makes ALL comparisons meaningless.**

---

## Step 1: Calculate Target Architectures

### Target: 731,264 parameters (match baseline)

### Standard GPT (Baseline)
```python
config = GPTConfig(
    vocab_size=1000,
    n_embd=128,
    n_head=4,
    n_layer=3,
    block_size=64,
    dropout=0.1
)
# Result: 731,264 params ✓
```

### Wide & Deep GPT (FIX NEEDED)

**Current (WRONG): n_embd=128 → 997,506 params**

**Fixed (CORRECT): n_embd=96 → ~730,000 params**

```python
# Fix in: recsys_llm_research/wide_deep_nanogpt.py

class WideDeepNanoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        # BEFORE: n_embd passed directly (causes +36% params)
        # AFTER: Reduce n_embd to compensate for dual embeddings

        # Wide component: vocab_size * n_embd
        # Deep component: vocab_size * n_embd
        # Total embedding params: 2 * vocab_size * n_embd
        # vs Standard: vocab_size * n_embd

        # Solution: Use 75% of n_embd for each path
        adjusted_embd = int(n_embd * 0.75)

        self.wide_embd = WideDeepEmbedding(
            vocab_size, adjusted_embd, ...
        )
```

**Verification:**
```python
model = WideDeepNanoGPT(vocab_size=1000, n_embd=128, ...)
total_params = sum(p.numel() for p in model.parameters())
assert 730_000 < total_params < 732_000, f"Wrong param count: {total_params}"
```

### PMI Sparse GPT (FIX NEEDED)

**Current (WRONG): n_embd=128, n_layer=3 → 859,264 params**

**Option 1: Reduce n_layer**
```python
config = GPTConfig(
    vocab_size=1000,
    n_embd=128,
    n_head=4,
    n_layer=2,  # Down from 3
    block_size=64
)
# Should give ~730K params
```

**Option 2: Reduce n_embd**
```python
config = GPTConfig(
    vocab_size=1000,
    n_embd=110,  # Down from 128
    n_head=4,    # May need to adjust (must divide n_embd)
    n_layer=3,
    block_size=64
)
# Should give ~730K params
```

**Verification:**
```python
model = create_efficient_pmi_model(vocab_size=1000, embed_dim=110, ...)
total_params = sum(p.numel() for p in model.parameters())
assert 730_000 < total_params < 732_000, f"Wrong param count: {total_params}"
```

---

## Step 2: Update Benchmark Code

### Fix rigorous_benchmark.py

```python
def main():
    # Model configuration - PARAMETER MATCHED
    vocab_size = 1000
    seq_len = 64
    batch_size = 2

    # Target parameter count (match standard GPT)
    TARGET_PARAMS = 731_264

    # Standard GPT
    std_config = {
        'n_embd': 128,
        'n_head': 4,
        'n_layer': 3
    }

    # Wide & Deep (adjusted to match params)
    wd_config = {
        'n_embd': 96,  # REDUCED to compensate for dual embeddings
        'n_head': 4,
        'n_layer': 3
    }

    # PMI Sparse (adjusted to match params)
    pmi_config = {
        'n_embd': 110,  # REDUCED to match total params
        'n_head': 4,    # Must check divisibility
        'n_layer': 3
    }

    # VERIFY parameter matching
    std_model = create_standard_gpt(vocab_size, **std_config, block_size=seq_len)
    std_params = sum(p.numel() for p in std_model.parameters())

    wd_model = WideDeepNanoGPT(vocab_size, **wd_config, block_size=seq_len)
    wd_params = sum(p.numel() for p in wd_model.parameters())

    pmi_model = create_efficient_pmi_model(vocab_size, **pmi_config, max_seq_len=seq_len)
    pmi_params = sum(p.numel() for p in pmi_model.parameters())

    # ASSERT parameter matching (within 1% tolerance)
    tolerance = 0.01
    assert abs(wd_params - std_params) / std_params < tolerance, \
        f"Wide & Deep params {wd_params} != Standard {std_params}"
    assert abs(pmi_params - std_params) / std_params < tolerance, \
        f"PMI Sparse params {pmi_params} != Standard {std_params}"

    print(f"\n✓ PARAMETER MATCHING VERIFIED:")
    print(f"  Standard: {std_params:,}")
    print(f"  Wide & Deep: {wd_params:,} (Δ={wd_params-std_params:+,})")
    print(f"  PMI Sparse: {pmi_params:,} (Δ={pmi_params-std_params:+,})")
```

---

## Step 3: Quality Gates

### Perplexity Thresholds

Before claiming ANY results:

```python
# After training each model:
perplexity = calculate_perplexity(model, test_data)

# QUALITY GATE
BASELINE_PPL = 365  # From standard GPT
MAX_ACCEPTABLE_PPL = BASELINE_PPL * 1.1  # +10% max

if perplexity > MAX_ACCEPTABLE_PPL:
    print(f"❌ QUALITY GATE FAILED: {perplexity:.1f} > {MAX_ACCEPTABLE_PPL:.1f}")
    print("   Model is broken. Do not benchmark speed until this is fixed.")
    sys.exit(1)
else:
    print(f"✓ QUALITY GATE PASSED: {perplexity:.1f} ≤ {MAX_ACCEPTABLE_PPL:.1f}")
```

### Speed Thresholds

Only after passing quality gate:

```python
# Speed comparison
speedup = baseline_time / model_time

# SPEED GATE
MIN_ACCEPTABLE_SPEEDUP = 0.8  # 20% slower max

if speedup < MIN_ACCEPTABLE_SPEEDUP:
    print(f"⚠ Speed degradation: {speedup:.2f}x (> 20% slower)")
    print("   Must provide compelling benefit to justify overhead")
```

---

## Step 4: Retrain Everything

### Training Script

```python
# train_parameter_matched.py

import torch
from model import GPT, GPTConfig
from recsys_llm_research.wide_deep_nanogpt import WideDeepNanoGPT

# IDENTICAL training recipe for all models
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 3e-4,
    'max_iters': 5000,
    'eval_interval': 100,
    'eval_iters': 50,
    'optimizer': 'AdamW',
    'weight_decay': 0.1,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0
}

# Target: 731,264 parameters for ALL models
TARGET_PARAMS = 731_264

# Train Standard GPT
std_model = create_standard_gpt(
    vocab_size=1000,
    n_embd=128,
    n_head=4,
    n_layer=3,
    block_size=64
)
assert abs(sum(p.numel() for p in std_model.parameters()) - TARGET_PARAMS) < 1000
train_model(std_model, TRAINING_CONFIG, save_path='standard_gpt.pt')

# Train Wide & Deep (parameter-matched)
wd_model = WideDeepNanoGPT(
    vocab_size=1000,
    n_embd=96,  # ADJUSTED
    n_head=4,
    n_layer=3,
    block_size=64
)
assert abs(sum(p.numel() for p in wd_model.parameters()) - TARGET_PARAMS) < 1000
train_model(wd_model, TRAINING_CONFIG, save_path='wide_deep_gpt.pt')

# Train PMI Sparse (parameter-matched)
pmi_model = create_efficient_pmi_model(
    vocab_size=1000,
    embed_dim=110,  # ADJUSTED
    num_heads=4,
    num_layers=3,
    max_seq_len=64
)
assert abs(sum(p.numel() for p in pmi_model.parameters()) - TARGET_PARAMS) < 1000
train_model(pmi_model, TRAINING_CONFIG, save_path='pmi_sparse_gpt.pt')
```

---

## Step 5: Verification Checklist

### Before Running ANY Benchmarks

- [ ] All models have 731,264 ± 1% parameters
- [ ] All models trained with IDENTICAL recipe
- [ ] All models pass quality gate (perplexity < 400)
- [ ] Parameter matching verified programmatically
- [ ] Training logs show similar convergence

### After Training

- [ ] Standard GPT: perplexity ~365 (baseline)
- [ ] Wide & Deep: perplexity < 400 (within 10%)
- [ ] PMI Sparse: perplexity < 400 OR remove model

### After Benchmarking

- [ ] Speed comparisons are fair (matched params)
- [ ] Memory comparisons are fair (matched params)
- [ ] Quality is acceptable (< 10% degradation)

---

## Expected Timeline

### Day 1: Fix Architectures
- [ ] Calculate adjusted n_embd for Wide & Deep
- [ ] Calculate adjusted n_embd or n_layer for PMI Sparse
- [ ] Update model creation code
- [ ] Verify parameter counts programmatically

### Days 2-3: Retrain Models
- [ ] Train Standard GPT (baseline)
- [ ] Train Wide & Deep (parameter-matched)
- [ ] Train PMI Sparse (parameter-matched)
- [ ] Monitor perplexity convergence

### Day 4: Quality Gates
- [ ] Check final perplexities
- [ ] If any model > 400 perplexity → debug or remove
- [ ] Document training curves

### Day 5: Valid Benchmarking
- [ ] Run rigorous_benchmark.py with parameter-matched models
- [ ] Verify all comparisons are fair
- [ ] Generate honest results table

---

## Honest Results Template

Only after ALL above steps:

```markdown
## Parameter-Matched Results

All models trained with 731,264 parameters (±1%).

| Model | Params | Perplexity | Speed | Memory |
|-------|--------|-----------|-------|---------|
| Standard GPT | 731,264 | 365 | 1.00x | 100 MB |
| Wide & Deep | 731,264 | [X] | [Y]x | [Z] MB |
| PMI Sparse | 731,264 | [X] | [Y]x | [Z] MB |

**Quality Gates:**
- ✓ All models < 400 perplexity
- ✓ All models trained identically
- ✓ Fair comparison (matched parameters)

**Speed Analysis:**
- Wide & Deep: [explain why slower/faster]
- PMI Sparse: [prove sparsity or remove]
```

---

## STOP Points

If after retraining:

**Wide & Deep perplexity > 450:**
- ❌ Architecture is broken
- ❌ Do not publish
- ❌ Debug or abandon

**PMI Sparse perplexity > 450:**
- ❌ Sparsity destroying quality
- ❌ Remove from paper
- ❌ Document as failed attempt

**Any model 3x worse than baseline:**
- ❌ STOP immediately
- ❌ Fix root cause
- ❌ Do not benchmark speed

---

## Bottom Line

**DO NOT run benchmarks until:**
1. All models have ~731K params
2. All models retrained identically
3. All models pass quality gate (< 400 perplexity)

**The only valid next step:**
"Fixed parameters to 731K, retrained, perplexity results: [TABLE]"
