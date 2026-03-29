# Next Steps - Action Plan

## Current Status: Debugging Phase (Week 5-6)

The research is NOT publication-ready. Critical issues have been identified and partially addressed. This document outlines the remaining work.

---

## Immediate Priorities (This Week)

### 1. Run Rigorous Benchmark
```bash
python rigorous_benchmark.py
```

**What this does**:
- ✅ Reports accurate parameter counts
- ✅ Measures perplexity on synthetic test data
- ✅ Tracks memory usage (GPU/CPU)
- ✅ Profiles attention operations to verify sparsity
- ✅ Provides honest assessment

**Expected outcome**: Clear diagnosis of what's broken

### 2. Fix Wide & Deep Parallelization

**Current issue**: 21% slower than baseline (should be <5%)

**Root cause**: Sequential computation of wide and deep paths

**Fix location**: `recsys_llm_research/wide_deep_embedding.py`

**Required changes**:
```python
# Current (SLOW):
class WideDeepEmbedding(nn.Module):
    def forward(self, x):
        wide = self.static_embedding(x)
        deep = self.dynamic_embedding(x)
        combined = self.combine_layer(torch.cat([wide, deep], dim=-1))
        return combined  # Sequential bottleneck

# Fixed (FAST):
class WideDeepEmbedding(nn.Module):
    def forward(self, x):
        # Parallel computation
        wide = self.static_embedding(x)
        deep = self.dynamic_embedding(x)
        # Fused operation
        return torch.addcmul(wide, deep, self.alpha)
```

**Verification**:
- Re-run `python rigorous_benchmark.py`
- Speedup should be 0.95x or better (not 0.79x)

### 3. Add Acceptance Rate for Speculative Decoding

**What's missing**: Measure how many draft tokens are accepted

**Add to `rigorous_benchmark.py`**:
```python
def measure_acceptance_rate(draft_model, target_model, prompts, num_samples=100):
    """
    Measure acceptance rate for speculative decoding

    Returns:
        acceptance_rate (float): Percentage of draft tokens accepted (should be 70-85%)
    """
    total_drafted = 0
    total_accepted = 0

    for prompt in prompts:
        # Draft 5 tokens
        draft_tokens = draft_model.generate(prompt, max_new_tokens=5)
        # Verify with target model
        accepted = target_model.verify_and_correct(prompt, draft_tokens)

        total_drafted += len(draft_tokens)
        total_accepted += sum(draft_tokens == accepted)

    acceptance_rate = (total_accepted / total_drafted) * 100
    return acceptance_rate
```

**Expected result**: 70-85% for good draft model, <50% means draft model is poor

---

## Medium-Term Tasks (Week 7-8)

### 4. Real Test Data - WikiText or Similar

**Current issue**: Using synthetic random data for perplexity

**Required**: Test on actual language modeling benchmark

**Options**:
- WikiText-103 (standard LM benchmark)
- Penn Treebank
- Shakespeare corpus (already in repo)

**Implementation**:
```python
# In rigorous_benchmark.py
from datasets import load_dataset

def load_wikitext_test():
    """Load WikiText-103 test set"""
    dataset = load_dataset('wikitext', 'wikitext-103-v1', split='test')
    # Tokenize and prepare batches
    return data_loader

# Then use:
perplexity = calculate_perplexity(model, load_wikitext_test(), device)
```

### 5. Ablation Studies

**What's missing**: Understanding which components matter

**Required experiments**:

#### A. Wide & Deep Ablations
- Baseline: Standard embedding
- Wide only: Static embedding only
- Deep only: Dynamic embedding only
- Wide + Deep: Full model

**Measures**: Speed, perplexity, memory

#### B. PMI Sparse Ablations
- Dense attention (baseline)
- Top-k=8 (very sparse)
- Top-k=16 (moderate)
- Top-k=32 (less sparse)

**Measures**: Speed, perplexity, memory

#### C. Speculative Decoding Ablations
- No drafting (baseline)
- Bigram draft model
- Trigram draft model
- Small transformer draft

**Measures**: Speed, acceptance rate, generation quality

**Implementation**:
```python
def run_ablation_study(configs, test_data):
    """
    Run ablation study across different configurations

    Args:
        configs: List of model configurations to test
        test_data: Held-out test set

    Returns:
        DataFrame with results for each configuration
    """
    results = []
    for config in configs:
        model = create_model(config)
        metrics = benchmark_model_comprehensive(model, test_data)
        results.append(metrics)
    return pd.DataFrame(results)
```

### 6. Fix or Remove PMI Sparse Attention

**Two options**:

#### Option A: Fix it (Hard - requires CUDA/Triton)
Implement true sparse attention using custom kernel:
```python
import triton
import triton.language as tl

@triton.jit
def sparse_attention_kernel(Q, K_sparse, indices, ...):
    """Custom kernel for O(n*k) sparse attention"""
    # Triton implementation of sparse matmul
    pass
```

**Effort**: 2-3 weeks
**Outcome**: Might achieve 2-3x speedup

#### Option B: Be honest about failure (Easy - 1 day)
Update RESEARCH_SUMMARY.md:
```markdown
## PMI Sparse Attention: Failed Implementation

**Attempted**: Sparse attention based on token co-occurrence
**Result**: 25x slower than baseline
**Reason**: PyTorch doesn't have sparse attention primitives
**Learning**: Custom GPU kernels required for sparse operations

This is a valuable **negative result** that guides future research.
```

**Recommendation**: Option B for now, Option A as future work

---

## Long-Term Goals (Month 2-3)

### 7. Scale Up to Real Model Sizes

**Current**: Tiny models (0.7-1M parameters)
**Required**: At least 10-100M parameters

Benefits may only appear at scale:
- Wide & Deep: Parameter sharing matters more
- Sparse attention: O(n²) vs O(n*k) matters more
- Memory efficiency: Becomes critical

### 8. Real-World Evaluation

**Current**: Forward pass speed only
**Required**: Full training/finetuning

Tasks to evaluate:
- Language modeling (perplexity on WikiText)
- Few-shot learning (GPT-3 style prompts)
- Downstream tasks (classification, QA)

### 9. Publication-Ready Writeup

Once all fixes complete:
- Honest abstract (no overclaiming)
- Proper related work (cite all prior art)
- Detailed methodology (reproducible)
- Ablation studies (show what matters)
- Limitations section (be explicit)

---

## Decision Points

### Should We Continue This Research?

**If PMI sparse stays 25x slower after fixes**:
- ✅ Still publish as negative result
- ✅ Explains why naive approaches fail
- ✅ Guides future research toward GPU kernels

**If Wide & Deep can't be optimized**:
- ✅ Document the overhead honestly
- ✅ Test at larger scales (may improve)
- ✅ Provide as reference implementation

**If speculative decode works well**:
- ✅ Measure acceptance rate thoroughly
- ✅ Compare draft model quality
- ✅ Cite original paper prominently

### Pivot Options

If current approaches don't pan out:

#### Option 1: Focus on engineering
- Optimize what exists
- Create high-quality reference implementations
- Contribute to open-source (Triton kernels)

#### Option 2: Explore other RecSys techniques
- Factorization machines for attention
- Neural collaborative filtering for token selection
- Deep & Cross networks for feature interactions

#### Option 3: Hybrid approaches
- Combine Wide & Deep + sparse attention
- Mix speculative decode + retrieval
- Ensemble methods

---

## Quality Checklist (Before Claiming Results)

### Must Have
- [ ] Accurate parameter counts (DONE ✅)
- [ ] Perplexity on real test set (TODO 🔧)
- [ ] Memory usage tracking (DONE ✅)
- [ ] Acceptance rate for speculative (TODO 🔧)
- [ ] Proper citations (DONE ✅)
- [ ] Ablation studies (TODO 🔧)
- [ ] Profile showing actual sparsity (DONE ✅)

### Should Have
- [ ] WikiText-103 evaluation
- [ ] Comparison to FlashAttention/other baselines
- [ ] Statistical significance tests
- [ ] Multiple random seeds
- [ ] Hardware utilization metrics

### Nice to Have
- [ ] Visualization of attention patterns
- [ ] Case studies of generation quality
- [ ] Comparison across model sizes
- [ ] Deployment considerations

---

## Timeline

### This Week (Week 5)
- [x] Create `rigorous_benchmark.py` with fixes
- [x] Document all issues in `CRITICAL_ISSUES_AND_FIXES.md`
- [ ] Fix Wide & Deep parallelization
- [ ] Run benchmark and analyze results

### Next Week (Week 6)
- [ ] Add acceptance rate measurement
- [ ] Implement WikiText-103 evaluation
- [ ] Start ablation studies
- [ ] Decide on PMI sparse (fix or document failure)

### Week 7-8
- [ ] Complete all ablation studies
- [ ] Test at larger model scales
- [ ] Write honest conclusions
- [ ] Prepare for review

### Month 2 (If continuing)
- [ ] Implement Triton kernels for sparse attention
- [ ] Scale to 100M parameter models
- [ ] Comprehensive evaluation suite
- [ ] Draft publication

---

## How to Use This Document

1. **Weekly review**: Check off completed items
2. **Priority focus**: Work on "Immediate Priorities" first
3. **Decision points**: Evaluate whether to pivot or continue
4. **Quality check**: Use checklist before claiming results

## Questions to Answer Weekly

1. Is PMI sparse getting faster or should we abandon it?
2. Is Wide & Deep overhead acceptable or can we fix it?
3. What is the acceptance rate for speculative decoding?
4. Are we ready to test on real data (WikiText)?
5. Should we pivot to different RecSys techniques?

---

## Contact/Notes

If you need help:
- PMI sparse: Check `efficient_pmi_sparse_attention.py`
- Wide & Deep: Check `recsys_llm_research/wide_deep_embedding.py`
- Benchmark: Run `python rigorous_benchmark.py`
- Issues: Read `CRITICAL_ISSUES_AND_FIXES.md`

**Remember**: Research is iterative. Negative results are valuable. Honesty > Hype.
