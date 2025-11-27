# Experiment Tracking

This directory contains tools for tracking and comparing RecSys-LLM transfer experiments.

## Quick Start

### 1. Track an Experiment

```python
from experiments.compare_results import ExperimentTracker

tracker = ExperimentTracker()

# Log your experiment
tracker.log_experiment(
    experiment_name="my_experiment_v1",
    model_name="My Custom Model",
    metrics={
        "avg_time": 0.15,
        "speed_vs_baseline": 0.56,
        "parameters": 850000
    },
    config={
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "your_custom_param": "value"
    },
    notes="Optional notes about the experiment"
)
```

### 2. Compare Experiments

```python
# Compare all experiments against baseline
tracker.compare_experiments(
    baseline_name="standard_gpt",
    metric_key="avg_time"
)
```

Output example:
```
================================================================================
Experiment Comparison (Baseline: Standard GPT)
================================================================================

Model                          Value           vs Baseline     Parameters
--------------------------------------------------------------------------------
Standard GPT                   0.0841          1.00x           731264
Wide & Deep GPT                0.1220          1.45x           997506
PMI Sparse GPT                 0.5403          6.42x           859264
```

### 3. Generate Report

```python
# Generate markdown report of all experiments
tracker.generate_report(output_file="experiments/RESULTS.md")
```

## Directory Structure

```
experiments/
├── README.md              # This file
├── compare_results.py     # Experiment tracking system
├── results/               # JSON files with experiment results
│   ├── experiment_1.json
│   ├── experiment_2.json
│   └── ...
└── RESULTS.md            # Generated markdown report
```

## Running Benchmarks

### 1. Standard Benchmark
```bash
python accurate_benchmark.py
```

This will:
- Run all model variants
- Save results to `experiments/results/`
- Generate comparison report

### 2. Individual Model Tests

Test specific models:
```python
# Test Wide & Deep model
from recsys_llm_research.wide_deep_nanogpt import create_wide_deep_model
model = create_wide_deep_model(vocab_size=50257, n_embd=128, n_head=4, n_layer=4)
# Run your tests...
```

## Experiment Metadata

Each experiment should include:

**Metrics** (required):
- `avg_time`: Average inference time in seconds
- `speed_vs_baseline`: Relative speed compared to baseline
- `parameters`: Total model parameters

**Config** (required):
- `n_layer`: Number of transformer layers
- `n_head`: Number of attention heads
- `n_embd`: Embedding dimension
- `block_size`: Maximum sequence length
- `batch_size`: Batch size used
- Any model-specific parameters

**Notes** (optional):
- Description of the experiment
- Special observations
- Issues encountered

## Viewing Results

### Command Line
```bash
python experiments/compare_results.py
```

### In Code
```python
from experiments.compare_results import ExperimentTracker

tracker = ExperimentTracker()

# Load all experiments
experiments = tracker.load_all_experiments()

# Print summary
for exp in experiments:
    print(f"{exp['model_name']}: {exp['metrics']['avg_time']:.4f}s")
```

## Best Practices

1. **Always log baseline first**: Run standard GPT and log results before testing variations
2. **Use consistent configs**: Keep batch size, sequence length, etc. constant across experiments
3. **Add meaningful notes**: Document why an experiment succeeded or failed
4. **Version experiments**: Use version numbers in experiment names (e.g., "wide_deep_v2")
5. **Track hardware**: Note if running on different hardware (CPU vs GPU)

## Example Workflow

```python
from experiments.compare_results import ExperimentTracker

tracker = ExperimentTracker()

# 1. Run and log baseline
baseline_metrics = run_benchmark(standard_model)
tracker.log_experiment(
    experiment_name="standard_gpt_baseline",
    model_name="Standard GPT",
    metrics=baseline_metrics,
    config=baseline_config,
    notes="Baseline on CPU"
)

# 2. Run and log your variant
variant_metrics = run_benchmark(my_variant_model)
tracker.log_experiment(
    experiment_name="my_variant_v1",
    model_name="My Variant Model",
    metrics=variant_metrics,
    config=variant_config,
    notes="Testing new architecture"
)

# 3. Compare
tracker.compare_experiments()

# 4. Generate report
tracker.generate_report()
```

## Contributing Results

When adding new experiments:
1. Run the experiment with proper warmup
2. Log results using ExperimentTracker
3. Verify results are saved in `experiments/results/`
4. Update RESULTS.md using `generate_report()`
5. Commit both JSON results and updated RESULTS.md
