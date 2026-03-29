# Setup Instructions

## Quick Fix for Virtual Environment Issue

If you see the error:
```
-bash: /mnt/c/Users/.../recsys_env/bin/python: No such file or directory
```

This means your shell is still trying to use the deleted virtual environment. Follow these steps:

### Step 1: Deactivate the Old Environment

```bash
deactivate
```

### Step 2: Create a New Virtual Environment (Recommended)

```bash
# Create new virtual environment
python -m venv venv

# Activate it (Linux/Mac)
source venv/bin/activate

# Or on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### Step 4: Run Benchmarks

```bash
python accurate_benchmark.py
```

## Alternative: Use System Python Directly

If you don't want to use a virtual environment:

```bash
# Just run with system Python
python accurate_benchmark.py

# Or specify the full path
/usr/bin/python3 accurate_benchmark.py
```

## Verify Setup

```bash
# Check Python version
python --version

# Check if PyTorch is installed
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check if dependencies are available
python -c "import numpy, transformers; print('Dependencies OK')"
```

## Running Research Experiments

Once setup is complete:

```bash
# Run comprehensive benchmark
python accurate_benchmark.py

# Track experiments
python experiments/compare_results.py

# View research summary
cat RESEARCH_SUMMARY.md
```

## Troubleshooting

### Issue: Module not found errors
**Solution**: Install missing dependencies
```bash
pip install <missing_module>
```

### Issue: CUDA/GPU errors
**Solution**: The code will automatically fall back to CPU. To explicitly use CPU:
```bash
python accurate_benchmark.py --device=cpu
```

### Issue: Permission errors
**Solution**: Make sure you have write permissions in the experiments/results directory
```bash
mkdir -p experiments/results
chmod -R u+w experiments/
```

## Development Setup

For development work on the research code:

```bash
# Install in development mode
pip install -e .

# Install additional dev dependencies
pip install pytest black flake8 mypy

# Run tests (if available)
pytest tests/
```
