"""
Experiment Tracking and Comparison System
Tracks and compares results from different RecSys-LLM transfer experiments
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class ExperimentTracker:
    """Track and compare experiment results"""

    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        experiment_name: str,
        model_name: str,
        metrics: Dict,
        config: Dict,
        notes: Optional[str] = None
    ):
        """
        Log an experiment result

        Args:
            experiment_name: Name of the experiment (e.g., "wide_deep_v1")
            model_name: Model being tested (e.g., "Wide & Deep GPT")
            metrics: Dictionary of metrics (e.g., {"avg_time": 0.12, "speed_vs_baseline": 0.69})
            config: Dictionary of configuration parameters
            notes: Optional notes about the experiment
        """
        timestamp = datetime.now().isoformat()

        result = {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "model_name": model_name,
            "metrics": metrics,
            "config": config,
            "notes": notes
        }

        # Save to JSON file
        filename = f"{experiment_name}_{timestamp.replace(':', '-')}.json"
        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Experiment logged: {filepath}")
        return filepath

    def load_experiment(self, filepath: str) -> Dict:
        """Load a single experiment result"""
        with open(filepath, 'r') as f:
            return json.load(f)

    def load_all_experiments(self) -> List[Dict]:
        """Load all experiment results"""
        experiments = []
        for filepath in self.results_dir.glob("*.json"):
            experiments.append(self.load_experiment(filepath))
        return sorted(experiments, key=lambda x: x['timestamp'])

    def compare_experiments(
        self,
        baseline_name: str = "standard_gpt",
        metric_key: str = "avg_time"
    ):
        """
        Compare all experiments against a baseline

        Args:
            baseline_name: Name of the baseline experiment
            metric_key: Metric to compare (e.g., "avg_time", "parameters")
        """
        experiments = self.load_all_experiments()

        if not experiments:
            print("No experiments found!")
            return

        # Find baseline
        baseline = None
        for exp in experiments:
            if baseline_name in exp['experiment_name'].lower():
                baseline = exp
                break

        if not baseline:
            print(f"Baseline '{baseline_name}' not found!")
            return

        baseline_value = baseline['metrics'].get(metric_key)

        print(f"\n{'='*80}")
        print(f"Experiment Comparison (Baseline: {baseline['model_name']})")
        print(f"{'='*80}\n")
        print(f"{'Model':<30} {'Value':<15} {'vs Baseline':<15} {'Parameters':<15}")
        print(f"{'-'*80}")

        for exp in experiments:
            model_name = exp['model_name']
            value = exp['metrics'].get(metric_key, 'N/A')
            params = exp['metrics'].get('parameters', 'N/A')

            if isinstance(value, (int, float)) and baseline_value:
                relative = value / baseline_value
                print(f"{model_name:<30} {value:<15.4f} {relative:<15.2f}x {params:<15}")
            else:
                print(f"{model_name:<30} {str(value):<15} {'N/A':<15} {params:<15}")

        print(f"\n{'='*80}\n")

    def generate_report(self, output_file: str = "experiments/RESULTS.md"):
        """Generate a markdown report of all experiments"""
        experiments = self.load_all_experiments()

        if not experiments:
            print("No experiments to report!")
            return

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# Experiment Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary table
            f.write("## Summary\n\n")
            f.write("| Model | Avg Time (s) | Speed vs Baseline | Parameters | Notes |\n")
            f.write("|-------|-------------|-------------------|------------|-------|\n")

            for exp in experiments:
                model = exp['model_name']
                metrics = exp['metrics']
                avg_time = metrics.get('avg_time', 'N/A')
                speed_vs = metrics.get('speed_vs_baseline', 'N/A')
                params = metrics.get('parameters', 'N/A')
                notes = exp.get('notes', '')[:50]  # Truncate long notes

                f.write(f"| {model} | {avg_time} | {speed_vs} | {params} | {notes} |\n")

            # Detailed results
            f.write("\n## Detailed Results\n\n")

            for exp in experiments:
                f.write(f"### {exp['model_name']}\n\n")
                f.write(f"**Experiment**: {exp['experiment_name']}\n")
                f.write(f"**Timestamp**: {exp['timestamp']}\n\n")

                f.write("**Metrics**:\n")
                for key, value in exp['metrics'].items():
                    f.write(f"- {key}: {value}\n")

                f.write("\n**Configuration**:\n")
                for key, value in exp['config'].items():
                    f.write(f"- {key}: {value}\n")

                if exp.get('notes'):
                    f.write(f"\n**Notes**: {exp['notes']}\n")

                f.write("\n---\n\n")

        print(f"Report generated: {output_path}")


def example_usage():
    """Example of how to use the experiment tracker"""
    tracker = ExperimentTracker()

    # Log baseline experiment
    tracker.log_experiment(
        experiment_name="standard_gpt_baseline",
        model_name="Standard GPT",
        metrics={
            "avg_time": 0.0841,
            "speed_vs_baseline": 1.00,
            "parameters": 731264
        },
        config={
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 128,
            "block_size": 256,
            "batch_size": 8
        },
        notes="Baseline nanoGPT performance"
    )

    # Log Wide & Deep experiment
    tracker.log_experiment(
        experiment_name="wide_deep_v1",
        model_name="Wide & Deep GPT",
        metrics={
            "avg_time": 0.1220,
            "speed_vs_baseline": 0.69,
            "parameters": 997506
        },
        config={
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 128,
            "block_size": 256,
            "batch_size": 8,
            "wide_deep_enabled": True
        },
        notes="Separates memorization from generalization"
    )

    # Compare experiments
    tracker.compare_experiments(baseline_name="standard_gpt")

    # Generate report
    tracker.generate_report()


if __name__ == "__main__":
    print("Experiment Tracking System")
    print("=" * 80)
    print("\nUsage:")
    print("1. Import ExperimentTracker")
    print("2. Create instance: tracker = ExperimentTracker()")
    print("3. Log experiments: tracker.log_experiment(...)")
    print("4. Compare: tracker.compare_experiments()")
    print("5. Generate report: tracker.generate_report()")
    print("\nRun example_usage() to see it in action\n")

    # Run example
    response = input("Run example? (y/n): ")
    if response.lower() == 'y':
        example_usage()
