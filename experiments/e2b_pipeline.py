"""Continue the conditional study, opening acceptance data once methods are fixed."""
import json
import subprocess
from pathlib import Path


def invoke(*arguments):
    command = ["/home/user/Conda/CDF_FFT/bin/python", "-u", *arguments]
    subprocess.run(command, check=True)


def main():
    if Path("results/e2b/gate/opened.json").exists():
        if not Path("results/e2b/gate/run.json").exists():
            invoke("experiments/e2b_direction_comparison.py", "--phase", "gate")
        Path("logs/e2b_experiments.done").write_text("All E2b experiments completed\n")
        return
    invoke("-m", "pytest", "-q", "tests/")
    # Repeat the same pre-training factorial settings with full search-attempt logging.
    invoke("experiments/e2b_factorial_error.py")
    invoke("experiments/e2b_direction_comparison.py", "--phase", "development")
    trigger = json.loads(Path("results/e2b/trigger.json").read_text())["trigger"]
    if trigger:
        invoke("experiments/e2b_estimator_training.py", "--effective")
        invoke("experiments/e2b_direction_comparison.py", "--phase", "effective_development")
    invoke("experiments/e2b_direction_comparison.py", "--phase", "gate")
    Path("logs/e2b_experiments.done").write_text("All E2b experiments completed\n")


if __name__ == "__main__":
    main()
