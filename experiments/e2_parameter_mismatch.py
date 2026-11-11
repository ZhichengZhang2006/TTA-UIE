import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.e1b_output_matched import main

if __name__ == "__main__":
    main(synthetic=True)
