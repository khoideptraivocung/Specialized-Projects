import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.auto_ml import run_pipeline

def run():
    return run_pipeline()

if __name__ == "__main__":
    run()