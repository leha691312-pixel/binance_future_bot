"""
Train ML model (demo synthetic hoặc từ backtest trades)
Usage: python scripts/train_ml.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ml_agent import MLAgent

def main():
    ml = MLAgent()
    report = ml.train_demo()
    print("Training report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
