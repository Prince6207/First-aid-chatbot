"""
Training Script
Train the Bayesian Network on the Mendeley medical dataset.

Usage:
    # Train on real Mendeley dataset:
    python scripts/train_bn.py --data data/mendeley_dataset.csv --output models/bn_model.pkl

    # Create and train on demo dataset (no real data needed):
    python scripts/train_bn.py --demo

    # Evaluate the trained model:
    python scripts/train_bn.py --demo --evaluate
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reasoner.bn_trainer import BayesianNetworkTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def evaluate_model(model_path: Path, test_data_path: Path):
    """Quick evaluation of trained model."""
    import pickle
    import pandas as pd
    import numpy as np
    from pgmpy.inference import VariableElimination

    logger.info("Evaluating model...")

    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    model = saved["model"]
    metadata = saved["metadata"]

    ve = VariableElimination(model)
    df = pd.read_csv(test_data_path)

    if 'disease' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'disease'})

    symptom_cols = metadata["symptoms"]
    correct = 0
    total = 0
    top3_correct = 0

    sample_size = min(50, len(df))
    for _, row in df.sample(n=sample_size, random_state=42).iterrows():
        true_disease = row['disease']
        evidence = {}
        for sym in symptom_cols:
            if sym in row and pd.notna(row[sym]):
                val = int(row[sym])
                if val in [0, 1]:
                    evidence[sym] = val

        try:
            result = ve.query(
                variables=["disease"],
                evidence=evidence,
                show_progress=False
            )
            states = result.state_names["disease"]
            probs = dict(zip(states, result.values.flatten()))
            top_disease = max(probs, key=probs.get)
            top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
            top3_names = [d for d, _ in top3]

            if top_disease == true_disease:
                correct += 1
            if true_disease in top3_names:
                top3_correct += 1
            total += 1
        except Exception:
            pass

    if total > 0:
        acc = correct / total * 100
        top3_acc = top3_correct / total * 100
        logger.info(f"Evaluation on {total} samples:")
        logger.info(f"  Top-1 Accuracy: {acc:.1f}%")
        logger.info(f"  Top-3 Accuracy: {top3_acc:.1f}%")
        return acc, top3_acc
    return 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(description="Train the MedDecide Bayesian Network")
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to the Mendeley CSV dataset"
    )
    parser.add_argument(
        "--output", type=Path, default=MODELS_DIR / "bn_model.pkl",
        help="Output path for trained model (default: models/bn_model.pkl)"
    )
    parser.add_argument(
        "--laplace-alpha", type=float, default=1.0,
        help="Laplace smoothing equivalent sample size (default: 1.0)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate and train on a demo synthetic dataset"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate the model after training"
    )
    args = parser.parse_args()

    trainer = BayesianNetworkTrainer(laplace_alpha=args.laplace_alpha)

    if args.demo or args.data is None:
        logger.info("Creating synthetic demo dataset...")
        data_path = trainer.create_demo_dataset()
        logger.info(f"Demo dataset created at: {data_path}")
    else:
        data_path = args.data
        if not data_path.exists():
            logger.error(f"Dataset not found: {data_path}")
            sys.exit(1)

    logger.info(f"Training on: {data_path}")
    logger.info(f"Output: {args.output}")

    model = trainer.train(data_path, args.output)

    # Print model summary
    meta_path = args.output.with_suffix('.meta.json')
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        logger.info(f"\n{'='*50}")
        logger.info(f"Model Summary")
        logger.info(f"{'='*50}")
        logger.info(f"  Diseases: {len(meta['diseases'])}")
        logger.info(f"  Symptoms: {len(meta['symptoms'])}")
        logger.info(f"  Training samples: {meta['n_samples']}")
        logger.info(f"  Laplace alpha: {meta['laplace_alpha']}")
        logger.info(f"  Saved to: {args.output}")
        logger.info(f"{'='*50}")

    if args.evaluate:
        evaluate_model(args.output, data_path)

    logger.info("✅ Training complete!")
    logger.info(f"Run the app with: streamlit run ui/app.py")


if __name__ == "__main__":
    main()
