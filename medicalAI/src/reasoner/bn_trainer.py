"""
Bayesian Network Trainer
Trains a Naive Bayes-style BN (Disease → Symptoms) on the Mendeley dataset.

Structure:
  Disease → Symptom_1, Symptom_2, ..., Symptom_N

Supports:
  - Wide format CSV:  disease, sym1, sym2, ... (binary columns)
  - Long format CSV:  disease, symptom, present (1/0)
  - Auto-detection of format

Usage:
  python scripts/train_bn.py --data data/mendeley_dataset.csv --output models/bn_model.pkl
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


class BayesianNetworkTrainer:
    """
    Trains and persists a pgmpy BayesianNetwork for medical diagnosis.

    The structure is a Naive Bayes:
      Disease (parent) → each Symptom_i (child)

    CPTs are estimated via BayesianEstimator (Laplace smoothing by default).
    """

    def __init__(
        self,
        canonical_path: Path = DATA_DIR / "canonical_symptoms.json",
        laplace_alpha: float = 1.0,
        unknown_disease_name: str = "Unknown/Other",
        unknown_prior_weight: float = 0.05,
    ):
        self.laplace_alpha = laplace_alpha
        self.unknown_disease_name = unknown_disease_name
        self.unknown_prior_weight = unknown_prior_weight

        with open(canonical_path) as f:
            vocab = json.load(f)
        self.canonical_symptoms = vocab["symptoms"]
        self.canonical_diseases = vocab["diseases"]

    # ── Data Loading ─────────────────────────────────────────────────────────

    def load_dataset(self, csv_path: Path) -> pd.DataFrame:
        """
        Load and normalize the dataset into wide binary format.

        Expected wide format:
            Disease | symptom_1 | symptom_2 | ...
            Flu     |     1     |     0     | ...

        Expected long format:
            disease | symptom       | present
            Flu     | fever         | 1
            Flu     | headache      | 0
        """
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Detect format
        if 'disease' in df.columns and 'symptom' in df.columns:
            logger.info("Detected long format — pivoting to wide format.")
            df = self._long_to_wide(df)
        elif 'disease' not in df.columns:
            # First column is likely the disease
            df = df.rename(columns={df.columns[0]: 'disease'})

        logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)-1} symptom columns")
        return df

    def _long_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert long format to wide binary format."""
        present_col = 'present' if 'present' in df.columns else df.columns[2]
        df[present_col] = pd.to_numeric(df[present_col], errors='coerce').fillna(0).astype(int)
        wide = df.pivot_table(
            index='disease', columns='symptom',
            values=present_col, aggfunc='max', fill_value=0
        ).reset_index()
        return wide

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map dataset columns to canonical symptom names and add Unknown/Other disease.
        """
        from src.normalizer.symptom_normalizer import SymptomNormalizer
        normalizer = SymptomNormalizer()

        # Map disease names
        disease_col = df['disease'].str.strip()
        df['disease'] = disease_col

        # Map symptom column names to canonical
        symptom_cols = [c for c in df.columns if c != 'disease']
        col_map = {}
        for col in symptom_cols:
            text_col = col.replace('_', ' ')
            match = normalizer._exact_match(text_col) or \
                    normalizer._token_overlap_match(text_col)
            if match:
                col_map[col] = match[0]
            else:
                col_map[col] = col  # keep as-is if no match

        df = df.rename(columns=col_map)

        # Keep only canonical symptom columns that exist
        valid_symptom_cols = [c for c in self.canonical_symptoms if c in df.columns]
        df = df[['disease'] + valid_symptom_cols]

        # Binarize
        for col in valid_symptom_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(0, 1).astype(int)

        # Add Unknown/Other disease (synthetic row with 0s)
        unknown_row = {col: 0 for col in valid_symptom_cols}
        unknown_row['disease'] = self.unknown_disease_name
        unknown_df = pd.DataFrame([unknown_row] * max(1, int(len(df) * self.unknown_prior_weight)))
        df = pd.concat([df, unknown_df], ignore_index=True)

        logger.info(f"Preprocessed: {df['disease'].nunique()} diseases, "
                    f"{len(valid_symptom_cols)} symptoms")
        return df

    # ── BN Construction ───────────────────────────────────────────────────────

    def build_structure(self, symptom_cols: List[str]) -> "pgmpy.models.BayesianNetwork":
        """Build Naive Bayes BN structure: Disease → each Symptom."""
        from pgmpy.models import BayesianNetwork
        edges = [("disease", sym) for sym in symptom_cols]
        model = BayesianNetwork(edges)
        logger.info(f"BN structure: 1 disease node + {len(symptom_cols)} symptom nodes")
        return model

    def fit(self, df: pd.DataFrame) -> "pgmpy.models.BayesianNetwork":
        """
        Fit CPTs using BayesianEstimator (Laplace smoothing).

        Args:
            df: Wide format DataFrame with 'disease' + binary symptom columns.

        Returns:
            Fitted BayesianNetwork
        """
        from pgmpy.models import BayesianNetwork
        from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator

        symptom_cols = [c for c in df.columns if c != 'disease']
        model = self.build_structure(symptom_cols)

        logger.info(f"Fitting BN on {len(df)} samples with Laplace α={self.laplace_alpha}...")
        model.fit(
            df,
            estimator=BayesianEstimator,
            prior_type="BDeu",
            equivalent_sample_size=self.laplace_alpha * 10
        )
        assert model.check_model(), "BN model check failed!"
        logger.info("BN fitted successfully.")
        return model

    # ── Training Pipeline ─────────────────────────────────────────────────────

    def train(
        self,
        csv_path: Path,
        output_path: Path = MODELS_DIR / "bn_model.pkl"
    ) -> "pgmpy.models.BayesianNetwork":
        """Full training pipeline: load → preprocess → fit → save."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_raw = self.load_dataset(csv_path)
        df = self.preprocess(df_raw)

        model = self.fit(df)

        # Save model + metadata
        metadata = {
            "diseases": list(df['disease'].unique()),
            "symptoms": [c for c in df.columns if c != 'disease'],
            "n_samples": len(df),
            "laplace_alpha": self.laplace_alpha,
        }
        with open(output_path, 'wb') as f:
            pickle.dump({"model": model, "metadata": metadata}, f)

        # Also save metadata as JSON
        meta_path = output_path.with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {output_path}")
        logger.info(f"Metadata saved to {meta_path}")
        return model

    # ── Synthetic Demo Dataset ─────────────────────────────────────────────────

    @staticmethod
    def create_demo_dataset(output_path: Path = DATA_DIR / "demo_dataset.csv") -> Path:
        """
        Creates a small synthetic dataset for demo/testing when
        the real Mendeley dataset is not available.
        """
        np.random.seed(42)

        diseases_symptoms = {
            "Common Cold":     ["runny_nose", "sneezing", "sore_throat", "mild_fever", "cough"],
            "Influenza":       ["fever", "body_ache", "fatigue", "headache", "chills", "cough"],
            "COVID-19":        ["fever", "dry_cough", "fatigue", "loss_of_smell", "loss_of_taste", "shortness_of_breath"],
            "Pneumonia":       ["fever", "productive_cough", "shortness_of_breath", "chest_pain", "fatigue"],
            "Gastroenteritis": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "fever"],
            "UTI":             ["painful_urination", "frequent_urination", "lower_back_pain", "mild_fever"],
            "Migraine":        ["severe_headache", "nausea", "blurred_vision", "dizziness"],
            "Heart Attack":    ["chest_pain", "shortness_of_breath", "sweating", "nausea", "palpitations"],
            "Dengue":          ["high_fever", "body_ache", "rash", "joint_pain", "headache"],
            "Malaria":         ["high_fever", "chills", "sweating", "headache", "nausea", "vomiting"],
            "Appendicitis":    ["abdominal_pain", "fever", "nausea", "vomiting", "loss_of_appetite"],
            "Hypertension":    ["headache", "dizziness", "blurred_vision", "palpitations"],
            "Asthma":          ["wheezing", "shortness_of_breath", "chest_tightness", "dry_cough"],
            "Meningitis":      ["severe_headache", "neck_stiffness", "fever", "confusion", "chills"],
        }

        # Get all unique symptoms
        all_syms = sorted(set(s for syms in diseases_symptoms.values() for s in syms))

        rows = []
        for disease, main_syms in diseases_symptoms.items():
            for _ in range(50):  # 50 synthetic patients per disease
                row = {"disease": disease}
                for sym in all_syms:
                    if sym in main_syms:
                        row[sym] = int(np.random.random() > 0.15)  # 85% present
                    else:
                        row[sym] = int(np.random.random() < 0.05)  # 5% noise
                rows.append(row)

        df = pd.DataFrame(rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Demo dataset saved: {len(df)} rows to {output_path}")
        return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    trainer = BayesianNetworkTrainer()

    # Create and train on demo dataset
    demo_path = trainer.create_demo_dataset()
    model = trainer.train(demo_path, MODELS_DIR / "bn_model.pkl")
    print(f"\nTraining complete. Diseases in model: {len(model.get_cpds())}")
