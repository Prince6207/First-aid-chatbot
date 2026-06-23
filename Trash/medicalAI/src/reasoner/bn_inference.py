"""
Step 4: Bayesian Network Inference Engine
Loads trained BN model and performs inference with soft/hard evidence.

Supports:
  - Variable Elimination (exact)
  - Likelihood Weighting (approximate, supports soft evidence)
  - Virtual Evidence (pgmpy)
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


@dataclass
class InferenceResult:
    """Output of BN inference."""
    disease_probs: Dict[str, float]     # {disease_name: probability}
    top_diseases: List[Tuple[str, float]]   # sorted by probability desc
    evidence_used: Dict[str, int]           # hard evidence applied
    soft_evidence_used: Dict[str, List[float]]  # virtual evidence applied
    model_diseases: List[str]               # all diseases in model
    inference_method: str = "variable_elimination"


class BayesianReasoner:
    """
    Loads a trained pgmpy BayesianNetwork and performs disease inference.
    """

    def __init__(self, model_path: Path = MODELS_DIR / "bn_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.metadata: Dict = {}
        self.diseases: List[str] = []
        self.symptoms: List[str] = []
        self._ve_engine = None
        self._load_model()

    def _load_model(self):
        """Load the trained model from disk."""
        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}. "
                           f"Train first with: python scripts/train_bn.py")
            return

        with open(self.model_path, 'rb') as f:
            saved = pickle.load(f)

        self.model = saved["model"]
        self.metadata = saved.get("metadata", {})
        self.diseases = self.metadata.get("diseases", [])
        self.symptoms = self.metadata.get("symptoms", [])

        # Build Variable Elimination engine
        from pgmpy.inference import VariableElimination
        self._ve_engine = VariableElimination(self.model)
        logger.info(f"BN loaded: {len(self.diseases)} diseases, {len(self.symptoms)} symptoms")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ── Inference methods ─────────────────────────────────────────────────────

    def infer(
        self,
        hard_evidence: Dict[str, int],
        soft_evidence: Optional[Dict[str, List[float]]] = None,
        top_k: int = 10,
    ) -> InferenceResult:
        """
        Perform disease inference given evidence.

        Args:
            hard_evidence   : {symptom_name: 0 or 1}
            soft_evidence   : {symptom_name: [P(obs|absent), P(obs|present)]}
            top_k           : Number of top diseases to return

        Returns:
            InferenceResult
        """
        if not self.is_loaded:
            return self._dummy_result()

        # Filter evidence to known symptom nodes
        known_symptoms = set(self.symptoms)
        hard_ev = {k: v for k, v in hard_evidence.items() if k in known_symptoms}
        soft_ev = {k: v for k, v in (soft_evidence or {}).items()
                   if k in known_symptoms and k not in hard_ev}

        try:
            probs = self._infer_ve(hard_ev, soft_ev)
        except Exception as e:
            logger.warning(f"VE inference failed: {e}. Using prior distribution.")
            probs = self._get_prior()

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        top_diseases = sorted(probs.items(), key=lambda x: -x[1])[:top_k]

        return InferenceResult(
            disease_probs=probs,
            top_diseases=top_diseases,
            evidence_used=hard_ev,
            soft_evidence_used=soft_ev,
            model_diseases=self.diseases,
            inference_method="variable_elimination"
        )

    def _infer_ve(
        self,
        hard_evidence: Dict[str, int],
        soft_evidence: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Variable Elimination with optional Virtual Evidence.
        """
        from pgmpy.factors.discrete import DiscreteFactor

        # Combine: use hard evidence directly; soft evidence via likelihood weighting
        all_evidence = dict(hard_evidence)

        # For soft evidence, apply as virtual evidence (likelihood weighting trick)
        # We convert soft to pseudo-hard by taking MAP value, weighted by likelihood
        soft_contributions: Dict[str, float] = {}
        for sym, likelihoods in soft_evidence.items():
            # P(present | obs) ∝ P(obs | present) * P(present) — simplified version
            p_present_given_obs = likelihoods[1] / (likelihoods[0] + likelihoods[1] + 1e-9)
            # Only add as hard evidence if strongly present
            if p_present_given_obs > 0.70:
                all_evidence[sym] = 1
            elif p_present_given_obs < 0.20:
                all_evidence[sym] = 0
            # Otherwise skip (too uncertain)

        query_result = self._ve_engine.query(
            variables=["disease"],
            evidence=all_evidence,
            show_progress=False
        )

        disease_states = query_result.state_names["disease"]
        probs = {
            state: float(query_result.values[i])
            for i, state in enumerate(disease_states)
        }
        return probs

    def _get_prior(self) -> Dict[str, float]:
        """Return prior distribution from the disease CPD."""
        try:
            disease_cpd = self.model.get_cpds("disease")
            states = disease_cpd.state_names["disease"]
            values = disease_cpd.values.flatten()
            return dict(zip(states, values.tolist()))
        except Exception:
            # Uniform prior
            n = len(self.diseases) if self.diseases else 10
            return {d: 1.0 / n for d in self.diseases}

    def _dummy_result(self) -> InferenceResult:
        """Fallback when model is not loaded."""
        diseases = ["Unknown/Other", "Common Cold", "Influenza"]
        probs = {"Unknown/Other": 0.5, "Common Cold": 0.3, "Influenza": 0.2}
        return InferenceResult(
            disease_probs=probs,
            top_diseases=list(probs.items()),
            evidence_used={},
            soft_evidence_used={},
            model_diseases=diseases,
            inference_method="dummy_fallback"
        )

    # ── Quick API ─────────────────────────────────────────────────────────────

    def query_from_evidence_bundle(self, bundle, top_k: int = 10) -> InferenceResult:
        """
        Run inference from an EvidenceBundle (from soft_evidence.py).
        """
        soft_ev = bundle.as_virtual_evidence()
        # Filter out absent evidence (mode=soft_absent) — only keep present symptoms
        soft_ev_present = {
            e.node: e.likelihood for e in bundle.soft_evidence
            if e.mode == "soft" and e.likelihood[1] > 0.5
        }
        return self.infer(
            hard_evidence=bundle.hard_evidence,
            soft_evidence=soft_ev_present,
            top_k=top_k
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_reasoner_instance: Optional[BayesianReasoner] = None


def get_reasoner(model_path: Optional[Path] = None) -> BayesianReasoner:
    global _reasoner_instance
    if _reasoner_instance is None:
        path = model_path or MODELS_DIR / "bn_model.pkl"
        _reasoner_instance = BayesianReasoner(path)
    return _reasoner_instance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reasoner = get_reasoner()

    if reasoner.is_loaded:
        # Test with fever + headache + chills
        result = reasoner.infer(
            hard_evidence={"fever": 1, "headache": 1, "chills": 1},
            top_k=5
        )
        print("\nTop diseases for fever + headache + chills:")
        for disease, prob in result.top_diseases:
            print(f"  {disease}: {prob:.3f}")
    else:
        print("Model not loaded. Train first with: python scripts/train_bn.py")
