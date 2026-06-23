"""
Step 5: Uncertainty & Novelty Detector
Analyzes inference results for:
  - Shannon entropy of disease distribution
  - Max probability (confidence of top prediction)
  - Open-set novelty score (KL from uniform)
  - Flags: uncertain, novel, known_dangerous
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.reasoner.bn_inference import InferenceResult

logger = logging.getLogger(__name__)

# Diseases considered high-risk (triggers conservative escalation)
HIGH_RISK_DISEASES = {
    "Heart Attack", "Stroke", "Meningitis", "Encephalitis",
    "Appendicitis", "Pulmonary Embolism", "Aortic Dissection",
    "Ectopic Pregnancy", "Cholera", "Kidney Failure",
    "Sepsis", "Anaphylaxis"
}


@dataclass
class UncertaintyReport:
    """Uncertainty and novelty assessment of inference result."""
    entropy: float              # Shannon entropy H(P(Disease))
    max_prob: float             # P(top disease)
    top_disease: str
    novelty_score: float        # KL(P || Uniform) — high = concentrated/known
    is_uncertain: bool          # H > entropy_threshold
    is_novel: bool              # max_prob < novelty_threshold
    high_risk_detected: bool    # Known dangerous disease in top predictions
    high_risk_diseases: List[str]   # Which dangerous diseases were flagged
    confidence_label: str       # "High", "Moderate", "Low", "Very Low"
    warning_messages: List[str]


class UncertaintyDetector:
    """
    Computes uncertainty and novelty metrics from BN inference results.

    Parameters
    ----------
    entropy_threshold   : H above this → mark as uncertain (bits)
    novelty_threshold   : max_prob below this → mark as novel/unknown
    high_risk_prob_threshold : if any high-risk disease > this → alert
    top_k_for_risk      : how many top diseases to scan for high-risk
    """

    def __init__(
        self,
        entropy_threshold: float = 3.0,
        novelty_threshold: float = 0.10,
        high_risk_prob_threshold: float = 0.08,
        top_k_for_risk: int = 5
    ):
        self.entropy_threshold = entropy_threshold
        self.novelty_threshold = novelty_threshold
        self.high_risk_prob_threshold = high_risk_prob_threshold
        self.top_k_for_risk = top_k_for_risk

    def analyze(self, result: InferenceResult) -> UncertaintyReport:
        """
        Compute uncertainty metrics.

        Args:
            result: InferenceResult from BN inference.

        Returns:
            UncertaintyReport
        """
        probs = result.disease_probs
        if not probs:
            return self._empty_report()

        # Shannon entropy
        entropy = self._shannon_entropy(probs)

        # Max prob and top disease
        top_disease, max_prob = result.top_diseases[0] if result.top_diseases else ("Unknown", 0.0)

        # KL divergence from uniform (novelty = negative KL, high KL = known disease)
        novelty_score = self._kl_from_uniform(probs)

        # Flags
        is_uncertain = entropy > self.entropy_threshold
        is_novel = max_prob < self.novelty_threshold or top_disease == "Unknown/Other"

        # High-risk scan
        high_risk_found = []
        for disease, prob in result.top_diseases[:self.top_k_for_risk]:
            if disease in HIGH_RISK_DISEASES and prob >= self.high_risk_prob_threshold:
                high_risk_found.append(disease)

        high_risk_detected = len(high_risk_found) > 0

        # Confidence label
        confidence_label = self._confidence_label(max_prob, entropy)

        # Warning messages
        warnings = self._build_warnings(
            is_uncertain, is_novel, high_risk_detected,
            high_risk_found, max_prob, top_disease
        )

        return UncertaintyReport(
            entropy=round(entropy, 4),
            max_prob=round(max_prob, 4),
            top_disease=top_disease,
            novelty_score=round(novelty_score, 4),
            is_uncertain=is_uncertain,
            is_novel=is_novel,
            high_risk_detected=high_risk_detected,
            high_risk_diseases=high_risk_found,
            confidence_label=confidence_label,
            warning_messages=warnings
        )

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def _shannon_entropy(probs: Dict[str, float]) -> float:
        """H(P) = -Σ p_i log2(p_i)  (in bits)"""
        entropy = 0.0
        for p in probs.values():
            if p > 1e-12:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _kl_from_uniform(probs: Dict[str, float]) -> float:
        """
        KL(P || Uniform) = Σ p_i log(p_i / (1/N))
        High value = concentrated distribution = model is confident.
        Near 0 = spread like uniform = uncertain / novel.
        """
        n = len(probs)
        if n == 0:
            return 0.0
        uniform = 1.0 / n
        kl = 0.0
        for p in probs.values():
            if p > 1e-12:
                kl += p * math.log(p / uniform)
        return kl

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _confidence_label(max_prob: float, entropy: float) -> str:
        if max_prob >= 0.50:
            return "High"
        elif max_prob >= 0.25:
            return "Moderate"
        elif max_prob >= 0.10:
            return "Low"
        else:
            return "Very Low"

    @staticmethod
    def _build_warnings(
        is_uncertain: bool, is_novel: bool,
        high_risk_detected: bool, high_risk_diseases: List[str],
        max_prob: float, top_disease: str
    ) -> List[str]:
        warnings = []
        if high_risk_detected:
            diseases_str = ", ".join(high_risk_diseases)
            warnings.append(
                f"⚠️  High-risk condition detected: {diseases_str}. "
                f"Seek medical attention immediately."
            )
        if is_uncertain:
            warnings.append(
                "⚠️  Symptom pattern is ambiguous — multiple conditions are possible. "
                "A medical professional should evaluate."
            )
        if is_novel:
            warnings.append(
                "⚠️  Symptoms don't match known disease profiles well. "
                "This could be an unusual presentation or a rare condition."
            )
        if top_disease == "Unknown/Other":
            warnings.append(
                "ℹ️  Best match is 'Unknown/Other' — symptoms may represent a rare or "
                "atypical condition requiring specialist evaluation."
            )
        return warnings

    @staticmethod
    def _empty_report() -> UncertaintyReport:
        return UncertaintyReport(
            entropy=0.0, max_prob=0.0, top_disease="Unknown",
            novelty_score=0.0, is_uncertain=True, is_novel=True,
            high_risk_detected=False, high_risk_diseases=[],
            confidence_label="Very Low",
            warning_messages=["No inference results available."]
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_detector_instance = None


def get_detector() -> UncertaintyDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = UncertaintyDetector()
    return _detector_instance


def analyze_uncertainty(result: InferenceResult) -> UncertaintyReport:
    return get_detector().analyze(result)


if __name__ == "__main__":
    from src.reasoner.bn_inference import InferenceResult

    # Mock result for testing
    mock_probs = {
        "Influenza": 0.45, "Common Cold": 0.20, "COVID-19": 0.15,
        "Pneumonia": 0.08, "Heart Attack": 0.05, "Unknown/Other": 0.07
    }
    mock_result = InferenceResult(
        disease_probs=mock_probs,
        top_diseases=sorted(mock_probs.items(), key=lambda x: -x[1]),
        evidence_used={"fever": 1, "cough": 1},
        soft_evidence_used={},
        model_diseases=list(mock_probs.keys())
    )

    report = analyze_uncertainty(mock_result)
    print(f"Entropy: {report.entropy:.3f} bits")
    print(f"Max prob: {report.max_prob:.3f} ({report.top_disease})")
    print(f"Confidence: {report.confidence_label}")
    print(f"Uncertain: {report.is_uncertain}, Novel: {report.is_novel}")
    print(f"High-risk: {report.high_risk_detected} → {report.high_risk_diseases}")
    print(f"Warnings: {report.warning_messages}")
