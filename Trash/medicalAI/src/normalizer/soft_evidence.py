"""
Step 3: Soft Evidence Creator
Converts normalized symptom confidences → P(symptom=present) likelihood arrays
suitable for pgmpy's Virtual Evidence / likelihood weighting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.normalizer.symptom_normalizer import NormalizationResult, NormalizedSymptom


@dataclass
class SoftEvidence:
    """
    Soft evidence for a single BN node.

    pgmpy Virtual Evidence format:
      likelihood = [P(E | node=state_0), P(E | node=state_1), ...]
    For a binary symptom node (absent=0, present=1):
      likelihood = [1 - confidence, confidence]
    """
    node: str                          # Canonical symptom name
    likelihood: List[float]            # [P(obs|absent), P(obs|present)]
    confidence: float                  # Original extractor confidence
    mode: str = "soft"                 # "soft" or "hard"


@dataclass
class EvidenceBundle:
    """All evidence extracted from one patient input."""
    soft_evidence: List[SoftEvidence] = field(default_factory=list)
    hard_evidence: Dict[str, int] = field(default_factory=dict)  # node → 0/1

    def as_virtual_evidence(self) -> Dict[str, List[float]]:
        """Return dict suitable for pgmpy virtual evidence."""
        return {e.node: e.likelihood for e in self.soft_evidence}

    def top_symptoms(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k symptoms by P(present) confidence."""
        items = [(e.node, e.likelihood[1]) for e in self.soft_evidence]
        return sorted(items, key=lambda x: -x[1])[:top_k]


class SoftEvidenceCreator:
    """
    Converts normalized symptom confidences into probabilistic evidence.

    Parameters
    ----------
    hard_threshold : float
        If confidence ≥ this value, create hard evidence (binary).
        Otherwise create soft (likelihood) evidence.
    absent_weight : float
        When a symptom is NOT mentioned, we optionally add soft
        absent evidence. Set to None to skip absent symptoms.
    """

    def __init__(
        self,
        hard_threshold: float = 0.95,
        absent_weight: float = 0.1,
        use_soft_path: bool = True,
    ):
        self.hard_threshold = hard_threshold
        self.absent_weight = absent_weight
        self.use_soft_path = use_soft_path

    def create_evidence(
        self,
        norm_result: NormalizationResult,
        all_symptoms: List[str]
    ) -> EvidenceBundle:
        """
        Build evidence bundle from normalized symptoms.

        Args:
            norm_result    : NormalizationResult from normalizer.
            all_symptoms   : Full list of canonical symptom names (BN nodes).

        Returns:
            EvidenceBundle
        """
        bundle = EvidenceBundle()
        mentioned_canonicals = {n.canonical for n in norm_result.normalized}

        for sym in norm_result.normalized:
            p_present = self._confidence_to_probability(sym.confidence)

            if not self.use_soft_path or sym.confidence >= self.hard_threshold:
                # Hard evidence
                bundle.hard_evidence[sym.canonical] = 1
            else:
                # Soft evidence (virtual evidence)
                p_absent = 1.0 - p_present
                bundle.soft_evidence.append(SoftEvidence(
                    node=sym.canonical,
                    likelihood=[p_absent, p_present],
                    confidence=sym.confidence,
                    mode="soft"
                ))

        # Optionally add soft absent evidence for non-mentioned symptoms
        if self.absent_weight is not None:
            for sym_name in all_symptoms:
                if sym_name not in mentioned_canonicals:
                    bundle.soft_evidence.append(SoftEvidence(
                        node=sym_name,
                        likelihood=[1.0, self.absent_weight],
                        confidence=1.0 - self.absent_weight,
                        mode="soft_absent"
                    ))

        return bundle

    @staticmethod
    def _confidence_to_probability(confidence: float) -> float:
        """
        Convert extractor confidence to P(symptom=present).

        Applies a mild calibration:  p = 0.5 + 0.5 * confidence
        This prevents probabilities near 0 or 1 from overwhelming the network.
        """
        # Clip to [0, 1] just in case
        confidence = max(0.0, min(1.0, confidence))
        # Calibrated probability
        p = 0.5 + 0.5 * confidence
        return round(p, 4)


def create_evidence(
    norm_result: NormalizationResult,
    all_symptoms: List[str],
    hard_threshold: float = 0.95,
    use_soft_path: bool = True,
) -> EvidenceBundle:
    """Convenience wrapper."""
    creator = SoftEvidenceCreator(
        hard_threshold=hard_threshold,
        use_soft_path=use_soft_path
    )
    return creator.create_evidence(norm_result, all_symptoms)


if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.extractor.symptom_extractor import extract_symptoms
    from src.normalizer.symptom_normalizer import normalize_symptoms

    # Load canonical symptoms
    data_dir = Path(__file__).parent.parent.parent / "data"
    with open(data_dir / "canonical_symptoms.json") as f:
        all_symptoms = json.load(f)["symptoms"]

    text = "I have fever, headache, and I'm feeling very nauseous."
    extraction = extract_symptoms(text)
    normalization = normalize_symptoms(extraction)
    bundle = create_evidence(normalization, all_symptoms)

    print(f"\nSoft evidence ({len(bundle.soft_evidence)} entries):")
    for ev in bundle.soft_evidence[:10]:
        if ev.mode == "soft":
            print(f"  {ev.node}: P(absent)={ev.likelihood[0]:.3f}, P(present)={ev.likelihood[1]:.3f}")

    print(f"\nHard evidence: {bundle.hard_evidence}")
    print(f"\nTop symptoms: {bundle.top_symptoms(5)}")
