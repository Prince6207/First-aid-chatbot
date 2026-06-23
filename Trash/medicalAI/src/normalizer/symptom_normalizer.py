"""
Step 2: Symptom Normalizer (Minimal Version)

Maps extracted symptom phrases → canonical symptom vocabulary.

Strategy:
1. Exact / alias match
2. Token overlap match
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.extractor.symptom_extractor import ExtractionResult

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class NormalizedSymptom:
    raw_text: str
    canonical: str
    confidence: float
    match_method: str


@dataclass
class NormalizationResult:
    original_text: str
    normalized: List[NormalizedSymptom] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)


class SymptomNormalizer:
    """
    Minimal symptom normalizer.
    Uses alias lookup + token overlap.
    """

    def __init__(
        self,
        canonical_path: Path = DATA_DIR / "canonical_symptoms.json",
        aliases_path: Path = DATA_DIR / "symptom_aliases.json",
    ):

        # Load canonical symptoms
        with open(canonical_path) as f:
            vocab = json.load(f)

        self.canonical_symptoms = vocab["symptoms"]

        # Load aliases
        with open(aliases_path) as f:
            raw_aliases: Dict[str, List[str]] = json.load(f)

        self.alias_map: Dict[str, str] = {}

        for canonical, aliases in raw_aliases.items():
            self.alias_map[canonical.lower()] = canonical

            for alias in aliases:
                self.alias_map[alias.lower()] = canonical

        logger.info(
            f"Normalizer loaded: {len(self.canonical_symptoms)} canonical symptoms"
        )

    # ---------- helpers ----------

    def _clean(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _exact_match(self, text: str) -> Optional[str]:

        cleaned = self._clean(text)

        if cleaned in self.alias_map:
            return self.alias_map[cleaned]

        cleaned = cleaned.replace(" ", "_")

        if cleaned in self.alias_map:
            return self.alias_map[cleaned]

        return None

    def _token_match(self, text: str) -> Optional[str]:

        tokens = set(self._clean(text).split())

        best_score = 0
        best_symptom = None

        for alias, canonical in self.alias_map.items():

            alias_tokens = set(alias.split())

            intersection = len(tokens & alias_tokens)
            union = len(tokens | alias_tokens)

            if union == 0:
                continue

            score = intersection / union

            if score > best_score:
                best_score = score
                best_symptom = canonical

        if best_score >= 0.4:
            return best_symptom

        return None

    # ---------- main normalize ----------

    def normalize(self, extraction: ExtractionResult) -> NormalizationResult:

        result = NormalizationResult(original_text=extraction.original_text)

        seen = {}

        for symptom in extraction.symptoms:

            raw = symptom.text

            canonical = self._exact_match(raw)

            method = "alias"

            if canonical is None:
                canonical = self._token_match(raw)
                method = "token_overlap"

            if canonical is None:
                result.unmatched.append(raw)
                continue

            confidence = symptom.confidence

            existing = seen.get(canonical)

            if existing is None or confidence > existing.confidence:
                seen[canonical] = NormalizedSymptom(
                    raw_text=raw,
                    canonical=canonical,
                    confidence=confidence,
                    match_method=method,
                )

        result.normalized = list(seen.values())

        return result


# -------- convenience wrapper --------

_normalizer_instance: Optional[SymptomNormalizer] = None


def get_normalizer() -> SymptomNormalizer:

    global _normalizer_instance

    if _normalizer_instance is None:
        _normalizer_instance = SymptomNormalizer()

    return _normalizer_instance


def normalize_symptoms(extraction: ExtractionResult) -> NormalizationResult:

    normalizer = get_normalizer()

    return normalizer.normalize(extraction)


# ---------- test ----------

if __name__ == "__main__":

    import logging
    from src.extractor.symptom_extractor import extract_symptoms

    logging.basicConfig(level=logging.INFO)

    text = "I have high fever, severe headache and chest pain."

    extraction = extract_symptoms(text)

    result = normalize_symptoms(extraction)

    print("\nNormalized symptoms:\n")

    for s in result.normalized:
        print(
            f"{s.raw_text} → {s.canonical} "
            f"[{s.match_method}] conf={s.confidence:.2f}"
        )

    if result.unmatched:
        print("\nUnmatched:", result.unmatched)