"""
Step 2: Symptom Normalizer
Maps raw extracted symptom phrases → canonical symptom vocabulary (BN nodes).

Strategy:
  1. Exact / lowercase match against alias dictionary
  2. Token overlap matching (fast)
  3. SBERT cosine similarity (semantic fallback for unknowns)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.extractor.symptom_extractor import ExtractedSymptom, ExtractionResult

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@dataclass
class NormalizedSymptom:
    """A symptom mapped to a canonical BN node."""
    raw_text: str               # Original extracted phrase
    canonical: str              # Canonical node name
    confidence: float           # Extractor confidence × match confidence
    match_method: str           # "exact", "alias", "token_overlap", "semantic", "unmatched"
    semantic_score: float = 0.0


@dataclass
class NormalizationResult:
    """Full normalization output for one patient input."""
    original_text: str
    normalized: List[NormalizedSymptom] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)


class SymptomNormalizer:
    """
    Maps extracted symptom phrases to canonical vocabulary.

    Parameters
    ----------
    canonical_path : Path to canonical_symptoms.json
    aliases_path   : Path to symptom_aliases.json
    sbert_model    : Sentence-BERT model name (used only as fallback)
    semantic_threshold : Minimum cosine similarity to accept semantic match
    """

    def __init__(
        self,
        canonical_path: Path = DATA_DIR / "canonical_symptoms.json",
        aliases_path: Path = DATA_DIR / "symptom_aliases.json",
        sbert_model: str = "all-MiniLM-L6-v2",
        semantic_threshold: float = 0.55,
    ):
        self.semantic_threshold = semantic_threshold
        self.sbert_model_name = sbert_model
        self._sbert = None
        self._canonical_embeddings: Optional[np.ndarray] = None

        # Load vocabulary
        with open(canonical_path) as f:
            vocab = json.load(f)
        self.canonical_symptoms: List[str] = vocab["symptoms"]

        with open(aliases_path) as f:
            raw_aliases: Dict[str, List[str]] = json.load(f)

        # Build flat alias → canonical lookup
        self._alias_map: Dict[str, str] = {}
        for canonical, aliases in raw_aliases.items():
            for alias in aliases:
                self._alias_map[alias.lower().strip()] = canonical
            self._alias_map[canonical.lower().strip()] = canonical

        logger.info(f"Normalizer loaded: {len(self.canonical_symptoms)} canonical symptoms, "
                    f"{len(self._alias_map)} alias entries")

    # ── SBERT lazy loader ────────────────────────────────────────────────────

    def _get_sbert(self):
        if self._sbert is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SBERT model: {self.sbert_model_name}")
                self._sbert = SentenceTransformer(self.sbert_model_name)
                # Pre-compute embeddings for canonical symptoms
                canonical_texts = [s.replace("_", " ") for s in self.canonical_symptoms]
                self._canonical_embeddings = self._sbert.encode(
                    canonical_texts, normalize_embeddings=True, show_progress_bar=False
                )
                logger.info("SBERT loaded and canonical embeddings cached.")
            except Exception as e:
                logger.warning(f"SBERT unavailable: {e}. Semantic matching disabled.")
        return self._sbert

    # ── Matching methods ─────────────────────────────────────────────────────

    def _normalize_text(self, text: str) -> str:
        """Clean text for matching."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _exact_match(self, text: str) -> Optional[Tuple[str, float]]:
        """Direct lookup in alias map."""
        norm = self._normalize_text(text)
        if norm in self._alias_map:
            return self._alias_map[norm], 1.0
        # Also try with underscores
        norm_under = norm.replace(' ', '_')
        if norm_under in self._alias_map:
            return self._alias_map[norm_under], 1.0
        return None

    def _token_overlap_match(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Find best matching alias by token-level Jaccard similarity.
        Fast alternative before calling SBERT.
        """
        tokens = set(self._normalize_text(text).split())
        if len(tokens) == 0:
            return None

        best_score = 0.0
        best_canonical = None

        for alias, canonical in self._alias_map.items():
            alias_tokens = set(alias.split())
            intersection = len(tokens & alias_tokens)
            union = len(tokens | alias_tokens)
            score = intersection / union if union > 0 else 0.0
            if score > best_score:
                best_score = score
                best_canonical = canonical

        if best_score >= 0.4:
            return best_canonical, best_score
        return None

    def _semantic_match(self, text: str) -> Optional[Tuple[str, float]]:
        """Use SBERT cosine similarity to find the closest canonical symptom."""
        sbert = self._get_sbert()
        if sbert is None or self._canonical_embeddings is None:
            return None

        query_emb = sbert.encode(
            [text.replace("_", " ")],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        scores = self._canonical_embeddings @ query_emb[0]   # cosine similarity
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.semantic_threshold:
            return self.canonical_symptoms[best_idx], best_score
        return None

    # ── Main normalize method ─────────────────────────────────────────────────

    def normalize_symptom(self, extracted: ExtractedSymptom) -> NormalizedSymptom:
        """Map a single extracted symptom to a canonical node."""
        raw = extracted.text

        # 1. Exact / alias match
        match = self._exact_match(raw)
        if match:
            canonical, match_conf = match
            return NormalizedSymptom(
                raw_text=raw, canonical=canonical,
                confidence=extracted.confidence * match_conf,
                match_method="alias", semantic_score=match_conf
            )

        # 2. Token overlap
        match = self._token_overlap_match(raw)
        if match:
            canonical, match_conf = match
            return NormalizedSymptom(
                raw_text=raw, canonical=canonical,
                confidence=extracted.confidence * match_conf,
                match_method="token_overlap", semantic_score=match_conf
            )

        # 3. Semantic (SBERT) fallback
        match = self._semantic_match(raw)
        if match:
            canonical, match_conf = match
            return NormalizedSymptom(
                raw_text=raw, canonical=canonical,
                confidence=extracted.confidence * match_conf,
                match_method="semantic", semantic_score=match_conf
            )

        # 4. Unmatched
        return NormalizedSymptom(
            raw_text=raw, canonical="",
            confidence=0.0, match_method="unmatched"
        )

    def normalize(self, extraction: ExtractionResult) -> NormalizationResult:
        """
        Normalize all extracted symptoms.

        Args:
            extraction: ExtractionResult from the extractor.

        Returns:
            NormalizationResult with matched canonical symptoms.
        """
        result = NormalizationResult(original_text=extraction.original_text)

        for sym in extraction.symptoms:
            normalized = self.normalize_symptom(sym)
            if normalized.match_method != "unmatched" and normalized.canonical:
                # Deduplicate: keep highest confidence for same canonical
                existing = next(
                    (n for n in result.normalized if n.canonical == normalized.canonical),
                    None
                )
                if existing is None:
                    result.normalized.append(normalized)
                elif normalized.confidence > existing.confidence:
                    result.normalized.remove(existing)
                    result.normalized.append(normalized)
            else:
                result.unmatched.append(sym.text)

        logger.debug(f"Normalized {len(result.normalized)} symptoms, "
                     f"{len(result.unmatched)} unmatched.")
        return result


# ── Convenience functions ────────────────────────────────────────────────────

_normalizer_instance: Optional[SymptomNormalizer] = None


def get_normalizer() -> SymptomNormalizer:
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = SymptomNormalizer()
    return _normalizer_instance


def normalize_symptoms(extraction: ExtractionResult) -> NormalizationResult:
    """Normalize an ExtractionResult."""
    return get_normalizer().normalize(extraction)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.extractor.symptom_extractor import extract_symptoms

    test_text = "I have been experiencing high fever, severe headache, body aches, and I am feeling very nauseous."
    extraction = extract_symptoms(test_text)
    result = normalize_symptoms(extraction)

    print(f"\nInput: {test_text}")
    print(f"\nNormalized symptoms ({len(result.normalized)}):")
    for sym in result.normalized:
        print(f"  '{sym.raw_text}' → {sym.canonical} "
              f"[{sym.match_method}] conf={sym.confidence:.2f}")
    if result.unmatched:
        print(f"\nUnmatched: {result.unmatched}")
