"""from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

MODEL_NAME = "d4data/biomedical-ner-all"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
from transformers import pipeline

ner = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple",
    framework="pt"
)

text = "Patient has fever, cough and headache."

entities = ner(text)

symptoms = [e["word"] for e in entities]

print("Extracted:", symptoms)"""

# ////----

"""
Step 1: Symptom Extractor
Uses BioBERT (or a biomedical NER model) to extract symptom entities
from free-text clinical descriptions with confidence scores.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
@dataclass
class ExtractedSymptom:
    text: str
    confidence: float
    start: int
    end: int
    entity_type: str = "SYMPTOM"
    source: str = "model"  # model | rule


@dataclass
class ExtractionResult:
    """Full result from symptom extraction."""

    original_text: str
    symptoms: List[ExtractedSymptom] = field(default_factory=list)
    model_name: str = ""
    fallback_used: bool = False


class BioBERTExtractor:
    """
    Symptom extractor using a biomedical NER model.

    Primary model: d4data/biomedical-ner-all  (BIO NER for clinical text)
    Fallback:      Rule-based extraction using symptom vocabulary.

    The model annotates tokens with entity types; we filter for
    symptom-related entities (SIGN_SYMPTOM, DISEASE_DISORDER, etc.)
    """

    # Entity types in biomedical NER that map to symptoms
    SYMPTOM_ENTITY_TYPES = {
        "SIGN_SYMPTOM",
        "SYMPTOM",
        "DISEASE_DISORDER",
        "BIOLOGICAL_ATTRIBUTE",
        "CLINICAL_EVENT",
    }

    # Fallback: common symptom keywords for regex-based extraction
    SYMPTOM_KEYWORDS = [
        r"\b(fever|febrile|pyrexia|temperature)\b",
        r"\b(cough(?:ing)?)\b",
        r"\b(headache|head\s+pain|cephalalgia)\b",
        r"\b(fatigue|tired(?:ness)?|exhaustion|lethargy)\b",
        r"\b(nausea|nauseous|queasy)\b",
        r"\b(vomit(?:ing)?|emesis)\b",
        r"\b(diarrhea|diarrhoea|loose\s+(?:stool|motion))\b",
        r"\b(pain|ache|aching|sore|hurt(?:ing)?)\b",
        r"\b(chest\s+(?:pain|tightness|pressure))\b",
        r"\b(short(?:ness)?\s+of\s+breath|breathless(?:ness)?|dyspnea|dyspnoea)\b",
        r"\b(rash|itching|itch(?:y)?|pruritus)\b",
        r"\b(dizziness|dizzy|vertigo|lightheaded(?:ness)?)\b",
        r"\b(chills?|shiver(?:ing)?|rigors?)\b",
        r"\b(sweating|perspir(?:ation|ing)|night\s+sweats?)\b",
        r"\b(weakness|weak|malaise|unwell)\b",
        r"\b(swelling|swollen|edema|oedema)\b",
        r"\b(jaundice|yellow(?:ing)?)\b",
        r"\b(confusion|confused|disoriented?)\b",
        r"\b(seizures?|convulsions?|fits?)\b",
        r"\b(palpitations?|racing\s+heart)\b",
        r"\b(loss\s+of\s+(?:smell|taste|appetite|consciousness))\b",
        r"\b(wheezing|wheeze)\b",
        r"\b(blood\s+in\s+(?:stool|urine|sputum))\b",
        r"\b(numbness|tingling|paresthesia)\b",
        r"\b(tremors?|shaking)\b",
    ]

    def __init__(
        self,
        model_name: str = "d4data/biomedical-ner-all",
        device: str = "cpu",
        use_fallback_if_error: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_fallback_if_error = use_fallback_if_error
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the HuggingFace NER pipeline."""
        try:
            from transformers import (
                pipeline,
                AutoTokenizer,
                AutoModelForTokenClassification,
            )

            logger.info(f"Loading BioBERT NER model: {self.model_name}")
            self.pipeline = pipeline(
                task="ner",
                model=self.model_name,
                tokenizer=self.model_name,
                aggregation_strategy="simple",
                framework="pt",
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load {self.model_name}: {e}")
            logger.warning("Will use rule-based fallback extractor.")
            self.pipeline = None

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract symptoms using BOTH model and rule fallback.

        Model output + rule extraction are merged so symptoms are never missed.
        """

        model_result = None
        rule_result = None

        # Run model extraction
        if self.pipeline is not None:
            try:
                model_result = self._extract_with_model(text)
            except Exception as e:
                logger.warning(f"Model extraction failed: {e}")

        # Always run rule fallback
        rule_result = self._extract_with_rules(text)

        # Merge both results
        merged_symptoms = []

        if model_result:
            merged_symptoms.extend(model_result.symptoms)

        if rule_result:
            merged_symptoms.extend(rule_result.symptoms)

        merged_symptoms = _deduplicate_spans(merged_symptoms)

        return ExtractionResult(
            original_text=text,
            symptoms=merged_symptoms,
            model_name=self.model_name,
            fallback_used=True,
        )

    def _extract_with_model(self, text: str) -> ExtractionResult:
        """Use the NER pipeline to extract symptom spans."""
        result = ExtractionResult(original_text=text, model_name=self.model_name)
        try:
            entities = self.pipeline(text)
            for ent in entities:
                label = ent.get("entity_group", ent.get("entity", ""))
                # Keep only symptom-relevant entity types
                if any(
                    sym_type in label.upper()
                    for sym_type in ["SIGN", "SYMPTOM", "DISEASE", "BIOLOGICAL"]
                ):
                    symptom = ExtractedSymptom(
                        text=ent["word"].strip(),
                        confidence=float(ent["score"]),
                        start=ent["start"],
                        end=ent["end"],
                        entity_type=label,
                        source="model",
                    )
                    result.symptoms.append(symptom)
        except Exception as e:
            logger.error(f"Model inference failed: {e}. Falling back to rules.")
            return self._extract_with_rules(text)
        return result

    def _extract_with_rules(self, text: str) -> ExtractionResult:
        """
        Regex-based fallback symptom extractor.
        Assigns a fixed confidence of 0.75 (heuristic).
        """
        result = ExtractionResult(
            original_text=text, model_name="rule_based_fallback", fallback_used=True
        )
        text_lower = text.lower()
        seen_spans = set()

        for pattern in self.SYMPTOM_KEYWORDS:
            for m in re.finditer(pattern, text_lower, re.IGNORECASE):
                span = (m.start(), m.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    result.symptoms.append(
                        ExtractedSymptom(
                            text=text[m.start() : m.end()],
                            confidence=0.75,
                            start=m.start(),
                            end=m.end(),
                            entity_type="SYMPTOM",
                            source="rule",
                        )
                    )

        # Deduplicate overlapping spans — keep highest confidence
        result.symptoms = _deduplicate_spans(result.symptoms)
        return result


def _deduplicate_spans(symptoms: List[ExtractedSymptom]) -> List[ExtractedSymptom]:
    """Remove overlapping spans, keeping the one with highest confidence."""
    if not symptoms:
        return symptoms
    symptoms = sorted(symptoms, key=lambda s: (s.start, -s.confidence))
    deduped = []
    last_end = -1
    for sym in symptoms:
        if sym.start >= last_end:
            deduped.append(sym)
            last_end = sym.end
    return deduped


# ── Convenience wrapper ──────────────────────────────────────────────────────

_extractor_instance: Optional[BioBERTExtractor] = None


def get_extractor(model_name: str = "d4data/biomedical-ner-all") -> BioBERTExtractor:
    """Get or create a singleton extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = BioBERTExtractor(model_name=model_name)
    return _extractor_instance


def extract_symptoms(
    text: str, model_name: str = "d4data/biomedical-ner-all"
) -> ExtractionResult:
    """
    High-level function: extract symptoms from text.

    Args:
        text: Free-form symptom description.
        model_name: HuggingFace model name for NER.

    Returns:
        ExtractionResult
    """
    extractor = get_extractor(model_name)
    return extractor.extract(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_texts = [
        "I have been experiencing high fever, severe headache, and body aches for 3 days.",
        "Patient presents with chest pain, shortness of breath, and dizziness.",
        "I feel nauseous, have diarrhea, and lost my appetite.",
        "Sudden onset of confusion, neck stiffness, and sensitivity to light.",
    ]
    for txt in test_texts:
        result = extract_symptoms(txt)
        print(f"\nText: {txt}")
        print(f"Model: {result.model_name} | Fallback: {result.fallback_used}")
        for sym in result.symptoms:
            print(f"  [{sym.entity_type}] '{sym.text}' confidence={sym.confidence:.2f}")
