"""
MedDecide Pipeline Orchestrator
Ties all 7 steps together into a single callable pipeline.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


@dataclass
class PipelineResult:
    """Complete output from the full pipeline."""
    original_text: str
    # Step outputs
    extraction: object
    normalization: object
    evidence_bundle: object
    inference_result: object
    uncertainty_report: object
    decision_result: object
    explanation: object
    # Quick-access fields
    recommended_action: str
    recommended_label: str
    top_disease: str
    top_disease_prob: float
    confidence_label: str
    summary: str
    warnings: list
    inference_log: list
    success: bool = True
    error: str = ""


class MedDecidePipeline:
    """
    Full pipeline: text → symptoms → BN → decision → explanation.

    Parameters
    ----------
    model_path   : Path to trained BN model pickle
    use_llm      : Whether to use Claude for explanation phrasing
    use_soft_path: Whether to use soft evidence (vs hard 0/1)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_llm: bool = True,
        use_soft_path: bool = True,
        extractor_model: str = "d4data/biomedical-ner-all",
    ):
        self.model_path = model_path or MODELS_DIR / "bn_model.pkl"
        self.use_llm = use_llm
        self.use_soft_path = use_soft_path
        self.extractor_model = extractor_model

        # Load canonical symptoms list
        with open(DATA_DIR / "canonical_symptoms.json") as f:
            self._all_symptoms = json.load(f)["symptoms"]

        # Lazy-loaded components
        self._extractor = None
        self._normalizer = None
        self._reasoner = None
        self._uncertainty_detector = None
        self._decision_module = None
        self._explainer = None

    # ── Lazy component getters ─────────────────────────────────────────────────

    @property
    def extractor(self):
        if self._extractor is None:
            from src.extractor.symptom_extractor import BioBERTExtractor
            self._extractor = BioBERTExtractor(self.extractor_model)
        return self._extractor

    @property
    def normalizer(self):
        if self._normalizer is None:
            from src.normalizer.symptom_normalizer import SymptomNormalizer
            self._normalizer = SymptomNormalizer()
        return self._normalizer

    @property
    def reasoner(self):
        if self._reasoner is None:
            from src.reasoner.bn_inference import BayesianReasoner
            self._reasoner = BayesianReasoner(self.model_path)
        return self._reasoner

    @property
    def uncertainty_detector(self):
        if self._uncertainty_detector is None:
            from src.reasoner.uncertainty import UncertaintyDetector
            self._uncertainty_detector = UncertaintyDetector()
        return self._uncertainty_detector

    @property
    def decision_module(self):
        if self._decision_module is None:
            from src.decision.utility_decision import UtilityDecisionModule
            self._decision_module = UtilityDecisionModule()
        return self._decision_module

    @property
    def explainer(self):
        if self._explainer is None:
            from src.explainer.explanation_generator import ExplanationGenerator
            self._explainer = ExplanationGenerator(use_llm=self.use_llm)
        return self._explainer

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run(self, text: str) -> PipelineResult:
        """
        Run the full pipeline on patient input text.

        Args:
            text: Free-form symptom description.

        Returns:
            PipelineResult
        """
        try:
            logger.info(f"Pipeline running for input: {text[:80]}...")

            # Step 1: Extract symptoms
            extraction = self.extractor.extract(text)
            logger.debug(f"Step 1: {len(extraction.symptoms)} symptoms extracted")

            # Step 2: Normalize
            normalization = self.normalizer.normalize(extraction)
            logger.debug(f"Step 2: {len(normalization.normalized)} normalized")

            # Step 3: Create soft evidence
            from src.normalizer.soft_evidence import SoftEvidenceCreator
            ev_creator = SoftEvidenceCreator(use_soft_path=self.use_soft_path)
            evidence_bundle = ev_creator.create_evidence(normalization, self._all_symptoms)
            logger.debug(f"Step 3: Evidence bundle created")

            # Step 4: BN Inference
            inference_result = self.reasoner.query_from_evidence_bundle(evidence_bundle)
            logger.debug(f"Step 4: Inference complete — top: {inference_result.top_diseases[:2]}")

            # Step 5: Uncertainty analysis
            uncertainty_report = self.uncertainty_detector.analyze(inference_result)
            logger.debug(f"Step 5: H={uncertainty_report.entropy:.2f}, "
                         f"max_p={uncertainty_report.max_prob:.2f}")

            # Step 6: Decision
            decision_result = self.decision_module.decide(inference_result, uncertainty_report)
            logger.debug(f"Step 6: Recommended → {decision_result.recommended_action}")

            # Step 7: Explanation
            explanation = self.explainer.generate(
                text, normalization, inference_result,
                uncertainty_report, decision_result
            )
            logger.info(f"Pipeline complete. Action: {decision_result.recommended_label}")

            top_disease, top_prob = (
                inference_result.top_diseases[0] if inference_result.top_diseases
                else ("Unknown", 0.0)
            )

            return PipelineResult(
                original_text=text,
                extraction=extraction,
                normalization=normalization,
                evidence_bundle=evidence_bundle,
                inference_result=inference_result,
                uncertainty_report=uncertainty_report,
                decision_result=decision_result,
                explanation=explanation,
                recommended_action=decision_result.recommended_action,
                recommended_label=decision_result.recommended_label,
                top_disease=top_disease,
                top_disease_prob=top_prob,
                confidence_label=uncertainty_report.confidence_label,
                summary=explanation.summary,
                warnings=uncertainty_report.warning_messages,
                inference_log=explanation.inference_log,
                success=True,
            )

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            return PipelineResult(
                original_text=text,
                extraction=None, normalization=None, evidence_bundle=None,
                inference_result=None, uncertainty_report=None,
                decision_result=None, explanation=None,
                recommended_action="see_gp",
                recommended_label="🩺 See a Doctor (GP)",
                top_disease="Unknown", top_disease_prob=0.0,
                confidence_label="Very Low",
                summary="An error occurred during analysis. Please consult a doctor.",
                warnings=["System error — please seek medical advice directly."],
                inference_log=[f"[ERROR] {str(e)}"],
                success=False, error=str(e)
            )


# ── Module-level singleton ─────────────────────────────────────────────────────

_pipeline_instance: Optional[MedDecidePipeline] = None


def get_pipeline(**kwargs) -> MedDecidePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MedDecidePipeline(**kwargs)
    return _pipeline_instance


def run_pipeline(text: str, **kwargs) -> PipelineResult:
    """Convenience function to run the full pipeline."""
    return get_pipeline(**kwargs).run(text)
