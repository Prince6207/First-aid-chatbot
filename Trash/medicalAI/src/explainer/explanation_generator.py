"""
Step 7: Explanation Generator
Produces human-readable explanations combining:
  - Template-based structured summary
  - Optional LLM (Claude/Anthropic) for conversational phrasing
  - Red flags / monitoring advice
  - Inference step log for educational transparency
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.reasoner.bn_inference import InferenceResult
from src.reasoner.uncertainty import UncertaintyReport
from src.decision.utility_decision import DecisionResult
from src.normalizer.symptom_normalizer import NormalizationResult

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Full explanation output."""
    # Structured fields
    summary: str                        # 2-3 sentence human summary
    top_diseases_text: str              # Formatted top disease list
    decision_rationale: str             # Why this action was recommended
    monitoring_advice: str              # What to watch for
    red_flags: str                      # Escalation triggers
    inference_log: List[str]            # Step-by-step log for teachers
    warnings: List[str]                 # From uncertainty report
    # Optional: LLM-generated
    llm_response: Optional[str] = None
    used_llm: bool = False


# ── Red flags by condition ────────────────────────────────────────────────────

RED_FLAGS_BY_ACTION = {
    "rest": [
        "Fever above 39.5°C (103°F) or fever lasting more than 3 days",
        "Difficulty breathing or chest pain",
        "Confusion, stiff neck, or rash with fever",
        "Symptoms significantly worsening after 48 hours",
        "Signs of dehydration (no urination, dry mouth, extreme dizziness)"
    ],
    "otc": [
        "No improvement after 48–72 hours of OTC treatment",
        "Worsening symptoms or new symptoms appearing",
        "High fever (>39.5°C), chest pain, or difficulty breathing",
        "Blood in vomit, stool, or urine",
        "Severe abdominal pain or sudden severe headache"
    ],
    "see_gp": [
        "Symptoms worsen before your appointment — go to ER",
        "Sudden severe chest pain, breathlessness, or one-sided weakness",
        "High fever unresponsive to medication",
        "Loss of consciousness or confusion",
        "Severe allergic reaction (hives, swelling, difficulty breathing)"
    ],
    "visit_er": [
        "You're already being directed to Emergency — do not wait",
        "Call emergency services (112/911) if you cannot travel safely",
        "Do not eat or drink in case surgery is needed",
        "Bring a list of your current medications"
    ]
}

MONITORING_BY_ACTION = {
    "rest": (
        "Monitor your temperature every 4–6 hours. "
        "Keep a symptom diary. Stay well hydrated. "
        "Return to normal activities gradually as you improve."
    ),
    "otc": (
        "Follow dosage instructions carefully. "
        "Track which symptoms improve with medication. "
        "Monitor for any side effects or new symptoms. "
        "Re-assess after 48 hours."
    ),
    "see_gp": (
        "Note down all your symptoms, when they started, and their severity. "
        "List any medications you're currently taking. "
        "Track temperature and any pattern changes before your appointment."
    ),
    "visit_er": (
        "Do not delay. If driving, have someone else drive you. "
        "Note the time symptoms started. "
        "Bring ID and any regular medication."
    )
}


class ExplanationGenerator:
    """
    Generates structured and conversational explanations.

    Uses templates by default.
    If ANTHROPIC_API_KEY is set, uses Claude for conversational phrasing.
    """

    def __init__(self, use_llm: bool = True, model: str = "claude-sonnet-4-20250514"):
        self.use_llm = use_llm and bool(os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        if self.use_llm:
            logger.info("LLM explanation enabled via Anthropic API.")
        else:
            logger.info("Using template-based explanations (no API key set).")

    def generate(
        self,
        original_text: str,
        norm_result: NormalizationResult,
        inference_result: InferenceResult,
        uncertainty_report: UncertaintyReport,
        decision_result: DecisionResult,
    ) -> Explanation:
        """
        Generate a full explanation.

        Returns:
            Explanation object
        """
        # Build inference log
        log = self._build_inference_log(
            original_text, norm_result, inference_result,
            uncertainty_report, decision_result
        )

        # Build structured explanation
        top_diseases_text = self._format_top_diseases(inference_result)
        decision_rationale = self._format_decision_rationale(decision_result, uncertainty_report)
        monitoring = MONITORING_BY_ACTION.get(decision_result.recommended_action, "")
        red_flags = self._format_red_flags(decision_result.recommended_action)
        summary = self._build_summary(
            norm_result, inference_result, uncertainty_report, decision_result
        )

        explanation = Explanation(
            summary=summary,
            top_diseases_text=top_diseases_text,
            decision_rationale=decision_rationale,
            monitoring_advice=monitoring,
            red_flags=red_flags,
            inference_log=log,
            warnings=uncertainty_report.warning_messages,
        )

        # Optionally enhance with LLM
        if self.use_llm:
            llm_text = self._llm_enhance(
                original_text, norm_result, inference_result,
                uncertainty_report, decision_result
            )
            if llm_text:
                explanation.llm_response = llm_text
                explanation.used_llm = True

        return explanation

    # ── Template builders ─────────────────────────────────────────────────────

    def _build_summary(
        self, norm_result, inference_result, uncertainty_report, decision_result
    ) -> str:
        syms = [s.canonical.replace("_", " ") for s in norm_result.normalized[:5]]
        syms_text = ", ".join(syms) if syms else "the described symptoms"
        top_disease, top_prob = inference_result.top_diseases[0] if inference_result.top_diseases else ("Unknown", 0)

        confidence = uncertainty_report.confidence_label
        action = decision_result.recommended_label

        summary = (
            f"Based on the reported symptoms ({syms_text}), "
            f"the most likely condition is **{top_disease}** "
            f"(probability: {top_prob:.1%}, confidence: {confidence}). "
        )
        if uncertainty_report.is_uncertain:
            summary += (
                f"However, the symptom pattern is consistent with multiple conditions, "
                f"so a definitive diagnosis is uncertain. "
            )
        summary += f"The recommended action is: **{action}**."
        return summary

    def _format_top_diseases(self, result: InferenceResult, top_k: int = 5) -> str:
        lines = []
        for i, (disease, prob) in enumerate(result.top_diseases[:top_k], 1):
            bar = "█" * int(prob * 20)
            lines.append(f"{i}. **{disease}** — {prob:.1%}  {bar}")
        return "\n".join(lines)

    def _format_decision_rationale(
        self, decision: DecisionResult, uncertainty: UncertaintyReport
    ) -> str:
        parts = []
        # Sort EU values
        sorted_eu = sorted(decision.expected_utilities.items(), key=lambda x: -x[1])

        from src.decision.utility_decision import ACTION_LABELS
        parts.append("**Expected Utility Analysis:**")
        for action, eu in sorted_eu:
            label = ACTION_LABELS[action]
            marker = " ← **Recommended**" if action == decision.recommended_action else ""
            parts.append(f"  • {label}: EU = {eu:.1f}{marker}")

        if decision.escalation_triggered:
            parts.append(f"\n**⚠️ Safety Escalation:** {decision.escalation_reason}")

        return "\n".join(parts)

    def _format_red_flags(self, action: str) -> str:
        flags = RED_FLAGS_BY_ACTION.get(action, [])
        if not flags:
            return "No specific red flags."
        return "Seek immediate emergency care if:\n" + "\n".join(f"• {f}" for f in flags)

    def _build_inference_log(
        self, text, norm_result, inference_result, uncertainty_report, decision_result
    ) -> List[str]:
        """Build a step-by-step educational inference log."""
        log = []

        # Step 1
        log.append(f"[Step 1 — Symptom Extraction]\n  Input: \"{text[:100]}...\"" if len(text) > 100 else f"[Step 1 — Symptom Extraction]\n  Input: \"{text}\"")

        # Step 2
        syms_info = [(s.canonical, s.match_method, f"{s.confidence:.2f}") for s in norm_result.normalized]
        log.append(f"[Step 2 — Normalization]\n  Mapped: " +
                   ", ".join(f"{c} ({m}, {conf})" for c, m, conf in syms_info[:8]))

        # Step 3
        log.append(f"[Step 3 — Soft Evidence]\n  "
                   f"Created evidence for {len(norm_result.normalized)} symptoms\n  "
                   f"Hard evidence: {inference_result.evidence_used}\n  "
                   f"Soft evidence nodes: {len(inference_result.soft_evidence_used)}")

        # Step 4
        top5 = [(d, f"{p:.3f}") for d, p in inference_result.top_diseases[:5]]
        log.append(f"[Step 4 — Bayesian Inference]\n  Method: {inference_result.inference_method}\n  "
                   f"Top-5: " + " | ".join(f"{d}={p}" for d, p in top5))

        # Step 5
        log.append(f"[Step 5 — Uncertainty Analysis]\n  "
                   f"Shannon Entropy: H = {uncertainty_report.entropy:.3f} bits\n  "
                   f"Max P(Disease): {uncertainty_report.max_prob:.3f} ({uncertainty_report.top_disease})\n  "
                   f"KL from uniform: {uncertainty_report.novelty_score:.3f}\n  "
                   f"Flags: uncertain={uncertainty_report.is_uncertain}, "
                   f"novel={uncertainty_report.is_novel}, "
                   f"high_risk={uncertainty_report.high_risk_detected}")

        # Step 6
        eu_str = " | ".join(f"{a}={v:.1f}" for a, v in decision_result.expected_utilities.items())
        log.append(f"[Step 6 — Utility Decision]\n  EU values: {eu_str}\n  "
                   f"Recommended: {decision_result.recommended_action}\n  "
                   f"Escalation: {decision_result.escalation_triggered}")

        # Step 7
        log.append(f"[Step 7 — Explanation]\n  "
                   f"Generated structured + {'LLM' if self.use_llm else 'template'} explanation.")

        return log

    # ── LLM Enhancement ───────────────────────────────────────────────────────

    def _llm_enhance(
        self, original_text, norm_result, inference_result,
        uncertainty_report, decision_result
    ) -> Optional[str]:
        """Use Claude to generate a conversational explanation."""
        try:
            import anthropic

            syms = [s.canonical.replace("_", " ") for s in norm_result.normalized[:7]]
            top3 = [(d, p) for d, p in inference_result.top_diseases[:3]]
            action = decision_result.recommended_label
            escalated = decision_result.escalation_triggered

            prompt = f"""You are a medical AI assistant. A patient described their symptoms as:
"{original_text}"

Key extracted symptoms: {', '.join(syms)}

Top disease probabilities from our Bayesian Network:
{chr(10).join(f'- {d}: {p:.1%}' for d, p in top3)}

Recommended action: {action}
{'Safety escalation was triggered: ' + decision_result.escalation_reason if escalated else ''}
Uncertainty level: {uncertainty_report.confidence_label}

Write a clear, empathetic, 3-4 sentence response explaining:
1. What the symptoms might indicate
2. Why the recommended action is appropriate
3. One key monitoring point

Be compassionate but not alarmist. Do NOT provide a definitive diagnosis. 
Always recommend consulting a real doctor. Keep it under 150 words."""

            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────

_generator_instance: Optional[ExplanationGenerator] = None


def get_explainer(use_llm: bool = True) -> ExplanationGenerator:
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = ExplanationGenerator(use_llm=use_llm)
    return _generator_instance


def generate_explanation(
    original_text: str,
    norm_result: NormalizationResult,
    inference_result: InferenceResult,
    uncertainty_report: UncertaintyReport,
    decision_result: DecisionResult,
    use_llm: bool = True,
) -> Explanation:
    return get_explainer(use_llm).generate(
        original_text, norm_result, inference_result,
        uncertainty_report, decision_result
    )
