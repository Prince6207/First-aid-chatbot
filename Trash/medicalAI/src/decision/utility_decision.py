"""
Step 6: Utility Decision Module
Computes Expected Utility EU(action) = Σ P(disease) * U(action, disease)
and recommends the action with highest expected utility.

Actions: {rest, otc, see_gp, visit_er}
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.reasoner.bn_inference import InferenceResult
from src.reasoner.uncertainty import UncertaintyReport

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"

ACTIONS = ["rest", "otc", "see_gp", "visit_er"]

ACTION_LABELS = {
    "rest": "🏠 Rest at Home",
    "otc": "💊 Over-the-Counter Medication",
    "see_gp": "🩺 See a Doctor (GP)",
    "visit_er": "🚨 Go to Emergency Room"
}

ACTION_DESCRIPTIONS = {
    "rest": "Monitor your symptoms at home. Stay hydrated, rest, and watch for worsening.",
    "otc": "Mild symptoms that can be managed with over-the-counter medication and home care.",
    "see_gp": "Schedule an appointment with your doctor within 24–48 hours for proper evaluation.",
    "visit_er": "Seek emergency medical care immediately. Do not delay."
}


@dataclass
class ActionUtility:
    action: str
    label: str
    expected_utility: float
    description: str


@dataclass
class DecisionResult:
    """Output of the utility decision module."""
    recommended_action: str
    recommended_label: str
    recommended_description: str
    expected_utilities: Dict[str, float]        # {action: EU}
    action_utilities: List[ActionUtility]       # sorted by EU desc
    escalation_triggered: bool                  # ER escalated due to high-risk
    escalation_reason: str
    top_contributing_diseases: List[Tuple[str, float]]  # [(disease, weight)]


class UtilityDecisionModule:
    """
    Expected Utility Decision Maker.

    Loads utility table U(action, disease) from JSON and computes
    EU(a) = Σ_d P(d) * U(a, d) for each action a.

    Conservative safety weighting:
      - Missing a high-risk condition has high negative utility
      - 'Unknown/Other' utility is conservative (escalate to GP/ER)
      - If UncertaintyReport flags high-risk → force visit_er
    """

    def __init__(self, utility_path: Path = DATA_DIR / "utility_table.json"):
        with open(utility_path) as f:
            data = json.load(f)
        self.utilities: Dict[str, Dict[str, float]] = data["utilities"]
        self.thresholds: Dict[str, float] = data.get("thresholds", {})
        self._default_utility: Dict[str, float] = {
            "rest": -20, "otc": -10, "see_gp": 60, "visit_er": 50
        }
        logger.info(f"Utility table loaded: {len(self.utilities)} diseases × {len(ACTIONS)} actions")

    def decide(
        self,
        inference_result: InferenceResult,
        uncertainty_report: UncertaintyReport,
    ) -> DecisionResult:
        """
        Compute expected utility and recommend action.

        Args:
            inference_result   : BN inference output
            uncertainty_report : Uncertainty analysis output

        Returns:
            DecisionResult
        """
        probs = inference_result.disease_probs
        eu: Dict[str, float] = {action: 0.0 for action in ACTIONS}

        # Compute EU(a) = Σ P(d) * U(a, d)
        for disease, p in probs.items():
            utils = self.utilities.get(disease, self._default_utility)
            for action in ACTIONS:
                eu[action] += p * utils.get(action, self._default_utility[action])

        # Normalize EU to [0, 100] for display
        eu_raw = dict(eu)

        # Safety escalation: if uncertainty report flags high-risk → override to ER
        escalation_triggered = False
        escalation_reason = ""

        er_prob_threshold = self.thresholds.get("er_probability", 0.15)
        for disease in uncertainty_report.high_risk_diseases:
            if probs.get(disease, 0) > er_prob_threshold:
                escalation_triggered = True
                escalation_reason = (
                    f"High-risk condition '{disease}' probability "
                    f"({probs[disease]:.1%}) exceeds safety threshold."
                )
                break

        # Also escalate if Unknown/Other is top disease and system is uncertain
        if (uncertainty_report.is_novel and
                uncertainty_report.is_uncertain and
                not escalation_triggered):
            escalation_triggered = True
            escalation_reason = "Uncertain + novel symptom pattern — conservative escalation."

        # Choose best action
        if escalation_triggered:
            recommended = "visit_er"
        else:
            recommended = max(eu, key=eu.get)

        # Sort actions by EU for display
        action_utilities = sorted(
            [
                ActionUtility(
                    action=a,
                    label=ACTION_LABELS[a],
                    expected_utility=round(eu[a], 2),
                    description=ACTION_DESCRIPTIONS[a]
                )
                for a in ACTIONS
            ],
            key=lambda x: -x.expected_utility
        )

        # Top contributing diseases to the decision
        top_contributors = sorted(probs.items(), key=lambda x: -x[1])[:5]

        return DecisionResult(
            recommended_action=recommended,
            recommended_label=ACTION_LABELS[recommended],
            recommended_description=ACTION_DESCRIPTIONS[recommended],
            expected_utilities={a: round(eu[a], 2) for a in ACTIONS},
            action_utilities=action_utilities,
            escalation_triggered=escalation_triggered,
            escalation_reason=escalation_reason,
            top_contributing_diseases=top_contributors
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_decision_module: Optional[UtilityDecisionModule] = None


def get_decision_module() -> UtilityDecisionModule:
    global _decision_module
    if _decision_module is None:
        _decision_module = UtilityDecisionModule()
    return _decision_module


def make_decision(
    inference_result: InferenceResult,
    uncertainty_report: UncertaintyReport
) -> DecisionResult:
    return get_decision_module().decide(inference_result, uncertainty_report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.reasoner.bn_inference import InferenceResult
    from src.reasoner.uncertainty import analyze_uncertainty

    mock_probs = {
        "Influenza": 0.40, "Common Cold": 0.25, "COVID-19": 0.15,
        "Pneumonia": 0.10, "Heart Attack": 0.05, "Unknown/Other": 0.05
    }
    mock_result = InferenceResult(
        disease_probs=mock_probs,
        top_diseases=sorted(mock_probs.items(), key=lambda x: -x[1]),
        evidence_used={"fever": 1, "cough": 1, "body_ache": 1},
        soft_evidence_used={},
        model_diseases=list(mock_probs.keys())
    )
    unc_report = analyze_uncertainty(mock_result)
    decision = make_decision(mock_result, unc_report)

    print(f"\nRecommended action: {decision.recommended_label}")
    print(f"Escalation: {decision.escalation_triggered} — {decision.escalation_reason}")
    print("\nExpected utilities:")
    for au in decision.action_utilities:
        print(f"  {au.label}: EU={au.expected_utility:.2f}")
