"""
Quick Demo Script
Run the full pipeline without Streamlit for testing and validation.

Usage:
    python scripts/demo.py
    python scripts/demo.py --text "I have fever and headache"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S'
)

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

DEMO_CASES = [
    {
        "name": "Influenza-like",
        "text": "I have had a high fever, severe headache, chills, body aches, and fatigue for 2 days."
    },
    {
        "name": "Cardiac Emergency",
        "text": "I'm having crushing chest pain radiating to my left arm, shortness of breath, and sweating profusely."
    },
    {
        "name": "Gastroenteritis",
        "text": "Since yesterday I've had nausea, vomiting, watery diarrhea, stomach cramps, and I've lost my appetite."
    },
    {
        "name": "Meningitis Red Flags",
        "text": "Sudden terrible headache, stiff neck, high fever, confusion, and I'm very sensitive to light."
    },
    {
        "name": "Common Cold",
        "text": "Runny nose, sneezing, mild sore throat, slight fever. Nothing too serious."
    },
]


def print_result(result, verbose: bool = False):
    """Pretty-print pipeline result to console."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    console.print(f"\n[bold cyan]Input:[/bold cyan] {result.original_text[:100]}")
    console.print(f"[dim]{'─' * 70}[/dim]")

    # Extracted symptoms
    if result.normalization and result.normalization.normalized:
        syms = [s.canonical.replace('_', ' ') for s in result.normalization.normalized]
        console.print(f"[bold]Extracted Symptoms:[/bold] [green]{', '.join(syms)}[/green]")
    else:
        console.print("[dim]No symptoms extracted.[/dim]")

    # Top diseases
    if result.inference_result and result.inference_result.top_diseases:
        console.print("\n[bold]Top Diseases (Bayesian Network):[/bold]")
        table = Table(show_header=True, header_style="bold magenta", border_style="dim")
        table.add_column("Rank", width=5)
        table.add_column("Disease", width=30)
        table.add_column("Probability", width=12)
        table.add_column("Bar")

        for i, (disease, prob) in enumerate(result.inference_result.top_diseases[:6], 1):
            bar = "█" * int(prob * 30)
            color = "green" if i == 1 else "yellow" if i <= 3 else "dim"
            table.add_row(str(i), disease, f"{prob:.3f} ({prob:.1%})", f"[{color}]{bar}[/{color}]")
        console.print(table)

    # Uncertainty
    if result.uncertainty_report:
        unc = result.uncertainty_report
        console.print(f"\n[bold]Uncertainty Analysis:[/bold]")
        console.print(f"  Shannon Entropy:  [yellow]{unc.entropy:.3f} bits[/yellow]")
        console.print(f"  Max P(Disease):   [cyan]{unc.max_prob:.3f}[/cyan] ({unc.top_disease})")
        console.print(f"  Confidence:       [bold]{unc.confidence_label}[/bold]")
        console.print(f"  Is Uncertain:     {unc.is_uncertain}  |  Is Novel: {unc.is_novel}")
        if unc.high_risk_detected:
            console.print(f"  [bold red]⚠️  HIGH RISK: {', '.join(unc.high_risk_diseases)}[/bold red]")

    # Decision
    if result.decision_result:
        dec = result.decision_result
        console.print(f"\n[bold]Recommended Action:[/bold]")
        action_color = {"rest": "green", "otc": "blue", "see_gp": "yellow", "visit_er": "red"}
        color = action_color.get(dec.recommended_action, "white")
        console.print(f"  [{color}]{dec.recommended_label}[/{color}]")
        if dec.escalation_triggered:
            console.print(f"  [red]⚠️  Escalation: {dec.escalation_reason}[/red]")

        console.print("\n  Expected Utilities:")
        for action, eu in sorted(dec.expected_utilities.items(), key=lambda x: -x[1]):
            marker = " ◄ recommended" if action == dec.recommended_action else ""
            console.print(f"    {action:10s}  EU = {eu:6.2f}{marker}")

    # Warnings
    if result.warnings:
        console.print("\n[bold]Warnings:[/bold]")
        for w in result.warnings:
            console.print(f"  [yellow]{w}[/yellow]")

    # Inference log
    if verbose and result.inference_log:
        console.print("\n[bold dim]Inference Log:[/bold dim]")
        for step in result.inference_log:
            lines = step.split('\n')
            for line in lines:
                if line.startswith('[Step') or line.startswith('[Error'):
                    console.print(f"  [bold cyan]{line}[/bold cyan]")
                else:
                    console.print(f"  [dim]{line}[/dim]")

    console.print(f"\n[dim]{'═' * 70}[/dim]\n")


def main():
    parser = argparse.ArgumentParser(description="MedDecide Pipeline Demo")
    parser.add_argument("--text", type=str, default=None,
                        help="Custom symptom text to analyze")
    parser.add_argument("--all", action="store_true",
                        help="Run all demo cases")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full inference log")
    parser.add_argument("--train-first", action="store_true",
                        help="Train the BN model before running")
    args = parser.parse_args()

    # Train if needed
    model_path = MODELS_DIR / "bn_model.pkl"
    if args.train_first or not model_path.exists():
        print("Training Bayesian Network on demo dataset...")
        from src.reasoner.bn_trainer import BayesianNetworkTrainer
        trainer = BayesianNetworkTrainer()
        demo_path = trainer.create_demo_dataset()
        trainer.train(demo_path, model_path)
        print(f"Model saved to {model_path}\n")

    # Load pipeline
    from src.pipeline import MedDecidePipeline
    pipeline = MedDecidePipeline(use_llm=False)

    if args.text:
        result = pipeline.run(args.text)
        print_result(result, verbose=args.verbose)
    elif args.all:
        for case in DEMO_CASES:
            print(f"\n{'═'*70}")
            print(f"DEMO CASE: {case['name']}")
            result = pipeline.run(case['text'])
            print_result(result, verbose=args.verbose)
    else:
        # Run first demo case by default
        print(f"\n{'═'*70}")
        print("Running default demo case. Use --all for all cases or --text for custom input.")
        result = pipeline.run(DEMO_CASES[0]['text'])
        print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    main()
