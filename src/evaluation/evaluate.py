"""
Evaluation script for the Cred Support Agent.
Runs the evaluation dataset against the guardrails and query classifier
to measure routing accuracy and guardrail effectiveness.
"""

import json
from pathlib import Path
from typing import Any

from src.guardrails.input_validator import validate_input
from src.agents.crew import classify_query
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def load_eval_dataset() -> list[dict[str, Any]]:
    """Load evaluation cases from the dataset file."""
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("evaluation_cases", [])


def evaluate_guardrails(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Evaluate guardrail accuracy on the dataset.
    Tests whether malicious/OOD queries are correctly rejected
    and valid queries are correctly accepted.
    """
    results: list[dict[str, Any]] = []
    correct = 0
    total = 0

    for case in cases:
        total += 1
        query = case["query"]
        customer_id = case.get("customer_id")
        expected_agent = case["expected_agent"]

        validation = validate_input(query=query, customer_id=customer_id)

        # Guardrail cases should be rejected
        if expected_agent == "rejected":
            is_correct = not validation.is_valid
        else:
            is_correct = validation.is_valid

        if is_correct:
            correct += 1

        results.append({
            "id": case["id"],
            "category": case["category"],
            "query": query[:60],
            "expected": "rejected" if expected_agent == "rejected" else "accepted",
            "actual": "rejected" if not validation.is_valid else "accepted",
            "correct": is_correct,
            "risk_flags": validation.risk_flags,
        })

    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        "total_cases": total,
        "correct": correct,
        "accuracy_pct": round(accuracy, 1),
        "results": results,
    }


def evaluate_routing(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Evaluate query routing accuracy.
    Tests whether queries are routed to the correct agent type.
    """
    # Only evaluate non-guardrail cases
    routing_cases = [
        c for c in cases if c["expected_agent"] != "rejected"
    ]

    results: list[dict[str, Any]] = []
    correct = 0
    total = len(routing_cases)

    for case in routing_cases:
        query = case["query"]
        customer_id = case.get("customer_id")
        expected_agent = case["expected_agent"]

        actual_agent = classify_query(query, customer_id)
        is_correct = actual_agent == expected_agent

        if is_correct:
            correct += 1

        results.append({
            "id": case["id"],
            "query": query[:60],
            "expected_agent": expected_agent,
            "actual_agent": actual_agent,
            "correct": is_correct,
        })

    accuracy = (correct / total * 100) if total > 0 else 0

    return {
        "total_cases": total,
        "correct": correct,
        "accuracy_pct": round(accuracy, 1),
        "results": results,
    }


def run_evaluation() -> dict[str, Any]:
    """Run the full evaluation suite and print a report."""
    cases = load_eval_dataset()

    print("=" * 60)
    print("  Cred Support Agent — Evaluation Report")
    print("=" * 60)

    # Guardrail evaluation
    guardrail_report = evaluate_guardrails(cases)
    print(f"\n📋 Guardrail Evaluation: {guardrail_report['accuracy_pct']}% accuracy")
    print(f"   ({guardrail_report['correct']}/{guardrail_report['total_cases']} correct)")

    for r in guardrail_report["results"]:
        status = "✅" if r["correct"] else "❌"
        print(f"   {status} [{r['id']}] {r['query']}")
        if not r["correct"]:
            print(f"       Expected: {r['expected']}, Got: {r['actual']}")

    # Routing evaluation
    routing_report = evaluate_routing(cases)
    print(f"\n🔀 Routing Evaluation: {routing_report['accuracy_pct']}% accuracy")
    print(f"   ({routing_report['correct']}/{routing_report['total_cases']} correct)")

    for r in routing_report["results"]:
        status = "✅" if r["correct"] else "❌"
        print(f"   {status} [{r['id']}] {r['query']}")
        if not r["correct"]:
            print(
                f"       Expected: {r['expected_agent']}, "
                f"Got: {r['actual_agent']}"
            )

    print("\n" + "=" * 60)

    return {
        "guardrails": guardrail_report,
        "routing": routing_report,
    }


if __name__ == "__main__":
    setup_logging()
    run_evaluation()
