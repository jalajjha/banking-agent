"""
Input validation guardrails.
Detects prompt injection attempts, out-of-domain queries, and
unauthorized cross-customer data access.
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationResult(BaseModel):
    """Result of input validation checks."""

    is_valid: bool = Field(..., description="Whether the input passed all checks")
    rejection_reason: str | None = Field(
        default=None,
        description="Reason for rejection, if any",
    )
    sanitized_query: str = Field(
        default="", description="The cleaned query after sanitization"
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description="List of risk flags detected",
    )


# ── Prompt Injection Patterns ──

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions?|context|rules?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a\s+)?(different|new|another)", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(instructions?|prompt|system)", re.IGNORECASE),
    re.compile(r"override\s+(your|the|all)\s+(instructions?|rules?|constraints?)", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
    re.compile(r"show\s+me\s+all\s+(customer|user)\s+data", re.IGNORECASE),
    re.compile(r"dump\s+(the\s+)?(database|db|all\s+records)", re.IGNORECASE),
]

# ── Out-of-Domain Patterns ──

OUT_OF_DOMAIN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(write|generate|create)\s+(code|script|program|function)\b", re.IGNORECASE),
    re.compile(r"\b(python|javascript|java|c\+\+|html|css|sql)\s+(code|script)\b", re.IGNORECASE),
    re.compile(r"\b(recipe|cook|food|weather|sports|movie|game|music)\b", re.IGNORECASE),
    re.compile(r"\b(joke|poem|story|essay|homework|assignment)\b", re.IGNORECASE),
    re.compile(r"\b(medical|health|doctor|symptom|diagnosis)\b", re.IGNORECASE),
    re.compile(r"\b(legal\s+advice|lawyer|court\s+case)\b", re.IGNORECASE),
]

# ── Banking Domain Positive Signals ──

BANKING_KEYWORDS: set[str] = {
    "account", "balance", "loan", "emi", "credit", "debit", "card",
    "payment", "transaction", "transfer", "deposit", "withdrawal",
    "interest", "policy", "kyc", "aml", "dispute", "refund",
    "statement", "fee", "charge", "bank", "finance", "cred",
    "closure", "dormant", "mortgage", "insurance", "reward",
}


def check_prompt_injection(query: str) -> tuple[bool, str | None]:
    """
    Check if the query contains prompt injection patterns.

    Returns:
        Tuple of (is_injection, pattern_matched).
    """
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(query)
        if match:
            logger.warning(
                "prompt_injection_detected",
                matched_pattern=match.group(),
                query_preview=query[:100],
            )
            return True, match.group()
    return False, None


def check_out_of_domain(query: str) -> tuple[bool, str | None]:
    """
    Check if the query is outside the banking/FinTech domain.

    Returns:
        Tuple of (is_ood, reason).
    """
    query_lower = query.lower()

    # Check if query has any banking relevance
    has_banking_signal = any(
        keyword in query_lower for keyword in BANKING_KEYWORDS
    )

    # Check for explicit out-of-domain patterns
    for pattern in OUT_OF_DOMAIN_PATTERNS:
        match = pattern.search(query)
        if match and not has_banking_signal:
            logger.warning(
                "out_of_domain_detected",
                matched_pattern=match.group(),
                query_preview=query[:100],
            )
            return True, f"Query appears to be outside the banking domain: '{match.group()}'"

    return False, None


def check_cross_customer_access(
    query: str,
    customer_id: str | None,
) -> tuple[bool, str | None]:
    """
    Check for unauthorized cross-customer data access attempts.

    Detects when a query references customer IDs different from the
    authenticated customer's ID.

    Returns:
        Tuple of (is_violation, reason).
    """
    if customer_id is None:
        return False, None

    # Find all customer ID patterns in the query
    cust_id_pattern = re.compile(r"CUST\d{3}", re.IGNORECASE)
    referenced_ids = set(cust_id_pattern.findall(query.upper()))

    # Remove the authenticated customer's own ID
    referenced_ids.discard(customer_id.upper())

    if referenced_ids:
        logger.warning(
            "cross_customer_access_attempt",
            authenticated_customer=customer_id,
            referenced_customers=list(referenced_ids),
        )
        return True, (
            f"Access denied: You can only access your own account data. "
            f"Attempted to access: {', '.join(referenced_ids)}"
        )

    return False, None


def validate_input(
    query: str,
    customer_id: str | None = None,
) -> ValidationResult:
    """
    Run all input validation checks on a query.

    Args:
        query: The customer's raw query string.
        customer_id: Optional authenticated customer ID.

    Returns:
        ValidationResult indicating whether the query is safe to process.
    """
    risk_flags: list[str] = []

    # Check 1: Empty or too-short query
    if not query or len(query.strip()) < 3:
        return ValidationResult(
            is_valid=False,
            rejection_reason="Query is too short. Please provide a more detailed question.",
            sanitized_query="",
            risk_flags=["empty_query"],
        )

    # Check 2: Excessively long query (potential attack)
    if len(query) > 2000:
        return ValidationResult(
            is_valid=False,
            rejection_reason="Query exceeds the maximum allowed length of 2000 characters.",
            sanitized_query="",
            risk_flags=["excessive_length"],
        )

    # Check 3: Prompt injection
    is_injection, injection_match = check_prompt_injection(query)
    if is_injection:
        return ValidationResult(
            is_valid=False,
            rejection_reason=(
                "Your query was flagged by our security system. "
                "Please rephrase your question to focus on banking services."
            ),
            sanitized_query="",
            risk_flags=["prompt_injection"],
        )

    # Check 4: Out-of-domain
    is_ood, ood_reason = check_out_of_domain(query)
    if is_ood:
        return ValidationResult(
            is_valid=False,
            rejection_reason=(
                "I'm a banking support assistant and can only help with "
                "banking and financial services queries. Please ask me about "
                "accounts, loans, credit cards, policies, or transactions."
            ),
            sanitized_query="",
            risk_flags=["out_of_domain"],
        )

    # Check 5: Cross-customer access
    is_violation, violation_reason = check_cross_customer_access(
        query, customer_id
    )
    if is_violation:
        return ValidationResult(
            is_valid=False,
            rejection_reason=violation_reason,
            sanitized_query="",
            risk_flags=["cross_customer_access"],
        )

    # Sanitize: strip excessive whitespace
    sanitized = " ".join(query.split())

    return ValidationResult(
        is_valid=True,
        rejection_reason=None,
        sanitized_query=sanitized,
        risk_flags=risk_flags,
    )
