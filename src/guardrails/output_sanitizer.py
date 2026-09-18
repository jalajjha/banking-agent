"""
Output sanitizer — PII masking for customer-facing responses.
Masks account numbers, transaction IDs, and other sensitive identifiers
before returning responses to ensure data privacy compliance.
"""

import re

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ── PII Masking Patterns ──

PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # Account numbers: ACC-XXX-YYY → ACC-***-YYY (preserve suffix for context)
    (
        "account_number",
        re.compile(r"\bACC-(\d{3})-(\w{2,4})\b"),
        r"ACC-***-\2",
    ),
    # Transaction IDs: TXN-XXXXXXXX-XXX → TXN-********-XXX
    (
        "transaction_id",
        re.compile(r"\bTXN-(\d{8})-(\d{3})\b"),
        r"TXN-********-\2",
    ),
    # Loan IDs: LOAN-XXX-YY → LOAN-***-YY
    (
        "loan_id",
        re.compile(r"\bLOAN-(\d{3})-(\w{2})\b"),
        r"LOAN-***-\2",
    ),
    # Phone numbers: +91-XXXXX-XXXXX → +91-XXXXX-***XX
    (
        "phone_number",
        re.compile(r"\+91-(\d{5})-(\d{3})(\d{2})\b"),
        r"+91-*****-***\3",
    ),
    # Email addresses: name@domain.com → n***@domain.com
    (
        "email",
        re.compile(r"\b([a-zA-Z0-9._%+-])([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"),
        r"\1***@\3",
    ),
    # Aadhaar-like numbers: 12 digits → XXXX XXXX 1234
    (
        "aadhaar",
        re.compile(r"\b(\d{4})\s?(\d{4})\s?(\d{4})\b"),
        r"XXXX XXXX \3",
    ),
    # PAN Card: ABCDE1234F → ABCDE****F
    (
        "pan_card",
        re.compile(r"\b([A-Z]{5})(\d{4})([A-Z])\b"),
        r"\1****\3",
    ),
]


def mask_pii(text: str, log_masking: bool = True) -> str:
    """
    Mask personally identifiable information (PII) in the given text.

    Applies all PII masking patterns to replace sensitive data with
    masked versions while preserving enough context for readability.

    Args:
        text: The text to sanitize.
        log_masking: Whether to log when masking is applied.

    Returns:
        Sanitized text with PII masked.
    """
    masked_text = text
    masks_applied: list[str] = []

    for pii_type, pattern, replacement in PII_PATTERNS:
        if pattern.search(masked_text):
            masked_text = pattern.sub(replacement, masked_text)
            masks_applied.append(pii_type)

    if masks_applied and log_masking:
        logger.info(
            "pii_masked",
            masks_applied=masks_applied,
            num_masks=len(masks_applied),
        )

    return masked_text


def sanitize_output(
    response: str,
    mask_sensitive_data: bool = True,
) -> str:
    """
    Sanitize an agent's output before returning to the customer.

    Applies PII masking and removes any internal debug information
    that might have leaked into the response.

    Args:
        response: The raw agent response string.
        mask_sensitive_data: Whether to apply PII masking.

    Returns:
        Sanitized response string.
    """
    sanitized = response

    if mask_sensitive_data:
        sanitized = mask_pii(sanitized)

    # Remove any accidentally leaked internal markers
    internal_patterns = [
        re.compile(r"\[DEBUG\].*?\n", re.IGNORECASE),
        re.compile(r"\[INTERNAL\].*?\n", re.IGNORECASE),
        re.compile(r"API_KEY\s*[:=]\s*\S+", re.IGNORECASE),
        re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),
    ]

    for pattern in internal_patterns:
        sanitized = pattern.sub("", sanitized)

    # Clean up excessive whitespace
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)

    return sanitized.strip()
