"""
Customer record lookup tools for CrewAI agents.
Reads mock customer data from data/customers.json and provides
structured lookups for accounts, balances, loans, and transactions.
"""

import json
from pathlib import Path
from typing import Any

from crewai.tools import tool
from pydantic import BaseModel, Field

from src.utils.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ── In-memory customer database ──

_customer_db: dict[str, Any] | None = None


def _load_customer_db() -> dict[str, Any]:
    """Load and cache customer data from JSON file."""
    global _customer_db

    if _customer_db is not None:
        return _customer_db

    customers_file = settings.customers_file
    if not customers_file.exists():
        raise FileNotFoundError(
            f"Customer database not found: {customers_file}"
        )

    with open(customers_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Index by customer_id for O(1) lookup
    _customer_db = {
        customer["customer_id"]: customer
        for customer in data.get("customers", [])
    }

    logger.info("customer_db_loaded", count=len(_customer_db))
    return _customer_db


def _get_customer(customer_id: str) -> dict[str, Any] | None:
    """Retrieve a customer record by ID."""
    db = _load_customer_db()
    return db.get(customer_id)


# ── Pydantic Models ──


class CustomerSummary(BaseModel):
    """Summary of a customer profile."""

    customer_id: str
    name: str
    email: str
    kyc_status: str
    risk_profile: str
    num_accounts: int
    num_active_loans: int


class AccountInfo(BaseModel):
    """Account information."""

    account_id: str
    type: str
    balance: float | None = None
    credit_limit: float | None = None
    outstanding_balance: float | None = None
    status: str


class LoanInfo(BaseModel):
    """Loan information."""

    loan_id: str
    type: str
    principal: float
    outstanding: float
    interest_rate: float
    emi_amount: float
    remaining_emis: int
    next_emi_date: str | None
    status: str


class TransactionInfo(BaseModel):
    """Transaction record."""

    txn_id: str
    date: str
    type: str
    amount: float
    description: str
    category: str


# ── Tool Definitions ──


@tool("lookup_customer")
def lookup_customer(customer_id: str) -> str:
    """
    Look up a customer's profile by their Customer ID.

    Returns the customer's name, KYC status, risk profile, and a
    summary of their accounts and loans. Use this as the first step
    when a customer query includes a customer ID.

    Args:
        customer_id: The unique customer identifier (e.g., 'CUST001').

    Returns:
        A formatted string with the customer's profile summary,
        or an error message if the customer is not found.
    """
    try:
        customer = _get_customer(customer_id)
        if customer is None:
            logger.warning("customer_not_found", customer_id=customer_id)
            return f"Customer with ID '{customer_id}' was not found in our records."

        accounts = customer.get("accounts", [])
        loans = customer.get("loans", [])
        active_loans = [l for l in loans if l.get("status") != "closed"]

        summary = (
            f"Customer Profile:\n"
            f"  Name: {customer['name']}\n"
            f"  Customer ID: {customer['customer_id']}\n"
            f"  Email: {customer['email']}\n"
            f"  KYC Status: {customer['kyc_status']}\n"
            f"  Risk Profile: {customer['risk_profile']}\n"
            f"  Accounts: {len(accounts)}\n"
            f"  Active Loans: {len(active_loans)}\n"
        )

        # List accounts
        if accounts:
            summary += "\n  Accounts:\n"
            for acc in accounts:
                acc_type = acc.get("type", "unknown")
                acc_status = acc.get("status", "unknown")
                if acc_type == "savings":
                    summary += (
                        f"    - {acc['account_id']} ({acc_type}) | "
                        f"Balance: ₹{acc.get('balance', 0):,.2f} | "
                        f"Status: {acc_status}\n"
                    )
                elif acc_type == "credit_card":
                    summary += (
                        f"    - {acc['account_id']} ({acc.get('card_type', '')} Credit Card) | "
                        f"Limit: ₹{acc.get('credit_limit', 0):,.2f} | "
                        f"Outstanding: ₹{acc.get('outstanding_balance', 0):,.2f} | "
                        f"Status: {acc_status}\n"
                    )

        # List active loans
        if active_loans:
            summary += "\n  Active Loans:\n"
            for loan in active_loans:
                summary += (
                    f"    - {loan['loan_id']} ({loan['type']}) | "
                    f"Outstanding: ₹{loan.get('outstanding', 0):,.2f} | "
                    f"EMI: ₹{loan.get('emi_amount', 0):,.2f} | "
                    f"Status: {loan.get('status', 'unknown')}\n"
                )

        logger.info("customer_lookup_success", customer_id=customer_id)
        return summary

    except Exception as e:
        logger.error(
            "customer_lookup_error",
            customer_id=customer_id,
            error=str(e),
        )
        return f"Error looking up customer {customer_id}: {str(e)}"


@tool("get_account_balance")
def get_account_balance(customer_id: str) -> str:
    """
    Get the account balances for a customer.

    Returns savings account balances and credit card outstanding
    amounts for all accounts belonging to the customer.

    Args:
        customer_id: The unique customer identifier (e.g., 'CUST001').

    Returns:
        A formatted string with all account balances.
    """
    try:
        customer = _get_customer(customer_id)
        if customer is None:
            return f"Customer with ID '{customer_id}' was not found in our records."

        accounts = customer.get("accounts", [])
        if not accounts:
            return f"No accounts found for customer {customer_id}."

        result_parts: list[str] = [
            f"Account Balances for {customer['name']} ({customer_id}):\n"
        ]

        for acc in accounts:
            if acc["type"] == "savings":
                result_parts.append(
                    f"  Savings Account ({acc['account_id']}): "
                    f"₹{acc.get('balance', 0):,.2f}"
                )
            elif acc["type"] == "credit_card":
                result_parts.append(
                    f"  Credit Card ({acc['account_id']} — {acc.get('card_type', '')}):\n"
                    f"    Credit Limit: ₹{acc.get('credit_limit', 0):,.2f}\n"
                    f"    Outstanding: ₹{acc.get('outstanding_balance', 0):,.2f}\n"
                    f"    Available Credit: ₹{acc.get('credit_limit', 0) - acc.get('outstanding_balance', 0):,.2f}\n"
                    f"    Minimum Due: ₹{acc.get('minimum_due', 0):,.2f}\n"
                    f"    Due Date: {acc.get('due_date', 'N/A')}\n"
                    f"    Reward Points: {acc.get('reward_points', 0):,}"
                )

        logger.info("balance_lookup_success", customer_id=customer_id)
        return "\n".join(result_parts)

    except Exception as e:
        logger.error(
            "balance_lookup_error",
            customer_id=customer_id,
            error=str(e),
        )
        return f"Error retrieving balance for {customer_id}: {str(e)}"


@tool("get_loan_details")
def get_loan_details(customer_id: str) -> str:
    """
    Get detailed loan information for a customer.

    Returns details of all loans (active, overdue, and closed)
    including outstanding amounts, EMI details, and next payment dates.

    Args:
        customer_id: The unique customer identifier (e.g., 'CUST001').

    Returns:
        A formatted string with all loan details.
    """
    try:
        customer = _get_customer(customer_id)
        if customer is None:
            return f"Customer with ID '{customer_id}' was not found in our records."

        loans = customer.get("loans", [])
        if not loans:
            return f"No loans found for customer {customer_id} ({customer['name']})."

        result_parts: list[str] = [
            f"Loan Details for {customer['name']} ({customer_id}):\n"
        ]

        for loan in loans:
            status_emoji = {
                "active": "✅",
                "overdue": "⚠️",
                "closed": "🔒",
            }.get(loan.get("status", ""), "❓")

            result_parts.append(
                f"  {status_emoji} Loan {loan['loan_id']} ({loan['type'].title()}):\n"
                f"    Principal: ₹{loan.get('principal', 0):,.2f}\n"
                f"    Outstanding: ₹{loan.get('outstanding', 0):,.2f}\n"
                f"    Interest Rate: {loan.get('interest_rate', 0)}% p.a.\n"
                f"    EMI Amount: ₹{loan.get('emi_amount', 0):,.2f}\n"
                f"    Remaining EMIs: {loan.get('remaining_emis', 0)}\n"
                f"    Next EMI Date: {loan.get('next_emi_date', 'N/A')}\n"
                f"    Status: {loan.get('status', 'unknown').upper()}\n"
            )

        logger.info("loan_lookup_success", customer_id=customer_id)
        return "\n".join(result_parts)

    except Exception as e:
        logger.error(
            "loan_lookup_error",
            customer_id=customer_id,
            error=str(e),
        )
        return f"Error retrieving loans for {customer_id}: {str(e)}"


@tool("get_recent_transactions")
def get_recent_transactions(customer_id: str) -> str:
    """
    Get the most recent transactions for a customer.

    Returns a list of the customer's latest transactions including
    type, amount, description, and category.

    Args:
        customer_id: The unique customer identifier (e.g., 'CUST001').

    Returns:
        A formatted string with the recent transactions.
    """
    try:
        customer = _get_customer(customer_id)
        if customer is None:
            return f"Customer with ID '{customer_id}' was not found in our records."

        transactions = customer.get("recent_transactions", [])
        if not transactions:
            return f"No recent transactions found for customer {customer_id}."

        result_parts: list[str] = [
            f"Recent Transactions for {customer['name']} ({customer_id}):\n"
        ]

        for txn in transactions:
            txn_type = txn.get("type", "unknown")
            arrow = "⬆️" if txn_type == "credit" else "⬇️"

            result_parts.append(
                f"  {arrow} {txn['date']} | {txn['txn_id']}\n"
                f"    {txn_type.upper()}: ₹{txn.get('amount', 0):,.2f}\n"
                f"    Description: {txn.get('description', 'N/A')}\n"
                f"    Category: {txn.get('category', 'N/A')}\n"
                f"    Account: {txn.get('account_id', 'N/A')}\n"
            )

        logger.info(
            "transactions_lookup_success",
            customer_id=customer_id,
            count=len(transactions),
        )
        return "\n".join(result_parts)

    except Exception as e:
        logger.error(
            "transactions_lookup_error",
            customer_id=customer_id,
            error=str(e),
        )
        return f"Error retrieving transactions for {customer_id}: {str(e)}"
