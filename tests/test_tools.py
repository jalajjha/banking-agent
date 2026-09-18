"""
Unit tests for the record lookup and policy search tools.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.tools.record_lookup_tool import (
    _load_customer_db,
    _get_customer,
    lookup_customer,
    get_account_balance,
    get_loan_details,
    get_recent_transactions,
)


# ════════════════════════════════════════════════════════════════
# RECORD LOOKUP TOOL TESTS
# ════════════════════════════════════════════════════════════════


class TestCustomerDatabase:
    """Tests for the customer database loading and lookup."""

    def setup_method(self):
        """Reset the cached database before each test."""
        import src.tools.record_lookup_tool as module
        module._customer_db = None

    def test_loads_customer_db(self):
        """Test that the customer database loads successfully."""
        db = _load_customer_db()
        assert isinstance(db, dict)
        assert len(db) > 0

    def test_get_existing_customer(self):
        """Test retrieval of an existing customer."""
        customer = _get_customer("CUST001")
        assert customer is not None
        assert customer["customer_id"] == "CUST001"
        assert "name" in customer
        assert "accounts" in customer

    def test_get_nonexistent_customer(self):
        """Test retrieval of a non-existent customer returns None."""
        customer = _get_customer("CUST999")
        assert customer is None


class TestLookupCustomer:
    """Tests for the lookup_customer tool."""

    def setup_method(self):
        import src.tools.record_lookup_tool as module
        module._customer_db = None

    def test_lookup_existing_customer(self):
        """Test looking up an existing customer returns profile summary."""
        result = lookup_customer.run("CUST001")
        assert "Rajesh Kumar Sharma" in result
        assert "CUST001" in result
        assert "KYC Status" in result

    def test_lookup_nonexistent_customer(self):
        """Test looking up a non-existent customer returns error message."""
        result = lookup_customer.run("CUST999")
        assert "not found" in result.lower()

    def test_lookup_shows_accounts(self):
        """Test that customer lookup includes account information."""
        result = lookup_customer.run("CUST001")
        assert "ACC-001" in result or "Savings" in result or "savings" in result


class TestGetAccountBalance:
    """Tests for the get_account_balance tool."""

    def setup_method(self):
        import src.tools.record_lookup_tool as module
        module._customer_db = None

    def test_balance_existing_customer(self):
        """Test getting balance for an existing customer."""
        result = get_account_balance.run("CUST001")
        assert "Balance" in result or "balance" in result
        assert "CUST001" in result or "Rajesh" in result

    def test_balance_nonexistent_customer(self):
        """Test getting balance for non-existent customer."""
        result = get_account_balance.run("CUST999")
        assert "not found" in result.lower()


class TestGetLoanDetails:
    """Tests for the get_loan_details tool."""

    def setup_method(self):
        import src.tools.record_lookup_tool as module
        module._customer_db = None

    def test_loans_for_customer_with_loans(self):
        """Test getting loan details for a customer who has loans."""
        result = get_loan_details.run("CUST001")
        assert "Loan" in result or "loan" in result
        assert "EMI" in result or "emi" in result

    def test_loans_for_customer_without_loans(self):
        """Test getting loans for a customer with no active loans."""
        result = get_loan_details.run("CUST002")
        assert "No loans" in result or "no loans" in result.lower()

    def test_loans_nonexistent_customer(self):
        """Test getting loans for non-existent customer."""
        result = get_loan_details.run("CUST999")
        assert "not found" in result.lower()


class TestGetRecentTransactions:
    """Tests for the get_recent_transactions tool."""

    def setup_method(self):
        import src.tools.record_lookup_tool as module
        module._customer_db = None

    def test_transactions_existing_customer(self):
        """Test getting transactions for an existing customer."""
        result = get_recent_transactions.run("CUST001")
        assert "Transaction" in result or "transaction" in result.lower()

    def test_transactions_nonexistent_customer(self):
        """Test transactions for non-existent customer."""
        result = get_recent_transactions.run("CUST999")
        assert "not found" in result.lower()
