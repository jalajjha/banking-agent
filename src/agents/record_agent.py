"""
Record Agent — Customer data retrieval agent.
Uses structured tools to look up customer records, account balances,
loan details, and transaction history from the mock database.
"""

from crewai import Agent

from src.tools.record_lookup_tool import (
    lookup_customer,
    get_account_balance,
    get_loan_details,
    get_recent_transactions,
)


def create_record_agent() -> Agent:
    """
    Create and return the Customer Record Lookup Specialist agent.

    This agent handles all customer-specific queries requiring data
    from the structured database: account information, balances,
    loan details, and transaction history.
    """
    return Agent(
        role="Customer Record Lookup Specialist",
        goal=(
            "Accurately retrieve and present customer-specific information "
            "including account details, balances, loan information, and "
            "recent transactions. Always verify the customer ID before "
            "providing any data. Present financial figures clearly with "
            "proper formatting."
        ),
        backstory=(
            "You are an experienced customer service representative at "
            "Cred Financial Services who specializes in account inquiries. "
            "You have access to the customer database and can quickly "
            "retrieve account balances, loan details, and transaction "
            "history. You always verify customer identity through their "
            "Customer ID before sharing any information. You present "
            "financial data clearly and offer helpful context about "
            "the customer's financial position."
        ),
        tools=[
            lookup_customer,
            get_account_balance,
            get_loan_details,
            get_recent_transactions,
        ],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
