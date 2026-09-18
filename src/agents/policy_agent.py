"""
Policy Agent — RAG-powered agent that answers banking policy questions.
Uses only retrieved context from the vector store; never answers from
parametric memory alone.
"""

from crewai import Agent

from src.tools.policy_search_tool import search_policy_documents
from src.utils.config import settings


def create_policy_agent() -> Agent:
    """
    Create and return the Policy Specialist agent.

    This agent is responsible for answering all questions related to
    Cred Financial Services policies, including credit card, loans,
    KYC/AML, dispute resolution, and account closure procedures.

    It MUST use the policy search tool to retrieve relevant information
    before formulating any answer. It should never answer from memory.
    """
    return Agent(
        role="Banking Policy Specialist",
        goal=(
            "Accurately answer customer questions about Cred Financial Services "
            "policies by retrieving and citing relevant policy documents. "
            "Never fabricate policy details — always ground answers in "
            "retrieved document snippets with source attribution."
        ),
        backstory=(
            "You are a senior compliance and policy specialist at Cred Financial "
            "Services with 15 years of experience in banking regulations. "
            "You have deep expertise in credit card policies, loan terms, "
            "KYC/AML compliance, dispute resolution procedures, and account "
            "management rules. You always cite the specific policy document "
            "and section when providing answers. If the retrieved documents "
            "don't contain sufficient information, you clearly state that "
            "rather than guessing."
        ),
        tools=[search_policy_documents],
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
