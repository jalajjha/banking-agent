"""
CrewAI Crew Orchestrator — Routes queries to the appropriate agent(s),
manages conversation memory, and returns structured outputs.
"""

from typing import Any

from crewai import Agent, Crew, Task, Process
from pydantic import BaseModel, Field

from src.agents.policy_agent import create_policy_agent
from src.agents.record_agent import create_record_agent
from src.agents.support_agent import create_support_agent
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ── Structured Output Models ──


class AgentResponse(BaseModel):
    """Structured response from the crew orchestrator."""

    answer: str = Field(..., description="The agent's response to the query")
    agent_used: str = Field(
        ..., description="Which agent handled the query"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source references (policy documents, chunk IDs)",
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level: high, medium, or low",
    )


# ── Query Classification ──

POLICY_KEYWORDS: set[str] = {
    "policy", "rule", "regulation", "guideline", "procedure",
    "kyc", "aml", "compliance", "dispute", "chargeback",
    "late payment", "fee", "charge", "interest rate", "apr",
    "closure", "close my account", "dormant", "inactive", "penalty", "foreclosure",
    "prepayment", "insurance", "reward", "eligibility",
    "credit limit", "processing fee", "grievance", "ombudsman",
    "refund", "compensation", "emi default", "npa", "prerequisite",
}

RECORD_KEYWORDS: set[str] = {
    "balance", "account", "transaction", "statement", "loan",
    "emi", "payment", "due", "outstanding", "credit card",
    "recent", "history", "my account", "my balance", "my loan",
    "my transaction", "my payment", "my emi",
}


def classify_query(query: str, customer_id: str | None = None) -> str:
    """
    Classify a query to determine which agent should handle it.

    Args:
        query: The user's query string.
        customer_id: Optional customer ID (presence suggests record query).

    Returns:
        One of: 'policy', 'record', 'support'.
    """
    query_lower = query.lower()

    # If a customer ID is provided and query has record keywords → record agent
    if customer_id:
        for keyword in RECORD_KEYWORDS:
            if keyword in query_lower:
                return "record"

    # Check for policy keywords
    for keyword in POLICY_KEYWORDS:
        if keyword in query_lower:
            return "policy"

    # Check for record keywords (even without customer_id)
    for keyword in RECORD_KEYWORDS:
        if keyword in query_lower:
            return "record"

    # Default to support agent
    return "support"


class CredSupportCrew:
    """
    Main orchestrator that creates agents, classifies queries,
    builds tasks, and executes the CrewAI crew.
    """

    def __init__(self) -> None:
        self._policy_agent: Agent = create_policy_agent()
        self._record_agent: Agent = create_record_agent()
        self._support_agent: Agent = create_support_agent()
        self._conversation_memory: list[dict[str, str]] = []

    def _build_context_from_memory(self) -> str:
        """Build a conversation context string from recent memory."""
        if not self._conversation_memory:
            return ""

        # Keep last 5 exchanges for context
        recent = self._conversation_memory[-5:]
        context_parts = ["Previous conversation context:"]
        for entry in recent:
            context_parts.append(
                f"  User: {entry.get('query', '')}\n"
                f"  Agent: {entry.get('response', '')[:200]}..."
            )
        return "\n".join(context_parts)

    def _create_task(
        self,
        query: str,
        agent: Agent,
        agent_type: str,
        customer_id: str | None = None,
    ) -> Task:
        """Create a CrewAI Task for the given query and agent."""
        context_str = self._build_context_from_memory()

        if agent_type == "policy":
            description = (
                f"Answer the following banking policy question by searching the "
                f"policy documents. You MUST use the search_policy_documents tool "
                f"to retrieve relevant information before answering. Include "
                f"source references (document name and section) in your answer.\n\n"
                f"Question: {query}\n\n"
                f"{context_str}"
            )
            expected_output = (
                "A clear, accurate answer to the policy question, grounded in "
                "retrieved policy document excerpts. Include specific source "
                "references (e.g., 'According to POL-CC-001, Section 3.1...'). "
                "If insufficient information is found, clearly state that."
            )

        elif agent_type == "record":
            cid_note = (
                f"Customer ID: {customer_id}"
                if customer_id
                else "No customer ID provided — ask the user for their Customer ID."
            )
            description = (
                f"Retrieve and present the requested customer information. "
                f"Use the available lookup tools to get the required data.\n\n"
                f"Query: {query}\n"
                f"{cid_note}\n\n"
                f"{context_str}"
            )
            expected_output = (
                "Clear, well-formatted customer data including relevant "
                "account details, balances, loan information, or transaction "
                "history as requested. Present amounts in INR with proper formatting."
            )

        else:  # support
            description = (
                f"Provide helpful general banking support for the following query. "
                f"Stay within the banking and financial services domain. "
                f"If the query requires specific policy details or customer records, "
                f"advise the user accordingly.\n\n"
                f"Query: {query}\n\n"
                f"{context_str}"
            )
            expected_output = (
                "A helpful, clear response addressing the customer's general "
                "banking query. Stay within the banking domain and provide "
                "actionable guidance."
            )

        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )

    def process_query(
        self,
        query: str,
        customer_id: str | None = None,
    ) -> AgentResponse:
        """
        Process a customer query through the appropriate agent.

        Args:
            query: The customer's question or request.
            customer_id: Optional customer ID for record lookups.

        Returns:
            AgentResponse with the answer, agent used, and sources.
        """
        # Step 1: Classify the query
        agent_type = classify_query(query, customer_id)
        logger.info(
            "query_classified",
            query=query[:100],
            agent_type=agent_type,
            customer_id=customer_id,
        )

        # Step 2: Select the appropriate agent
        agent_map: dict[str, Agent] = {
            "policy": self._policy_agent,
            "record": self._record_agent,
            "support": self._support_agent,
        }
        agent = agent_map[agent_type]

        # Step 3: Build the task
        task = self._create_task(query, agent, agent_type, customer_id)

        # Step 4: Execute the crew
        try:
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )

            result = crew.kickoff()
            answer = str(result)

            # Step 5: Store in conversation memory
            self._conversation_memory.append(
                {"query": query, "response": answer, "agent": agent_type}
            )

            logger.info(
                "query_processed",
                agent_type=agent_type,
                response_length=len(answer),
            )

            return AgentResponse(
                answer=answer,
                agent_used=agent_type,
                sources=[],  # Sources are embedded in the answer text
                confidence="high" if agent_type != "support" else "medium",
            )

        except Exception as e:
            logger.error(
                "crew_execution_error",
                query=query[:100],
                agent_type=agent_type,
                error=str(e),
            )
            return AgentResponse(
                answer=(
                    "I apologize, but I encountered an error processing your "
                    "request. Please try again or contact our support team at "
                    "1800-XXX-XXXX for immediate assistance."
                ),
                agent_used=agent_type,
                sources=[],
                confidence="low",
            )

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self._conversation_memory.clear()
        logger.info("conversation_memory_cleared")


# Singleton instance
cred_crew = CredSupportCrew()
