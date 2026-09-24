"""Bedrock Knowledge Base capability: connect Pydantic AI agents to Amazon Bedrock Managed KBs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.bedrock_kb._toolset import BedrockKBToolset


@dataclass
class BedrockKnowledgeBase(AbstractCapability[AgentDepsT]):
    """Amazon Bedrock Knowledge Base retrieval capability.

    Gives agents access to a Bedrock Managed Knowledge Base for RAG. Supports:
    - Agentic retrieval (multi-step, reasoning-enhanced search)
    - Standard semantic retrieval
    - Direct document ingestion (CUSTOM data source)

    Usage:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.bedrock_kb import BedrockKnowledgeBase

        agent = Agent(
            'anthropic:claude-sonnet-4-20250514',
            capabilities=[
                BedrockKnowledgeBase(
                    knowledge_base_id='YOUR_KB_ID',
                    region_name='us-west-2',
                ),
            ],
        )
        ```

    References:
        - https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html
        - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_AgenticRetrieveStream.html
        - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_IngestKnowledgeBaseDocuments.html
    """

    knowledge_base_id: str = ''
    """Knowledge Base ID. Falls back to KNOWLEDGE_BASE_ID env var."""

    region_name: str = ''
    """AWS region. Falls back to AWS_REGION env var, then us-east-1."""

    data_source_id: str = ''
    """Data source ID for ingestion. Falls back to BEDROCK_DATA_SOURCE_ID env var."""

    data_source_type: str = 'S3'
    """Data source type: 'S3' (upload + sync) or 'CUSTOM' (direct ingestion via DLA)."""

    data_source_bucket: str = ''
    """S3 bucket for S3 data source mode. Falls back to BEDROCK_DATA_SOURCE_BUCKET env var."""

    use_agentic_retrieval: bool = True
    """Use agentic retrieval (multi-step reasoning) instead of standard Retrieve."""

    number_of_results: int = 5
    """Maximum number of results to return from retrieval."""

    include_ingest_tool: bool = False
    """Whether to expose the ingest_document tool to the agent."""

    def __post_init__(self) -> None:
        if self.number_of_results <= 0:
            raise ValueError(f'number_of_results must be positive, got {self.number_of_results}')

    def get_toolset(self) -> BedrockKBToolset[AgentDepsT]:
        """Build and return the Bedrock KB toolset."""
        return BedrockKBToolset[AgentDepsT](
            knowledge_base_id=self.knowledge_base_id,
            region_name=self.region_name,
            data_source_id=self.data_source_id,
            data_source_type=self.data_source_type,
            data_source_bucket=self.data_source_bucket,
            use_agentic_retrieval=self.use_agentic_retrieval,
            number_of_results=self.number_of_results,
            include_ingest_tool=self.include_ingest_tool,
        )
