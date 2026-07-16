"""Bedrock Knowledge Base capability: retrieval-augmented generation using Amazon Bedrock Managed Knowledge Bases."""

from pydantic_ai_harness.bedrock_kb._capability import BedrockKnowledgeBase
from pydantic_ai_harness.bedrock_kb._toolset import BedrockKBToolset

__all__ = ['BedrockKnowledgeBase', 'BedrockKBToolset']
