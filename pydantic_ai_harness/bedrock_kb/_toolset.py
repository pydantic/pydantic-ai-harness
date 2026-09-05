"""Bedrock Knowledge Base toolset: retrieval and ingestion tools."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

logger = logging.getLogger(__name__)


class BedrockKBToolset(FunctionToolset[AgentDepsT]):
    """Toolset providing Amazon Bedrock Knowledge Base retrieval and ingestion.

    Tools:
    - search_knowledge_base: Retrieve relevant documents for a query
    - ingest_document: Ingest a document into the knowledge base (optional)
    """

    def __init__(
        self,
        *,
        knowledge_base_id: str,
        region_name: str,
        data_source_id: str,
        data_source_type: str,
        data_source_bucket: str,
        use_agentic_retrieval: bool,
        number_of_results: int,
        include_ingest_tool: bool,
    ) -> None:
        super().__init__()
        self._knowledge_base_id = knowledge_base_id or os.environ.get('KNOWLEDGE_BASE_ID', '')
        self._region_name = region_name or os.environ.get('AWS_REGION', 'us-east-1')
        self._data_source_id = data_source_id or os.environ.get('BEDROCK_DATA_SOURCE_ID', '')
        self._data_source_type = data_source_type.upper()
        self._data_source_bucket = data_source_bucket or os.environ.get('BEDROCK_DATA_SOURCE_BUCKET', '')
        self._use_agentic_retrieval = use_agentic_retrieval
        self._number_of_results = number_of_results
        self._runtime_client: Any = None
        self._agent_client: Any = None

        self.add_function(self.search_knowledge_base, name='search_knowledge_base')
        if include_ingest_tool:
            self.add_function(self.ingest_document, name='ingest_document')

    @property
    def _get_runtime_client(self) -> Any:
        if self._runtime_client is None:
            import boto3
            from botocore.config import Config

            self._runtime_client = boto3.client(
                'bedrock-agent-runtime',
                region_name=self._region_name,
                config=Config(user_agent_extra='pydantic-ai-harness/bedrock-kb'),
            )
        return self._runtime_client

    @property
    def _get_agent_client(self) -> Any:
        if self._agent_client is None:
            import boto3
            from botocore.config import Config

            self._agent_client = boto3.client(
                'bedrock-agent',
                region_name=self._region_name,
                config=Config(user_agent_extra='pydantic-ai-harness/bedrock-kb'),
            )
        return self._agent_client

    async def search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base for documents relevant to the query.

        Args:
            query: The natural language question or search query.

        Returns:
            Formatted search results with content and metadata.
        """
        if not self._knowledge_base_id:
            raise ModelRetry('Knowledge Base ID is not configured. Set KNOWLEDGE_BASE_ID env var.')

        try:
            if self._use_agentic_retrieval:
                return await self._agentic_retrieve(query)
            else:
                return await self._standard_retrieve(query)
        except Exception as e:
            logger.warning('KB search failed: %s', e)
            raise ModelRetry(f'Knowledge Base search failed: {e}') from e

    async def _agentic_retrieve(self, query: str) -> str:
        """Use AgenticRetrieveStream for multi-step reasoning retrieval."""
        import asyncio

        def _call() -> str:
            response = self._get_runtime_client.retrieve_and_generate(
                input={'text': query},
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': self._knowledge_base_id,
                        'modelArn': 'arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-haiku-20240307-v1:0',
                    },
                },
            )
            # RetrieveAndGenerate returns generated output + citations
            output = response.get('output', {}).get('text', '')
            citations = response.get('citations', [])
            results = [f'Answer: {output}']
            for citation in citations[:self._number_of_results]:
                for ref in citation.get('retrievedReferences', []):
                    content = ref.get('content', {}).get('text', '')
                    source = ref.get('location', {}).get('s3Location', {}).get('uri', 'unknown')
                    results.append(f'(Source: {source})\n{content[:200]}...')
            return '\n\n---\n\n'.join(results) if results else 'No results found.'

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _call)

    async def _standard_retrieve(self, query: str) -> str:
        """Use standard Retrieve API for semantic search."""
        import asyncio

        def _call() -> str:
            kwargs: dict = {
                'knowledgeBaseId': self._knowledge_base_id,
                'retrievalQuery': {'text': query},
            }
            # managedSearchConfiguration for managed KBs, vectorSearchConfiguration for vector KBs
            # Let the API auto-detect by not specifying configuration (works for both)
            response = self._get_runtime_client.retrieve(**kwargs)
            results = []
            for r in response.get('retrievalResults', []):
                content = r.get('content', {}).get('text', '')
                score = r.get('score', 0)
                source = r.get('location', {}).get('s3Location', {}).get('uri', 'unknown')
                results.append(f'[Score: {score:.2f}] (Source: {source})\n{content}')
            if not results:
                return 'No results found.'
            return '\n\n---\n\n'.join(results)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _call)

    async def ingest_document(
        self,
        content: str,
        document_id: str = '',
        mime_type: str = '',
        s3_uri: str = '',
    ) -> str:
        """Ingest a document into the knowledge base.

        Supports three modes:
        - Inline text: provide `content` with plain text
        - S3 reference: provide `s3_uri` to ingest an existing S3 object
        - Binary: provide base64-encoded `content` with `mime_type`

        Args:
            content: Document text content, or base64-encoded bytes if mime_type is set.
            document_id: Optional identifier for the document.
            mime_type: MIME type for binary content (e.g., 'application/pdf').
            s3_uri: S3 URI to ingest (e.g., 's3://bucket/key'). Overrides content.

        Returns:
            Ingestion status message.
        """
        if not self._knowledge_base_id:
            raise ModelRetry('Knowledge Base ID is not configured.')
        if not self._data_source_id:
            raise ModelRetry('Data source ID is not configured for ingestion.')

        import asyncio
        import uuid

        doc_id = document_id or str(uuid.uuid4())

        def _call() -> str:
            if self._data_source_type == 'CUSTOM':
                return self._ingest_direct(doc_id, content, mime_type, s3_uri)
            else:
                return self._ingest_s3(doc_id, content, mime_type)

        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, _call)
        except Exception as e:
            logger.warning('Ingestion failed: %s', e)
            raise ModelRetry(f'Document ingestion failed: {e}') from e

    def _ingest_direct(self, doc_id: str, content: str, mime_type: str, s3_uri: str) -> str:
        """Ingest via IngestKnowledgeBaseDocuments API (CUSTOM data source)."""
        if s3_uri:
            doc = {
                'content': {
                    'dataSourceType': 'CUSTOM',
                    'custom': {
                        'customDocumentIdentifier': {'id': doc_id},
                        'sourceType': 'S3_LOCATION',
                        's3Location': {'uri': s3_uri},
                    },
                },
            }
        elif mime_type:
            doc = {
                'content': {
                    'dataSourceType': 'CUSTOM',
                    'custom': {
                        'customDocumentIdentifier': {'id': doc_id},
                        'sourceType': 'IN_LINE',
                        'inlineContent': {
                            'type': 'BYTE',
                            'byteContent': {'data': content, 'mimeType': mime_type},
                        },
                    },
                },
            }
        else:
            doc = {
                'content': {
                    'dataSourceType': 'CUSTOM',
                    'custom': {
                        'customDocumentIdentifier': {'id': doc_id},
                        'sourceType': 'IN_LINE',
                        'inlineContent': {
                            'type': 'TEXT',
                            'textContent': {'data': content},
                        },
                    },
                },
            }

        response = self._get_agent_client.ingest_knowledge_base_documents(
            knowledgeBaseId=self._knowledge_base_id,
            dataSourceId=self._data_source_id,
            documents=[doc],
        )
        status = response.get('documentDetails', [{}])[0].get('status', 'UNKNOWN')
        return f'Document "{doc_id}" ingestion started. Status: {status}'

    def _ingest_s3(self, doc_id: str, content: str, mime_type: str) -> str:
        """Ingest via S3 upload + StartIngestionJob."""
        import boto3

        if not self._data_source_bucket:
            return 'Error: No S3 bucket configured for ingestion.'

        s3 = boto3.client('s3', region_name=self._region_name)
        key = f'pydantic-ai-harness/{doc_id}.txt'
        s3.put_object(
            Bucket=self._data_source_bucket,
            Key=key,
            Body=content.encode('utf-8'),
            ContentType=mime_type or 'text/plain',
        )

        self._get_agent_client.start_ingestion_job(
            knowledgeBaseId=self._knowledge_base_id,
            dataSourceId=self._data_source_id,
            description=f'Ingest {doc_id} via pydantic-ai-harness',
        )
        return f'Document "{doc_id}" uploaded to s3://{self._data_source_bucket}/{key} and ingestion started.'
