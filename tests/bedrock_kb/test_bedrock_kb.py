"""Tests for Bedrock Knowledge Base capability."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBedrockKBToolset:
    """Tests for BedrockKBToolset."""

    def _make_toolset(self, **kwargs):
        from pydantic_ai_harness.bedrock_kb._toolset import BedrockKBToolset

        defaults = {
            'knowledge_base_id': 'TEST_KB',
            'region_name': 'us-west-2',
            'data_source_id': 'TEST_DS',
            'data_source_type': 'CUSTOM',
            'data_source_bucket': '',
            'use_agentic_retrieval': False,
            'number_of_results': 5,
            'include_ingest_tool': True,
        }
        defaults.update(kwargs)
        return BedrockKBToolset(**defaults)

    def test_standard_retrieve(self):
        ts = self._make_toolset()
        mock_client = MagicMock()
        mock_client.retrieve.return_value = {
            'retrievalResults': [
                {'content': {'text': 'Result 1'}, 'score': 0.95, 'location': {'s3Location': {'uri': 's3://b/k'}}},
                {'content': {'text': 'Result 2'}, 'score': 0.80, 'location': {'s3Location': {'uri': 's3://b/k2'}}},
            ]
        }
        ts._runtime_client = mock_client

        import asyncio
        result = asyncio.run(ts.search_knowledge_base('test query'))

        assert 'Result 1' in result
        assert 'Result 2' in result
        assert '0.95' in result
        mock_client.retrieve.assert_called_once()

    def test_ingest_direct_inline_text(self):
        ts = self._make_toolset()
        mock_client = MagicMock()
        mock_client.ingest_knowledge_base_documents.return_value = {
            'documentDetails': [{'status': 'STARTING'}]
        }
        ts._agent_client = mock_client

        import asyncio
        result = asyncio.run(ts.ingest_document(content='Hello world', document_id='doc-001'))

        assert 'doc-001' in result
        assert 'STARTING' in result
        call_kwargs = mock_client.ingest_knowledge_base_documents.call_args.kwargs
        doc = call_kwargs['documents'][0]
        assert doc['content']['custom']['inlineContent']['type'] == 'TEXT'
        assert doc['content']['custom']['inlineContent']['textContent']['data'] == 'Hello world'

    def test_ingest_direct_s3_reference(self):
        ts = self._make_toolset()
        mock_client = MagicMock()
        mock_client.ingest_knowledge_base_documents.return_value = {
            'documentDetails': [{'status': 'STARTING'}]
        }
        ts._agent_client = mock_client

        import asyncio
        result = asyncio.run(ts.ingest_document(content='', s3_uri='s3://bucket/file.pdf', document_id='s3-001'))

        assert 's3-001' in result
        doc = mock_client.ingest_knowledge_base_documents.call_args.kwargs['documents'][0]
        assert doc['content']['custom']['sourceType'] == 'S3_LOCATION'
        assert doc['content']['custom']['s3Location']['uri'] == 's3://bucket/file.pdf'

    def test_ingest_direct_binary(self):
        ts = self._make_toolset()
        mock_client = MagicMock()
        mock_client.ingest_knowledge_base_documents.return_value = {
            'documentDetails': [{'status': 'STARTING'}]
        }
        ts._agent_client = mock_client

        import asyncio
        result = asyncio.run(ts.ingest_document(content='base64data', mime_type='application/pdf', document_id='bin-001'))

        assert 'bin-001' in result
        doc = mock_client.ingest_knowledge_base_documents.call_args.kwargs['documents'][0]
        assert doc['content']['custom']['inlineContent']['type'] == 'BYTE'
        assert doc['content']['custom']['inlineContent']['byteContent']['mimeType'] == 'application/pdf'

    def test_ingest_s3_mode(self):
        ts = self._make_toolset(data_source_type='S3', data_source_bucket='test-bucket')
        mock_agent = MagicMock()
        ts._agent_client = mock_agent

        with patch('boto3.client') as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.return_value = mock_s3

            import asyncio
            result = asyncio.run(ts.ingest_document(content='Doc content', document_id='s3-doc'))

        assert 's3-doc' in result
        assert 'uploaded' in result
        mock_agent.start_ingestion_job.assert_called_once()

    def test_no_kb_id_raises_model_retry(self):
        from pydantic_ai.exceptions import ModelRetry

        ts = self._make_toolset(knowledge_base_id='')

        import asyncio
        with pytest.raises(ModelRetry, match='Knowledge Base ID'):
            asyncio.run(ts.search_knowledge_base('test'))

    def test_no_ds_id_raises_model_retry_on_ingest(self):
        from pydantic_ai.exceptions import ModelRetry

        ts = self._make_toolset(data_source_id='')

        import asyncio
        with pytest.raises(ModelRetry, match='Data source ID'):
            asyncio.run(ts.ingest_document(content='test'))


class TestBedrockKnowledgeBaseCapability:
    """Tests for the BedrockKnowledgeBase capability class."""

    def test_default_values(self):
        from pydantic_ai_harness.bedrock_kb._capability import BedrockKnowledgeBase

        kb = BedrockKnowledgeBase()
        assert kb.use_agentic_retrieval is True
        assert kb.number_of_results == 5
        assert kb.data_source_type == 'S3'
        assert kb.include_ingest_tool is False

    def test_invalid_number_of_results(self):
        from pydantic_ai_harness.bedrock_kb._capability import BedrockKnowledgeBase

        with pytest.raises(ValueError, match='number_of_results must be positive'):
            BedrockKnowledgeBase(number_of_results=0)

    def test_get_toolset_returns_toolset(self):
        from pydantic_ai_harness.bedrock_kb._capability import BedrockKnowledgeBase

        kb = BedrockKnowledgeBase(knowledge_base_id='TEST', region_name='us-west-2')
        toolset = kb.get_toolset()
        assert toolset._knowledge_base_id == 'TEST'
        assert toolset._region_name == 'us-west-2'
