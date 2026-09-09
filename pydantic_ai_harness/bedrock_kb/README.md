# Bedrock Knowledge Base

Connect your Pydantic AI agent to [Amazon Bedrock Managed Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html) for retrieval-augmented generation (RAG).

## Features

- **Agentic retrieval** — Multi-step, reasoning-enhanced search via `AgenticRetrieveStream`
- **Standard retrieval** — Semantic vector search via `Retrieve`
- **Direct ingestion** — Add documents without S3 using `IngestKnowledgeBaseDocuments` (CUSTOM data source)
- **S3 ingestion** — Upload to S3 + trigger sync (S3 data source)

## Quick start

```bash
uv add "pydantic-ai-harness[bedrock-kb]"
```

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

result = agent.run_sync('What are the company policies on remote work?')
print(result.output)
```

## Configuration

| Parameter | Description | Default |
|---|---|---|
| `knowledge_base_id` | Bedrock KB ID | Env: `KNOWLEDGE_BASE_ID` |
| `region_name` | AWS region | Env: `AWS_REGION` or `us-east-1` |
| `use_agentic_retrieval` | Use agentic multi-step retrieval | `True` |
| `number_of_results` | Max results per search | `5` |
| `data_source_id` | Data source ID (for ingestion) | Env: `BEDROCK_DATA_SOURCE_ID` |
| `data_source_type` | `"S3"` or `"CUSTOM"` | `"S3"` |
| `data_source_bucket` | S3 bucket (for S3 mode) | Env: `BEDROCK_DATA_SOURCE_BUCKET` |
| `include_ingest_tool` | Expose `ingest_document` tool | `False` |

## Tools exposed to the agent

### `search_knowledge_base(query: str) -> str`
Searches the KB for documents relevant to the query. Uses agentic retrieval by default.

### `ingest_document(content, document_id, mime_type, s3_uri) -> str`
*(Only when `include_ingest_tool=True`)*

Ingests a document into the KB. Three modes:
- **Inline text**: `content="Your text here"`
- **S3 reference**: `s3_uri="s3://bucket/file.pdf"`
- **Binary**: `content="<base64>", mime_type="application/pdf"`

## IAM Permissions

```json
{
    "Effect": "Allow",
    "Action": [
        "bedrock:Retrieve",
        "bedrock:AgenticRetrieveStream"
    ],
    "Resource": "arn:aws:bedrock:REGION:ACCOUNT:knowledge-base/KB_ID"
}
```

For ingestion, also add:
```json
{
    "Effect": "Allow",
    "Action": "bedrock:IngestKnowledgeBaseDocuments",
    "Resource": "arn:aws:bedrock:REGION:ACCOUNT:knowledge-base/KB_ID"
}
```

## Prerequisites

- AWS credentials configured (via environment, IAM role, or AWS profile)
- A Bedrock Managed Knowledge Base created (via console, CDK, or CloudFormation)
- `boto3 >= 1.43.2` installed

## References

- [Ingest documents directly into a knowledge base](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-direct-ingestion.html)
- [IngestKnowledgeBaseDocuments API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_IngestKnowledgeBaseDocuments.html)
- [Connect to a custom data source](https://docs.aws.amazon.com/bedrock/latest/userguide/custom-data-source-connector.html)
- [AgenticRetrieveStream API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_AgenticRetrieveStream.html)
