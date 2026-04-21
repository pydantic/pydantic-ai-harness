# Home Automation

Home Automation exposes smart home entities and service calls to a Pydantic AI
agent. The current backend targets the Home Assistant REST API.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import HomeAutomation
from pydantic_ai_harness.home_automation import HomeAssistantBackend

backend = HomeAssistantBackend(
    url='http://localhost:8123',
    token='...',
)

agent = Agent(
    'openai:gpt-5',
    capabilities=[HomeAutomation(backend=backend)],
)
```

## Tools

The capability exposes the backend methods directly as tools:

- `list_services(domain=None)`: list callable Home Assistant services and their normalized arguments.
- `list_entities(domain=None)`: list entity summaries.
- `get_state(entity_id)`: get the current state for one entity.
- `list_states(domain=None)`: list current entity states.
- `call_service(domain, entity_id, service_name, **data)`: call a service for one entity.

Agents should usually discover entities with `list_entities`, inspect available
services with `list_services`, then call `call_service` with the exact
`domain`, `service_name`, and `entity_id`.

## Home Assistant Backend

`HomeAssistantBackend` validates `/api/services` and `/api/states` responses
with Pydantic models, then converts them into backend-neutral dataclasses for
the agent-facing tools.

Service calls return a `ServiceCallResult` with:

- `changed_states`: states Home Assistant reports as changed during execution.
- `service_response`: response data for services that support or require it.
- `verified_state`: a best-effort post-call state for the target entity.

If Home Assistant returns no changed states or service response data,
`HomeAssistantBackend` may poll the target entity state to populate
`verified_state`. Configure this with `verification_poll_attempts` and
`verification_poll_interval`.

```python
backend = HomeAssistantBackend(
    url='http://localhost:8123',
    token='...',
    verification_poll_attempts=3,
    verification_poll_interval=0.5,
)
```

Injected `httpx.AsyncClient` instances are not closed by `aclose`; clients
created by the backend are closed by `aclose`.
