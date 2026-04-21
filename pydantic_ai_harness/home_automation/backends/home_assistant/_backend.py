import asyncio
from typing import Any

import httpx

from pydantic_ai_harness.home_automation.backends import Entity, EntityState, HomeBackend, Service, ServiceCallResult
from pydantic_ai_harness.home_automation.backends.home_assistant.models import (
    ENTITY_STATE_CATALOG_ADAPTER,
    SERVICE_CATALOG_ADAPTER,
    HAEntityState,
    HAServiceCallResult,
    ServiceCatalog,
    ServiceDescription,
)


class HomeAssistantBackend(HomeBackend):
    """Home Assistant adapter focused on `/api/services`."""

    def __init__(
        self,
        url: str,
        token: str,
        client: httpx.AsyncClient | None = None,
        *,
        verification_poll_attempts: int = 3,
        verification_poll_interval: float = 0.5,
    ) -> None:
        if verification_poll_attempts < 0:
            raise ValueError('verification_poll_attempts must be greater than or equal to 0.')
        if verification_poll_interval < 0:
            raise ValueError('verification_poll_interval must be greater than or equal to 0.')

        self.url = url.rstrip('/')
        self.token = token
        self._service_catalog_cache: ServiceCatalog | None = None
        self._verification_poll_attempts = verification_poll_attempts
        self._verification_poll_interval = verification_poll_interval
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url=self.url,
            headers={
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json',
            },
            timeout=15.0,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._owns_client:
            await self.client.aclose()

    async def list_services(self, domain: str | None = None) -> list[Service]:
        """Fetch and validate the Home Assistant REST service catalog."""
        catalog = await self._get_service_catalog()
        if domain is not None:
            catalog = [item for item in catalog if item.domain == domain]
        return [service for domain_services in catalog for service in domain_services.to_services()]

    async def list_entities(self, domain: str | None = None) -> list[Entity]:
        """Return Home Assistant entities, optionally filtered by domain."""
        ha_entities = await self._fetch_state_catalog()
        if domain is not None:
            return [ha_entity.to_entity() for ha_entity in ha_entities if ha_entity.domain == domain]
        return [ha_entity.to_entity() for ha_entity in ha_entities]

    async def get_state(self, entity_id: str) -> EntityState:
        """Fetch the current state for a single Home Assistant entity."""
        response = await self.client.get(f'/api/states/{entity_id}')
        if response.status_code == 404:
            raise ValueError(f'Entity {entity_id!r} was not found.')
        response.raise_for_status()
        ha_state = HAEntityState.model_validate(response.json())
        return ha_state.to_state()

    async def call_service(
        self,
        domain: str,
        entity_id: str,
        service_name: str,
        *,
        want_response: bool = False,
        **data: Any,
    ) -> ServiceCallResult:
        """Call one Home Assistant service and normalize the response.

        If Home Assistant returns no changed states or response data, this may perform
        follow-up state reads to populate ``verified_state``.
        """
        service_description = await self._get_service_description(domain, service_name)
        if service_description is None:
            raise ValueError(f'Service {domain}.{service_name} was not found.')
        previous_state = await self._get_state_for_verification(entity_id)

        payload = {'entity_id': entity_id, **data}
        response = await self.client.post(
            f'/api/services/{domain}/{service_name}',
            json=payload,
            params=self._build_service_call_params(service_description, want_response),
        )
        if response.status_code == 404:
            raise ValueError(f'Service {domain}.{service_name} was not found.')
        response.raise_for_status()
        response_json = response.json()
        if isinstance(response_json, list):
            changed_states = ENTITY_STATE_CATALOG_ADAPTER.validate_python(response_json)
            result = HAServiceCallResult(changed_states=changed_states).to_result()
        else:
            result = HAServiceCallResult.model_validate(response_json).to_result()

        result.verified_state = await self._get_verified_state(
            entity_id=entity_id,
            domain=domain,
            previous_state=previous_state,
            service_name=service_name,
            result=result,
        )
        return result

    async def list_states(self, domain: str | None = None) -> list[EntityState]:
        """Return current entity states, optionally filtered by domain."""
        ha_entities = await self._fetch_state_catalog()
        if domain is not None:
            ha_entities = [ha_entity for ha_entity in ha_entities if ha_entity.domain == domain]
        return [ha_entity.to_state() for ha_entity in ha_entities]

    async def _fetch_state_catalog(self) -> list[HAEntityState]:
        """Fetch and validate the full Home Assistant state catalog."""
        response = await self.client.get('/api/states')
        response.raise_for_status()
        return ENTITY_STATE_CATALOG_ADAPTER.validate_python(response.json())

    async def _get_service_catalog(self, *, refresh: bool = False) -> ServiceCatalog:
        """Fetch the service catalog once and reuse it until refreshed."""
        catalog = self._service_catalog_cache
        if catalog is None or refresh:
            response = await self.client.get('/api/services')
            response.raise_for_status()
            catalog = SERVICE_CATALOG_ADAPTER.validate_python(response.json())
            self._service_catalog_cache = catalog
        return catalog

    async def _get_service_description(
        self,
        domain: str,
        service_name: str,
    ) -> ServiceDescription | None:
        """Look up one service description from the cached service catalog."""
        catalog = await self._get_service_catalog()
        for domain_services in catalog:
            if domain_services.domain == domain:
                return domain_services.get_service_description(service_name)
        return None

    @staticmethod
    def _build_service_call_params(
        service_description: ServiceDescription,
        want_response: bool,
    ) -> dict[str, str] | None:
        """Build query parameters for service calls that need response payloads."""
        if service_description.response is None:
            return None

        if not service_description.response.optional or want_response:
            return {'return_response': ''}

        return None

    async def _get_verified_state(
        self,
        *,
        entity_id: str,
        domain: str,
        service_name: str,
        result: ServiceCallResult,
        previous_state: EntityState | None = None,
    ) -> EntityState | None:
        """Resolve a concrete post-call state for the target entity when possible."""
        for changed_state in result.changed_states:
            if changed_state.entity_id == entity_id:
                return changed_state

        if result.changed_states or result.service_response is not None:
            return None

        latest_state: EntityState | None = None
        for attempt in range(self._verification_poll_attempts):
            latest_state = await self._get_state_for_verification(entity_id)
            if latest_state is not None and self._state_changed(previous_state, latest_state):
                return latest_state
            if attempt < self._verification_poll_attempts - 1:
                await asyncio.sleep(self._verification_poll_interval)

        if latest_state is not None:
            return latest_state

        raise RuntimeError(
            f'Service {domain}.{service_name} executed but state for {entity_id!r} could not be verified.'
        )

    async def _get_state_for_verification(self, entity_id: str) -> EntityState | None:
        """Best-effort state fetch used to verify a service call outcome."""
        try:
            return await self.get_state(entity_id)
        except ValueError:
            return None

    @staticmethod
    def _state_changed(previous_state: EntityState | None, current_state: EntityState) -> bool:
        """Return whether a polled state differs from the pre-call snapshot."""
        if previous_state is None:
            return True
        return current_state.state != previous_state.state or current_state.last_updated != previous_state.last_updated
