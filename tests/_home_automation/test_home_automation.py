from collections.abc import Callable

import httpx
import pytest
from pydantic_ai import FunctionToolset

import pydantic_ai_harness
from pydantic_ai_harness.home_automation import HomeAssistantBackend, HomeAutomation, HomeAutomationToolset
from pydantic_ai_harness.home_automation.backends import Entity, EntityState, Service, ServiceCallResult
from pydantic_ai_harness.home_automation.backends.home_assistant._backend import HomeAssistantBackend as ExportedBackend
from pydantic_ai_harness.home_automation.backends.home_assistant.models import (
    HAEntityState,
    HAServiceCallResult,
    ServiceDescription,
    ServiceFieldDescription,
)

pytestmark = pytest.mark.anyio

RouteHandler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class MockHomeAssistant:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._routes: dict[tuple[str, str], RouteHandler] = {}

    def route(self, method: str, path: str, handler: RouteHandler) -> None:
        self._routes[(method.upper(), path)] = handler

    def json(self, method: str, path: str, payload: object, *, status_code: int = 200) -> None:
        self.route(method, path, lambda request: httpx.Response(status_code, json=payload))

    def count(self, method: str, path: str) -> int:
        return sum(1 for request in self.requests if request.method == method.upper() and request.url.path == path)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        handler = self._routes.get((request.method, request.url.path))
        if handler is None:
            raise AssertionError(f'Unexpected request: {request.method} {request.url}')  # pragma: no cover
        return handler(request)

    def backend(
        self,
        *,
        verification_poll_attempts: int = 3,
        verification_poll_interval: float = 0.5,
    ) -> HomeAssistantBackend:
        client = httpx.AsyncClient(base_url='http://example.test', transport=httpx.MockTransport(self.handle))
        return HomeAssistantBackend(
            url='http://example.test',
            token='token',
            client=client,
            verification_poll_attempts=verification_poll_attempts,
            verification_poll_interval=verification_poll_interval,
        )


class StubBackend:  # pragma: no cover - protocol test double
    async def list_services(self, domain: str | None = None) -> list[Service]:
        return []

    async def list_entities(self, domain: str | None = None) -> list[Entity]:
        return []

    async def get_state(self, entity_id: str) -> EntityState:
        return EntityState(entity_id=entity_id, last_updated='2026-04-19T12:00:00+00:00', state='on')

    async def list_states(self, domain: str | None = None) -> list[EntityState]:
        return []

    async def call_service(
        self,
        domain: str,
        entity_id: str,
        service_name: str,
        *,
        want_response: bool = False,
        **data: object,
    ) -> ServiceCallResult:
        return ServiceCallResult()


def _services_payload() -> list[dict[str, object]]:
    return [
        {
            'domain': 'light',
            'services': {
                'turn_on': {
                    'name': 'Turn on',
                    'description': 'Turn the light on.',
                    'target': {
                        'entity': {
                            'domain': 'light',
                        }
                    },
                    'fields': {
                        'brightness_pct': {
                            'name': 'Brightness',
                            'description': 'Brightness in percent.',
                            'required': False,
                            'selector': {
                                'number': {
                                    'min': 0,
                                    'max': 100,
                                    'unit_of_measurement': '%',
                                }
                            },
                            'filter': {
                                'supported_features': ['light.LightEntityFeature.EFFECT'],
                                'attribute': {
                                    'supported_color_modes': [
                                        'light.ColorMode.BRIGHTNESS',
                                        'light.ColorMode.HS',
                                    ]
                                },
                            },
                        },
                        'advanced_fields': {
                            'collapsed': True,
                            'fields': {
                                'transition': {
                                    'name': 'Transition',
                                    'selector': {
                                        'number': {
                                            'min': 0,
                                            'max': 300,
                                            'unit_of_measurement': 'seconds',
                                        }
                                    },
                                }
                            },
                        },
                    },
                    'description_placeholders': {'docs_url': 'https://example.test/docs'},
                },
                'turn_off': {
                    'name': 'Turn off',
                    'description': 'Turn the light off.',
                    'target': {
                        'entity': {
                            'domain': 'light',
                        }
                    },
                    'fields': {},
                    'response': {'optional': True},
                },
            },
        },
        {
            'domain': 'climate',
            'services': {
                'set_temperature': {
                    'name': 'Set temperature',
                    'description': 'Set target temperature.',
                    'target': {
                        'entity': {
                            'domain': 'climate',
                        }
                    },
                    'fields': {
                        'temperature': {
                            'required': True,
                            'selector': {
                                'number': {
                                    'min': 5,
                                    'max': 35,
                                }
                            },
                        }
                    },
                    'response': {'optional': False},
                }
            },
        },
    ]


def _states_payload() -> list[dict[str, object]]:
    return [
        {
            'entity_id': 'light.bedroom_light',
            'state': 'on',
            'attributes': {
                'friendly_name': 'Bedroom Light',
            },
            'last_changed': '2026-04-19T12:00:00+00:00',
            'last_reported': '2026-04-19T12:00:00+00:00',
            'last_updated': '2026-04-19T12:00:00+00:00',
            'context': {
                'id': 'abc',
                'parent_id': None,
                'user_id': None,
            },
        },
        {
            'entity_id': 'switch.coffee_machine',
            'state': 'off',
            'attributes': {
                'friendly_name': 'Coffee Machine',
            },
            'last_changed': '2026-04-19T12:05:00+00:00',
            'last_reported': '2026-04-19T12:05:00+00:00',
            'last_updated': '2026-04-19T12:05:00+00:00',
            'context': {
                'id': 'def',
                'parent_id': None,
                'user_id': None,
            },
        },
    ]


class TestHomeAutomation:
    def test_exposes_home_automation_toolset(self) -> None:
        capability = HomeAutomation(backend=StubBackend())

        toolset = capability.get_toolset()

        assert isinstance(toolset, FunctionToolset)
        assert isinstance(toolset, HomeAutomationToolset)
        assert toolset.backend is capability.backend
        assert set(toolset.tools) == {'list_services', 'list_entities', 'get_state', 'list_states', 'call_service'}

    def test_instructions_describe_entities_services_and_verification(self) -> None:
        instructions = HomeAutomation(backend=StubBackend()).get_instructions()

        assert 'smart home entities' in instructions
        assert 'list_services' in instructions
        assert 'verified_state' in instructions
        assert 'follow-up state reads' in instructions

    def test_package_exports_public_capability_only_at_root(self) -> None:
        assert pydantic_ai_harness.HomeAutomation is HomeAutomation
        assert HomeAssistantBackend is ExportedBackend
        with pytest.raises(AttributeError, match='HomeAssistantBackend'):
            getattr(pydantic_ai_harness, 'HomeAssistantBackend')
        with pytest.raises(AttributeError, match='missing'):
            getattr(pydantic_ai_harness, 'missing')


class TestHomeAssistantServices:
    async def test_list_services_parses_catalog_and_flattens_advanced_fields(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())

        services = await server.backend().list_services()

        assert [(service.domain, service.name) for service in services] == [
            ('light', 'turn_on'),
            ('light', 'turn_off'),
            ('climate', 'set_temperature'),
        ]
        light_turn_on = services[0]
        assert [arg.name for arg in light_turn_on.args] == ['brightness_pct', 'transition']
        assert light_turn_on.args[0].value_type == 'number'
        assert light_turn_on.args[0].minimum == 0
        assert light_turn_on.args[0].maximum == 100
        assert light_turn_on.args[1].value_type == 'number'
        assert light_turn_on.args[1].maximum == 300

    async def test_list_services_can_filter_by_domain(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())

        light_services = await server.backend().list_services(domain='light')

        assert [(service.domain, service.name) for service in light_services] == [
            ('light', 'turn_on'),
            ('light', 'turn_off'),
        ]

    async def test_list_services_maps_required_numeric_arguments(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())

        services = await server.backend().list_services(domain='climate')

        assert len(services) == 1
        service = services[0]
        assert service.domain == 'climate'
        assert service.name == 'set_temperature'
        assert len(service.args) == 1
        assert service.args[0].name == 'temperature'
        assert service.args[0].required is True
        assert service.args[0].minimum == 5
        assert service.args[0].maximum == 35

    async def test_list_services_returns_empty_for_missing_domain(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())

        services = await server.backend().list_services(domain='fan')

        assert services == []

    async def test_service_catalog_can_refresh_cache(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())
        backend = server.backend()

        await backend._get_service_catalog()
        await backend._get_service_catalog(refresh=True)

        assert server.count('GET', '/api/services') == 2

    async def test_get_service_description_returns_none_for_missing_domain(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())

        assert await server.backend()._get_service_description('fan', 'turn_on') is None


class TestHomeAssistantStates:
    async def test_list_states_can_filter_by_domain(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/states', _states_payload())

        states = await server.backend().list_states(domain='light')

        assert len(states) == 1
        assert states[0].entity_id == 'light.bedroom_light'
        assert states[0].state == 'on'
        assert states[0].last_updated == '2026-04-19T12:00:00+00:00'

    async def test_list_states_returns_all_states(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/states', _states_payload())

        states = await server.backend().list_states()

        assert [state.entity_id for state in states] == ['light.bedroom_light', 'switch.coffee_machine']

    async def test_list_entities_can_filter_by_domain(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/states', _states_payload())

        entities = await server.backend().list_entities(domain='switch')

        assert entities == [
            Entity(
                entity_id='switch.coffee_machine',
                domain='switch',
                name='Coffee Machine',
                current_state='off',
            )
        ]

    async def test_list_entities_returns_all_entities(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/states', _states_payload())

        entities = await server.backend().list_entities()

        assert [entity.entity_id for entity in entities] == ['light.bedroom_light', 'switch.coffee_machine']

    async def test_get_state_returns_one_entity_state(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/states/light.bedroom_light', _states_payload()[0])

        state = await server.backend().get_state('light.bedroom_light')

        assert state.entity_id == 'light.bedroom_light'
        assert state.state == 'on'
        assert state.last_updated == '2026-04-19T12:00:00+00:00'

    async def test_get_state_prefers_home_assistant_last_updated(self) -> None:
        payload = {
            **_states_payload()[0],
            'last_changed': '2026-04-19T12:00:00+00:00',
            'last_updated': '2026-04-19T12:10:00+00:00',
        }
        server = MockHomeAssistant()
        server.json('GET', '/api/states/light.bedroom_light', payload)

        state = await server.backend().get_state('light.bedroom_light')

        assert state.last_updated == '2026-04-19T12:10:00+00:00'

    async def test_get_state_raises_for_missing_entity(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/states/light.missing', {'message': 'Entity not found'}, status_code=404)

        with pytest.raises(ValueError, match=r"Entity 'light\.missing' was not found\."):
            await server.backend().get_state('light.missing')


class TestHomeAssistantClient:
    async def test_backend_can_close_owned_client(self) -> None:
        backend = HomeAssistantBackend(url='http://example.test', token='token')

        await backend.aclose()

        assert backend.client.is_closed

    async def test_backend_does_not_close_injected_client(self) -> None:
        client = httpx.AsyncClient(base_url='http://example.test')
        backend = HomeAssistantBackend(url='http://example.test', token='token', client=client)

        await backend.aclose()

        assert not client.is_closed
        await client.aclose()

    def test_backend_rejects_negative_verification_poll_attempts(self) -> None:
        with pytest.raises(ValueError, match='verification_poll_attempts'):
            HomeAssistantBackend(
                url='http://example.test',
                token='token',
                verification_poll_attempts=-1,
            )

    def test_backend_rejects_negative_verification_poll_interval(self) -> None:
        with pytest.raises(ValueError, match='verification_poll_interval'):
            HomeAssistantBackend(
                url='http://example.test',
                token='token',
                verification_poll_interval=-0.1,
            )


class TestHomeAssistantCallService:
    async def test_call_service_uses_cached_catalog_and_normalizes_changed_states(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())
        server.json('GET', '/api/states/light.bedroom_light', _states_payload()[0])

        def turn_off(request: httpx.Request) -> httpx.Response:
            assert 'return_response' not in request.url.params
            return httpx.Response(200, json=_states_payload()[:1])

        server.route('POST', '/api/services/light/turn_off', turn_off)
        backend = server.backend()

        await backend.list_services(domain='light')
        result = await backend.call_service(
            domain='light',
            entity_id='light.bedroom_light',
            service_name='turn_off',
        )

        assert server.count('GET', '/api/services') == 1
        assert len(result.changed_states) == 1
        assert result.changed_states[0].entity_id == 'light.bedroom_light'
        assert result.service_response is None
        assert result.verified_state == result.changed_states[0]

    async def test_call_service_forces_return_response_for_required_services(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())
        server.json('GET', '/api/states/climate.home', {'message': 'Entity not found'}, status_code=404)

        def set_temperature(request: httpx.Request) -> httpx.Response:
            assert 'return_response' in request.url.params
            return httpx.Response(
                200,
                json={
                    'changed_states': _states_payload()[:1],
                    'service_response': {
                        'weather.home': {
                            'forecast': [{'condition': 'sunny'}],
                        }
                    },
                },
            )

        server.route('POST', '/api/services/climate/set_temperature', set_temperature)

        result = await server.backend().call_service(
            domain='climate',
            entity_id='climate.home',
            service_name='set_temperature',
            temperature=21,
        )

        assert len(result.changed_states) == 1
        assert result.service_response == {
            'weather.home': {
                'forecast': [{'condition': 'sunny'}],
            }
        }
        assert result.verified_state is None

    async def test_call_service_can_request_optional_service_response(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())
        server.json('GET', '/api/states/light.bedroom_light', _states_payload()[0])

        def turn_off(request: httpx.Request) -> httpx.Response:
            assert 'return_response' in request.url.params
            return httpx.Response(
                200,
                json={
                    'changed_states': _states_payload()[:1],
                    'service_response': {'light.bedroom_light': {'acknowledged': True}},
                },
            )

        server.route('POST', '/api/services/light/turn_off', turn_off)

        result = await server.backend().call_service(
            domain='light',
            entity_id='light.bedroom_light',
            service_name='turn_off',
            want_response=True,
        )

        assert result.service_response == {'light.bedroom_light': {'acknowledged': True}}
        assert result.verified_state == result.changed_states[0]

    async def test_call_service_polls_for_state_when_service_returns_no_verification_payload(self) -> None:
        state_requests = 0
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())
        server.json('POST', '/api/services/light/turn_off', [])

        def get_state(request: httpx.Request) -> httpx.Response:
            nonlocal state_requests
            state_requests += 1
            if state_requests < 3:
                return httpx.Response(
                    200,
                    json={
                        **_states_payload()[0],
                        'state': 'off',
                        'last_changed': '2026-04-19T12:00:00+00:00',
                        'last_updated': '2026-04-19T12:00:00+00:00',
                    },
                )
            return httpx.Response(200, json=_states_payload()[0])

        server.route('GET', '/api/states/light.bedroom_light', get_state)

        result = await server.backend(verification_poll_interval=0.0).call_service(
            domain='light',
            entity_id='light.bedroom_light',
            service_name='turn_off',
        )

        assert result.changed_states == []
        assert result.service_response is None
        assert result.verified_state is not None
        assert result.verified_state.entity_id == 'light.bedroom_light'
        assert result.verified_state.state == 'on'
        assert state_requests == 3

    async def test_call_service_raises_for_unknown_service(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())

        with pytest.raises(ValueError, match='Service light.missing was not found.'):
            await server.backend().call_service(
                domain='light',
                entity_id='light.bedroom_light',
                service_name='missing',
            )

    async def test_call_service_raises_when_home_assistant_reports_missing_service(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/services', _services_payload())
        server.json('GET', '/api/states/light.bedroom_light', _states_payload()[0])
        server.json('POST', '/api/services/light/turn_off', {'message': 'not found'}, status_code=404)

        with pytest.raises(ValueError, match='Service light.turn_off was not found.'):
            await server.backend().call_service(
                domain='light',
                entity_id='light.bedroom_light',
                service_name='turn_off',
            )

    def test_build_service_call_params_for_services_without_response_data(self) -> None:
        service_description = ServiceDescription.model_validate({})

        assert HomeAssistantBackend._build_service_call_params(service_description, want_response=False) is None

    async def test_get_verified_state_returns_latest_state_when_state_does_not_change(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/states/light.bedroom_light', _states_payload()[0])
        previous_state = EntityState(
            entity_id='light.bedroom_light',
            last_updated='2026-04-19T12:00:00+00:00',
            state='on',
        )

        verified_state = await server.backend(verification_poll_attempts=1)._get_verified_state(
            entity_id='light.bedroom_light',
            domain='light',
            service_name='turn_on',
            result=ServiceCallResult(),
            previous_state=previous_state,
        )

        assert verified_state == previous_state

    async def test_get_verified_state_raises_when_state_cannot_be_verified(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/states/light.missing', {}, status_code=404)

        with pytest.raises(RuntimeError, match='could not be verified'):
            await server.backend(verification_poll_attempts=1)._get_verified_state(
                entity_id='light.missing',
                domain='light',
                service_name='turn_on',
                result=ServiceCallResult(),
            )

    async def test_get_verified_state_raises_cleanly_when_polling_is_disabled(self) -> None:
        server = MockHomeAssistant()
        server.json('GET', '/api/states/light.bedroom_light', _states_payload()[0])

        with pytest.raises(RuntimeError, match='could not be verified'):
            await server.backend(verification_poll_attempts=0)._get_verified_state(
                entity_id='light.bedroom_light',
                domain='light',
                service_name='turn_on',
                result=ServiceCallResult(),
            )

    def test_state_changed_returns_true_without_previous_state(self) -> None:
        assert HomeAssistantBackend._state_changed(
            None,
            EntityState(
                entity_id='light.bedroom_light',
                last_updated='2026-04-19T12:00:00+00:00',
                state='on',
            ),
        )


class TestHomeAssistantModels:
    @pytest.mark.parametrize(
        'field_name, field, expected_type',
        [
            ('select', {'selector': {'select': {'options': ['on', 'off']}}}, 'string'),
            ('color_temp', {'selector': {'color_temp': {'min': 2000, 'max': 6500}}}, 'number'),
            ('color_rgb', {'selector': {'color_rgb': {}}}, 'object'),
            ('boolean', {'selector': {'boolean': {}}}, 'boolean'),
            ('constant', {'selector': {'constant': {'value': True, 'label': 'Enabled'}}}, 'boolean'),
            ('text', {'selector': {'text': {'multiline': False}}}, 'string'),
            ('object', {'selector': {'object': {}}}, 'object'),
            ('entity', {'selector': {'entity': {'domain': 'light'}}}, 'entity_id'),
            ('device', {'selector': {'device': {'integration': 'test'}}}, 'device_id'),
            ('area', {'selector': {'area': {}}}, 'area_id'),
            ('fallback', {}, 'string'),
        ],
    )
    def test_service_field_description_maps_selector_variants(
        self,
        field_name: str,
        field: dict[str, object],
        expected_type: str,
    ) -> None:
        argument = ServiceFieldDescription.model_validate(field).to_service_argument(field_name)

        assert argument.name == field_name
        assert argument.value_type == expected_type

    def test_service_field_description_preserves_select_options(self) -> None:
        field = ServiceFieldDescription.model_validate({'selector': {'select': {'options': ['on', 'off']}}})

        assert field.to_service_argument('mode').enum == ('on', 'off')

    def test_service_field_description_preserves_color_temp_range(self) -> None:
        field = ServiceFieldDescription.model_validate({'selector': {'color_temp': {'min': 2000, 'max': 6500}}})

        argument = field.to_service_argument('color_temp_kelvin')

        assert argument.minimum == 2000
        assert argument.maximum == 6500

    def test_service_call_result_defaults_to_empty_changed_states(self) -> None:
        result = HAServiceCallResult.model_validate({'service_response': {'light.test': {'acknowledged': True}}})

        assert result.changed_states == []

    def test_entity_domain_is_derived_from_entity_id(self) -> None:
        ha_state = HAEntityState.model_validate(_states_payload()[0])

        assert ha_state.domain == 'light'
