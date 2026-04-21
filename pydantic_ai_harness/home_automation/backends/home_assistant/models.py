"""Home Assistant REST response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from pydantic_ai_harness.home_automation.backends._base_backend import (
    Entity,
    EntityState,
    Service,
    ServiceArgument,
    ServiceCallResult,
)


class ServiceResponseInfo(BaseModel):
    """Response metadata for a service."""

    optional: bool


class FieldFilter(BaseModel):
    """Visibility/applicability filters for one service field."""

    supported_features: list[Any] = Field(default_factory=list)
    attribute: dict[str, list[Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra='allow')


class NumberSelectorConfig(BaseModel):
    """Selector config for numeric inputs."""

    min: float | int | None = None
    max: float | int | None = None
    step: float | int | str | None = None
    unit_of_measurement: str | None = None

    model_config = ConfigDict(extra='allow')


class SelectSelectorConfig(BaseModel):
    """Selector config for select inputs."""

    options: list[Any] = Field(default_factory=list)
    multiple: bool | None = None
    translation_key: str | None = None

    model_config = ConfigDict(extra='allow')


class ColorTempSelectorConfig(BaseModel):
    """Selector config for color temperature inputs."""

    unit: str | None = None
    min: int | None = None
    max: int | None = None

    model_config = ConfigDict(extra='allow')


class HAEntitySelectorConfig(BaseModel):
    """Selector config for entity targets/fields."""

    domain: str | list[str] | None = None
    integration: str | list[str] | None = None
    device_class: str | list[str] | None = None
    supported_features: list[Any] = Field(default_factory=list)

    model_config = ConfigDict(extra='allow')


class DeviceSelectorConfig(BaseModel):
    """Selector config for device targets/fields."""

    integration: str | list[str] | None = None
    manufacturer: str | list[str] | None = None
    model: str | list[str] | None = None

    model_config = ConfigDict(extra='allow')


class AreaSelectorConfig(BaseModel):
    """Selector config for area targets/fields."""

    model_config = ConfigDict(extra='allow')


class Selector(BaseModel):
    """A Home Assistant selector declaration."""

    number: NumberSelectorConfig | None = None
    select: SelectSelectorConfig | None = None
    color_temp: ColorTempSelectorConfig | None = None
    color_rgb: dict[str, Any] | None = None
    boolean: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    object: dict[str, Any] | None = None
    entity: HAEntitySelectorConfig | None = None
    device: DeviceSelectorConfig | None = None
    area: AreaSelectorConfig | None = None
    constant: dict[str, Any] | None = None

    model_config = ConfigDict(extra='allow')


class ServiceFieldDescription(BaseModel):
    """Description for one service field, including nested field groups."""

    name: str | None = None
    description: str | None = None
    required: bool | None = None
    advanced: bool | None = None
    example: Any = None
    default: Any = None
    selector: Selector | None = None
    filter: FieldFilter | None = None
    fields: dict[str, ServiceFieldDescription] = Field(default_factory=dict)
    collapsed: bool | None = None

    model_config = ConfigDict(extra='allow')

    def to_service_argument(self, field_name: str) -> ServiceArgument:
        """Convert this Home Assistant field description to a service argument."""
        selector = self.selector

        if selector is not None and selector.number is not None:
            return ServiceArgument(
                name=field_name,
                value_type='number',
                required=bool(self.required),
                description=self.description,
                minimum=selector.number.min,
                maximum=selector.number.max,
            )

        if selector is not None and selector.select is not None:
            return ServiceArgument(
                name=field_name,
                value_type='string',
                required=bool(self.required),
                description=self.description,
                enum=tuple(str(option) for option in selector.select.options),
            )

        if selector is not None and selector.color_temp is not None:
            return ServiceArgument(
                name=field_name,
                value_type='number',
                required=bool(self.required),
                description=self.description,
                minimum=selector.color_temp.min,
                maximum=selector.color_temp.max,
            )

        if selector is not None and selector.color_rgb is not None:
            return ServiceArgument(
                name=field_name,
                value_type='object',
                required=bool(self.required),
                description=self.description,
            )

        if selector is not None and selector.boolean is not None:
            return ServiceArgument(
                name=field_name,
                value_type='boolean',
                required=bool(self.required),
                description=self.description,
            )

        if selector is not None and selector.constant is not None:
            value_type = 'boolean' if isinstance(selector.constant.get('value'), bool) else 'object'
            return ServiceArgument(
                name=field_name,
                value_type=value_type,
                required=bool(self.required),
                description=self.description,
            )

        if selector is not None and selector.text is not None:
            return ServiceArgument(
                name=field_name,
                value_type='string',
                required=bool(self.required),
                description=self.description,
            )

        if selector is not None and selector.object is not None:
            return ServiceArgument(
                name=field_name,
                value_type='object',
                required=bool(self.required),
                description=self.description,
            )

        if selector is not None and selector.entity is not None:
            return ServiceArgument(
                name=field_name,
                value_type='entity_id',
                required=bool(self.required),
                description=self.description,
            )

        if selector is not None and selector.device is not None:
            return ServiceArgument(
                name=field_name,
                value_type='device_id',
                required=bool(self.required),
                description=self.description,
            )

        if selector is not None and selector.area is not None:
            return ServiceArgument(
                name=field_name,
                value_type='area_id',
                required=bool(self.required),
                description=self.description,
            )

        return ServiceArgument(
            name=field_name,
            value_type='string',
            required=bool(self.required),
            description=self.description,
        )


class ServiceTarget(BaseModel):
    """Target metadata for a service."""

    entity: HAEntitySelectorConfig | list[HAEntitySelectorConfig] | None = None
    device: DeviceSelectorConfig | list[DeviceSelectorConfig] | None = None
    area: AreaSelectorConfig | list[AreaSelectorConfig] | None = None

    model_config = ConfigDict(extra='allow')


class ServiceDescription(BaseModel):
    """One Home Assistant service description."""

    name: str | None = None
    description: str | None = None
    target: ServiceTarget | None = None
    fields: dict[str, ServiceFieldDescription] = Field(default_factory=dict)
    description_placeholders: dict[str, str] = Field(default_factory=dict)
    response: ServiceResponseInfo | None = None

    model_config = ConfigDict(extra='allow')

    def to_service(self, domain: str, service_name: str) -> Service:
        """Convert this Home Assistant service description to a service."""
        args = tuple(self._iter_service_arguments())
        return Service(domain=domain, name=service_name, args=args)

    def _iter_service_arguments(self) -> list[ServiceArgument]:
        """Return top-level arguments plus one level of grouped advanced fields."""
        args: list[ServiceArgument] = []
        for field_name, field in self.fields.items():
            if field_name == 'advanced_fields':
                args.extend(
                    nested_field.to_service_argument(nested_field_name)
                    for nested_field_name, nested_field in field.fields.items()
                )
                continue
            args.append(field.to_service_argument(field_name))
        return args


class DomainServices(BaseModel):
    """Services grouped by Home Assistant domain."""

    domain: str
    services: dict[str, ServiceDescription]

    def to_services(self) -> list[Service]:
        """Convert all services in this domain to service models."""
        return [
            service_description.to_service(self.domain, service_name)
            for service_name, service_description in self.services.items()
        ]

    def get_service_description(self, service_name: str) -> ServiceDescription | None:
        """Return the Home Assistant description for one service."""
        return self.services.get(service_name)


ServiceFieldDescription.model_rebuild()
ServiceCatalog = list[DomainServices]
SERVICE_CATALOG_ADAPTER = TypeAdapter(ServiceCatalog)


class HAEntityAttributes(BaseModel):
    """Attributes returned for a Home Assistant entity state."""

    friendly_name: str | None = None
    supported_features: int | None = None
    device_class: str | None = None
    unit_of_measurement: str | None = None
    state_class: str | None = None
    icon: str | None = None
    entity_picture: str | None = None

    model_config = ConfigDict(extra='allow')


class HAEntityContext(BaseModel):
    """Context metadata returned with a Home Assistant entity state."""

    id: str
    parent_id: str | None = None
    user_id: str | None = None

    model_config = ConfigDict(extra='allow')


class HAEntityState(BaseModel):
    """One state object returned by Home Assistant."""

    entity_id: str
    state: str
    attributes: HAEntityAttributes = Field(default_factory=HAEntityAttributes)
    last_changed: str
    last_reported: str | None = None
    last_updated: str | None = None
    context: HAEntityContext | None = None

    def to_entity(self) -> Entity:
        """Convert this Home Assistant state to an entity summary."""
        domain, _, _object_id = self.entity_id.partition('.')
        return Entity(
            entity_id=self.entity_id,
            domain=domain,
            name=self.attributes.friendly_name,
            current_state=self.state,
        )

    def to_state(self) -> EntityState:
        """Convert this Home Assistant state to a lightweight state model."""
        return EntityState(
            entity_id=self.entity_id,
            state=self.state,
            last_updated=self.last_updated or self.last_changed,
        )

    @property
    def domain(self) -> str:
        """Return the entity domain prefix."""
        return self.entity_id.split('.', 1)[0]

    model_config = ConfigDict(extra='allow')


EntityCatalog = list[HAEntityState]
ENTITY_STATE_CATALOG_ADAPTER = TypeAdapter(EntityCatalog)


def _empty_ha_entity_states() -> list[HAEntityState]:
    return []


class HAServiceCallResult(BaseModel):
    """Response object returned by a Home Assistant service call."""

    changed_states: list[HAEntityState] = Field(default_factory=_empty_ha_entity_states)
    service_response: dict[str, Any] = Field(default_factory=dict)

    def to_result(self) -> ServiceCallResult:
        """Convert this Home Assistant service response to a call result."""
        return ServiceCallResult(
            changed_states=[state.to_state() for state in self.changed_states],
            service_response=self.service_response or None,
        )
