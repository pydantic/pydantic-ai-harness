"""Typed policy for who may invoke a Slack-hosted agent."""

from __future__ import annotations

from dataclasses import dataclass


def _normalized_user_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('SlackAccess.users() needs at least one non-empty Slack user ID')
    return value.strip()


@dataclass(frozen=True, slots=True, init=False)
class SlackAccess:
    """Choose whether a Slack app serves selected users or the whole workspace."""

    _allowed_user_ids: frozenset[str] | None

    @classmethod
    def users(cls, *user_ids: str) -> SlackAccess:
        """Allow only the listed Slack user IDs."""
        if not user_ids:
            raise ValueError('SlackAccess.users() needs at least one non-empty Slack user ID')
        access = object.__new__(cls)
        object.__setattr__(access, '_allowed_user_ids', frozenset(_normalized_user_id(user_id) for user_id in user_ids))
        return access

    @classmethod
    def workspace(cls) -> SlackAccess:
        """Allow every member who can reach the installed Slack app."""
        access = object.__new__(cls)
        object.__setattr__(access, '_allowed_user_ids', None)
        return access

    def allows(self, user_id: str) -> bool:
        """Whether `user_id` may invoke the agent."""
        return self._allowed_user_ids is None or user_id in self._allowed_user_ids

    @property
    def allowed_user_ids(self) -> frozenset[str] | None:
        """Allowed user IDs, or `None` when the whole workspace is allowed."""
        return self._allowed_user_ids
