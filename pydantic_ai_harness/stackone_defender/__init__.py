"""Prompt injection defense for tool results, using defender by StackOne."""

from pydantic_ai_harness.stackone_defender._capability import OnDetection, StackOneDefender

__all__ = ['OnDetection', 'StackOneDefender']
