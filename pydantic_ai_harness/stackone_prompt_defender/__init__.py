"""Prompt injection defense for tool results, using defender by StackOne."""

from pydantic_ai_harness.stackone_prompt_defender._capability import OnDetection, StackOnePromptDefender

__all__ = ['OnDetection', 'StackOnePromptDefender']
