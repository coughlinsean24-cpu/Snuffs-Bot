"""
Base Agent Class for AI Trading Agents

Provides common functionality for all agents including:
- Anthropic Claude API integration
- Response parsing and validation
- Token counting and cost tracking
- Error handling and retries
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypedDict
from datetime import datetime
from decimal import Decimal

import anthropic
from loguru import logger

from ...config.settings import get_settings


class AgentResponse(TypedDict):
    """Structured response from an agent"""
    agent_name: str
    decision: str  # 'BULLISH', 'BEARISH', 'NEUTRAL', 'APPROVE', 'REJECT', 'EXECUTE', etc.
    confidence: int  # 0-100
    reasoning: str
    data: Dict[str, Any]
    tokens_used: int
    estimated_cost: float
    timestamp: str


class BaseAgent(ABC):
    """
    Abstract base class for AI trading agents

    All agents share common Anthropic Claude integration and response handling.
    Subclasses implement specific analysis logic via the analyze() method.
    """

    # Cost per 1K tokens (Claude Sonnet 4 pricing)
    COST_PER_1K_INPUT = 0.003  # $0.003 per 1K input tokens
    COST_PER_1K_OUTPUT = 0.015  # $0.015 per 1K output tokens

    def __init__(self, name: str, role_description: str):
        """
        Initialize the agent

        Args:
            name: Agent identifier (e.g., 'market_agent')
            role_description: Brief description of agent's role
        """
        self.name = name
        self.role_description = role_description
        self.settings = get_settings()

        # Initialize Anthropic client (requires API key)
        api_key = self.settings.anthropic_api_key
        if not api_key:
            raise ValueError(
                f"Cannot initialize {name} agent: ANTHROPIC_API_KEY is not set. "
                "Set USE_LOCAL_AI=true to use local XGBoost model instead."
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)

        # Track usage
        self.total_tokens_used = 0
        self.total_cost = 0.0

        logger.info(f"Initialized {name} agent with model {self.settings.claude_model}")

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass

    @abstractmethod
    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build the user prompt from context data

        Args:
            context: Dictionary containing market data, positions, etc.

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the AI response into structured data

        Args:
            response_text: Raw response from Claude

        Returns:
            Parsed response dictionary
        """
        pass

    def count_tokens(self, text: str) -> int:
        """Estimate token count (approximately 4 characters per token for Claude)"""
        return len(text) // 4

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate API cost for a request"""
        input_cost = (input_tokens / 1000) * self.COST_PER_1K_INPUT
        output_cost = (output_tokens / 1000) * self.COST_PER_1K_OUTPUT
        return round(input_cost + output_cost, 6)

    def call_claude(
        self,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> tuple[str, int, int]:
        """
        Make a call to Anthropic Claude API

        Args:
            user_prompt: The user message
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        temp = temperature if temperature is not None else self.settings.claude_temperature
        max_tok = max_tokens if max_tokens is not None else self.settings.claude_max_tokens

        # Estimate input tokens
        input_tokens = self.count_tokens(self.system_prompt + user_prompt)

        logger.debug(f"{self.name}: Calling Claude ({input_tokens} estimated input tokens)")

        try:
            response = self.client.messages.create(
                model=self.settings.claude_model,
                max_tokens=max_tok,
                temperature=temp,
                system=self.system_prompt + "\n\nIMPORTANT: Always respond with valid JSON only. No markdown, no code blocks, just the JSON object.",
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            response_text = response.content[0].text

            # Get actual token usage from response
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Update tracking
            total_tokens = input_tokens + output_tokens
            self.total_tokens_used += total_tokens
            self.total_cost += self.estimate_cost(input_tokens, output_tokens)

            logger.debug(f"{self.name}: Received response ({output_tokens} output tokens)")

            return response_text, input_tokens, output_tokens

        except anthropic.APIError as e:
            logger.error(f"{self.name}: Claude API error: {e}")
            raise

    def analyze(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Perform analysis and return structured response

        Args:
            context: Dictionary containing all relevant data for analysis

        Returns:
            AgentResponse with decision, confidence, and reasoning
        """
        # Build the prompt
        user_prompt = self.build_user_prompt(context)

        # Call Claude
        response_text, input_tokens, output_tokens = self.call_claude(user_prompt)

        # Parse the response
        parsed = self.parse_response(response_text)

        # Calculate cost
        total_tokens = input_tokens + output_tokens
        cost = self.estimate_cost(input_tokens, output_tokens)

        # Build structured response
        return AgentResponse(
            agent_name=self.name,
            decision=parsed.get("decision", "UNKNOWN"),
            confidence=parsed.get("confidence", 0),
            reasoning=parsed.get("reasoning", ""),
            data=parsed,
            tokens_used=total_tokens,
            estimated_cost=cost,
            timestamp=datetime.now().isoformat()
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get cumulative usage statistics"""
        return {
            "agent_name": self.name,
            "total_tokens": self.total_tokens_used,
            "total_cost": round(self.total_cost, 4),
            "model": self.settings.claude_model
        }


def safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Safely parse JSON from AI response

    Handles common issues like markdown code blocks
    """
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        text = "\n".join(lines[1:-1])

    # Also handle case where text ends with ```
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        # Return a default structure
        return {
            "decision": "ERROR",
            "confidence": 0,
            "reasoning": f"Failed to parse response: {text[:200]}",
            "error": str(e)
        }
