"""
token_tracker.py — Simple accumulator for LLM token usage across calls.

Tracks input, output, and thinking tokens per labeled call. Provides
per-entry and overall summaries for evaluation cost reporting.
"""


class TokenTracker:
    """Accumulates token usage across multiple LLM calls."""

    def __init__(self):
        self._entries = []  # list of {label, input, output, thinking, total}

    def track(self, label, usage_metadata):
        """Record token usage from a single LLM call.

        Args:
            label: String identifier, e.g. "R01_report_generate".
            usage_metadata: Dict from AIMessage.response_metadata["usage_metadata"]
                            or similar. Expected keys vary by provider; we
                            extract what's available.
        """
        if usage_metadata is None:
            return

        input_tokens = (
            usage_metadata.get("input_tokens")
            or usage_metadata.get("prompt_token_count")
            or usage_metadata.get("prompt_tokens", 0)
        )
        output_tokens = (
            usage_metadata.get("output_tokens")
            or usage_metadata.get("candidates_token_count")
            or usage_metadata.get("completion_tokens", 0)
        )
        thinking_tokens = (
            usage_metadata.get("thoughts_token_count", 0)
        )
        total = input_tokens + output_tokens + thinking_tokens

        self._entries.append({
            "label": label,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total,
        })

    def get_entries_for_prefix(self, prefix):
        """Return all entries whose label starts with prefix."""
        return [e for e in self._entries if e["label"].startswith(prefix)]

    def sum_for_prefix(self, prefix):
        """Sum token counts for all entries matching prefix."""
        entries = self.get_entries_for_prefix(prefix)
        return {
            "input_tokens": sum(e["input_tokens"] for e in entries),
            "output_tokens": sum(e["output_tokens"] for e in entries),
            "thinking_tokens": sum(e["thinking_tokens"] for e in entries),
            "total_tokens": sum(e["total_tokens"] for e in entries),
            "num_calls": len(entries),
        }

    def get_summary(self):
        """Return overall token totals across all tracked calls."""
        return {
            "input_tokens": sum(e["input_tokens"] for e in self._entries),
            "output_tokens": sum(e["output_tokens"] for e in self._entries),
            "thinking_tokens": sum(e["thinking_tokens"] for e in self._entries),
            "total_tokens": sum(e["total_tokens"] for e in self._entries),
            "num_calls": len(self._entries),
        }

    def reset(self):
        """Clear all tracked entries."""
        self._entries = []


# Module-level singleton
tracker = TokenTracker()
