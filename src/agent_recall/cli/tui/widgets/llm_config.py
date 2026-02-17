from __future__ import annotations

from dataclasses import dataclass

from rich import box
from rich.panel import Panel
from rich.table import Table

from agent_recall.storage.models import LLMConfig


@dataclass(frozen=True)
class LLMConfigWidget:
    llm_config: LLMConfig
    api_key_set_display: str
    view: str

    def render(self) -> Panel:
        table = Table(
            expand=True,
            box=box.SIMPLE,
            pad_edge=False,
            collapse_padding=True,
        )
        if self.view == "all":
            table.add_column("Setting", style="table_header", width=12, no_wrap=True)
            table.add_column("Value", overflow="fold")
            table.add_row("Provider", self.llm_config.provider)
            table.add_row("Model", self.llm_config.model)
            table.add_row("Temperature", str(self.llm_config.temperature))
            table.add_row("Max tokens", str(self.llm_config.max_tokens))
            table.add_row("API Key Set", self.api_key_set_display)
        else:
            table.add_column("Provider", style="table_header")
            table.add_column("Model", style="table_header", overflow="fold")
            table.add_column("Base URL", style="table_header", overflow="fold")
            table.add_column("Temperature", style="table_header")
            table.add_column("Max tokens", style="table_header")
            table.add_column("API Key Set", style="table_header")
            table.add_row(
                self.llm_config.provider,
                self.llm_config.model,
                self.llm_config.base_url or "default",
                str(self.llm_config.temperature),
                str(self.llm_config.max_tokens),
                self.api_key_set_display,
            )
        return Panel(table, title="LLM Configuration", border_style="accent")
