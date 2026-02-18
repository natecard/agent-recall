# Style Guide

- [PATTERN] Building build_sources_data() helper function to extract source data preparation logic from dashboard.py made it reusable for both the existing read-only widget and the new interactive widget. Using existing _run_source_sync infrastructure avoided duplicating sync logic.
- [PATTERN] Followed existing widget pattern (RalphStatusWidget) for dataclass + render() method; tests needed DashboardPanels updates to include new field
- [PATTERN] Reuse existing dashboard panel pipeline for new widget integration; leverage existing test patterns for context-aware dashboard testing.
- [PATTERN] Reuse existing modal card styles and apply settings through FileStorage-backed TUI config updates.
- [PATTERN] Reuse existing panel builders and Rich Syntax rendering to deliver a readable detail view without new Textual widgets.
- [PATTERN] Reuse the activity result list to route sync actions through worker execution with existing command plumbing.
- [PATTERN] Reused the existing dashboard panel builder and view routing patterns to add a new view with minimal impact.
- [PATTERN] Used _build_slash_command_map() as single source of truth for hyphenated commands, then routed to handle_palette_action() which dispatches correctly to palette_router
- [PATTERN] Used Rich markup for heading styling instead of CSS rules since all headings share [bold accent] format
- [PATTERN] Checked existing tests first before implementing - discovered functionality was already complete via bash script.
- [PATTERN] Reused existing LLMProvider interface for CodingCLIProvider to maintain consistent abstraction over backend selection.
- [PATTERN] Checked existing implementation before starting new work - discovered full feature already built with tree view, content viewer, syntax highlighting, and scrolling support
- [PATTERN] Small targeted CSS edits achieved full acceptance criteria coverage without breaking visual hierarchy.
- [PATTERN] Small targeted edits to app.py and dashboard.py achieved full acceptance criteria coverage
- [PATTERN] Small targeted edit to remove capping logic and truncation indicator block
