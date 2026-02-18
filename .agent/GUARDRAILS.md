# Guardrails

- [GOTCHA] raise exception
- [GOTCHA] Dashboard render context needed a Ralph max-iterations value; read it from config to keep the panel accurate.
- [GOTCHA] OptionList entries need stable ids and centralized list wiring to avoid conflicting view updates.
- [GOTCHA] PRD passes flag was out of sync with actual implementation status. Code was complete but PRD showed unpassed.
- [GOTCHA] Textual widget methods that use query_one() must handle NoMatches exceptions when widget is not yet mounted (e.g., in unit tests). Added defensive try/except blocks to _update_button_state, mark_sync_complete, and update_sources methods.
- [GOTCHA] The TUI refresh path caches dashboard layout; layout changes must reset the signature to force a remount.
- [GOTCHA] Timeline detail depends on existing iteration reports; handle empty stores with a friendly placeholder.
- [GOTCHA] Widget already existed from prior work, only needed wiring into dashboard views
- [GOTCHA] Implementation was complete but PRD not updated - similar to AR-1005 and AR-1013 cases
