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
- [GOTCHA] Rich Panel titles cannot be updated in-place without replacing the entire Panel; moving dynamic content into the Panel body enables smoother updates
- [GOTCHA] Implementation was already complete - on_key and _move_highlight methods already existed with correct logic matching Command Palette pattern. Only test coverage was missing.
- [GOTCHA] When all widgets in a column (sidebar or main) are hidden, the empty Vertical container was still being created with width: 1fr or width: 44, consuming layout space unnecessarily.
- [GOTCHA] Test _DummyApp._run_backend_command was missing bypass_local parameter - fixed pre-existing test bug discovered during validation.
- [GOTCHA] Sandboxed uv execution required setting UV_CACHE_DIR=.uv-cache because ~/.cache/uv is not writable in this environment.
- [GOTCHA] A few Ralph command tests used triple-quoted multiline literals while command builders/emitters intentionally output single-line command strings.
