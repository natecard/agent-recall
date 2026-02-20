APP_CSS = """

    #root {
        height: 1fr;
        width: 100%;
        align: center top;
    }
    #app_shell {
        width: 96%;
        max-width: 210;
        height: 100%;
    }
    #dashboard {
        height: auto;
        overflow: auto;
        min-height: 0;
    }
    #dashboard.view-all {
        overflow: hidden;
    }
    #dashboard_all_grid {
        layout: horizontal;
        width: 100%;
        height: auto;
        min-height: 0;
    }
    #dashboard_all_sidebar {
        width: 44;
        min-width: 32;
        height: auto;
        min-height: 0;
        margin: 0 1 0 0;
    }
    #dashboard_all_main {
        width: 1fr;
        height: auto;
        min-height: 0;
    }
    #dashboard_all_sidebar #dashboard_knowledge,
    #dashboard_all_sidebar #dashboard_sources,
    #dashboard_all_sidebar #dashboard_settings,
    #dashboard_all_sidebar #dashboard_llm {
        width: 100%;
    }
    #dashboard_all_main #dashboard_timeline {
        height: 1fr;
        min-height: 18;
    }
    #dashboard_all_fullwidth {
        width: 100%;
        height: auto;
        min-height: 0;
    }
    #dashboard_all_fullwidth > * {
        width: 100%;
    }
    #dashboard_ralph {
        width: 100%;
        height: auto;
    }
    #dashboard_overview_row {
        layout: horizontal;
        width: 100%;
        height: auto;
        min-height: 0;
    }
    #dashboard_overview_row #dashboard_knowledge {
        width: 1fr;
        margin: 0 1 0 0;
    }
    #dashboard_overview_row #dashboard_sources {
        width: 1fr;
    }
    .narrow #dashboard_all_grid {
        layout: vertical;
    }
    .narrow #dashboard_all_sidebar {
        width: 100%;
        min-width: 0;
        margin: 0 0 1 0;
    }
    .narrow #dashboard_overview_row {
        layout: vertical;
    }
    .narrow #dashboard_overview_row #dashboard_knowledge {
        margin: 0 0 1 0;
    }
    #terminal_panel {
        height: 1fr;
        min-height: 6;
        overflow: auto;
        display: none;
        border: round $accent;
        background: $panel;
        padding: 0 1;
    }
    #activity {
        height: 1fr;
        min-height: 4;
        overflow: auto;
    }
    #activity_log {
        height: 1fr;
        overflow: auto;
        border: round $accent;
        background: $panel;
        padding: 0 1;
    }
    #activity_result_list {
        height: 1fr;
        overflow: auto;
        display: none;
    }
    #palette_overlay, #modal_overlay {
        align: center middle;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.30);
    }
    #palette_card {
        width: 90%;
        max-width: 200;
        height: auto;
        max-height: 78%;
        padding: 0 2;
        background: $panel;
        border: none;
        overflow: auto;
    }
    #modal_card {
        width: 90%;
        max-width: 200;
        height: auto;
        max-height: 82%;
        padding: 0 1;
        background: $panel;
        border: round $accent;
        overflow: auto;
    }
    .layout_modal {
        max-width: 72;
    }
    .modal_wide {
        width: 90%;
        max-width: 200;
        overflow: hidden;
    }
    #sessions_view_list {
        height: 1fr;
        margin-bottom: 1;
    }
    .modal_compact {
        width: 52%;
        max-width: 58;
        padding: 0 1;
    }
    .field_row_compact {
        margin: 0 0 0 0;
    }
    #model_config_columns {
        layout: horizontal;
        width: 100%;
        height: auto;
        min-height: 0;
    }
    .modal_column {
        width: 1fr;
        min-width: 0;
        height: auto;
        padding: 0 1 0 0;
    }
    .modal_column:last-child {
        padding: 0 0 0 1;
    }
    #iteration_detail_content {
        height: 1fr;
        overflow: auto;
        background: $panel;
        border: round $accent;
        padding: 0 1;
    }
    #palette_header {
        layout: horizontal;
        width: 100%;
        height: 1;
    }
    #palette_title, .modal_title {
        text-style: bold;
        margin-bottom: 0;
    }
    #palette_title {
        width: auto;
        text-wrap: nowrap;
    }
    #palette_close_hint {
        width: 1fr;
        color: $text-muted;
        text-align: right;
        text-wrap: nowrap;
    }
    .modal_subtitle {
        color: $text-muted;
    }
    .modal_title {
        padding-top: 0;
    }
    #palette_search {
        margin-bottom: 1;
    }
    #palette_options {
        height: 1fr;
        margin-bottom: 1;
    }
    #palette_hint, #setup_api_hint, #setup_repo_path, #model_api_hint {
        color: $text-muted;
    }
    #palette_hint {
    }
    #cli_input_container {
        height: auto;
        margin: 0 0 1 0;
    }
    #cli_input {
        border: round $accent;
        background: $panel;
        padding: 0 1;
    }
    #cli_input:focus {
        border: round $accent;
    }
    #cli_suggestions {
        dock: top;
        display: none;
        height: auto;
        max-height: 10;
        overflow: auto;
        border: round $accent;
        background: $panel;
        padding: 0;
        margin-bottom: 1;
    }
    #cli_suggestions:focus {
        display: none;
    }
    #cli_suggestions > .option-list--option {
        padding: 0 1;
    }
    #cli_suggestions > .option-list--option-highlighted {
        background: $accent;
        color: $text;
        text-style: bold;
    }
    #dashboard_header.banner-hidden {
        display: none;
    }
    #dashboard_header.banner-compact {
        height: 3;
        text-align: center;
        text-style: bold;
    }
    #dashboard_header.banner-normal {
        height: auto;
    }
    #dashboard_header.banner-large {
        height: auto;
    }
    .field_row {
        height: auto;
        margin: 0;
    }
    .field_label {
        width: 15;
        color: $text-muted;
        padding-top: 0;
    }
    .field_input {
        width: 1fr;
    }
    .setup_agents {
        height: auto;
    }
    .modal_actions {
        margin-top: 0;
        padding-top: 0;
        height: auto;
    }
    .section_title {
        text-style: bold;
    }
    #setup_status, #model_api_hint, #model_error, #settings_error, #layout_error {
        margin-top: 0;
    }
    #model_discovery_status {
        margin-top: 0;
        color: $text-muted;
    }
    #theme_picker_hint, #theme_modal_hint {
        color: $text-muted;
    }

"""
