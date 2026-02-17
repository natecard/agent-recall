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
        gap: 1;
    }
    #dashboard_all_sidebar {
        width: 44;
        min-width: 32;
        height: auto;
        min-height: 0;
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
    #dashboard_overview_row {
        layout: horizontal;
        width: 100%;
        height: auto;
        min-height: 0;
        gap: 1;
    }
    #dashboard_overview_row #dashboard_knowledge {
        width: 1fr;
    }
    #dashboard_overview_row #dashboard_sources {
        width: 1fr;
    }
    @media (max-width: 120) {
        #dashboard_all_grid {
            layout: vertical;
        }
        #dashboard_all_sidebar {
            width: 100%;
            min-width: 0;
        }
        #dashboard_overview_row {
            layout: vertical;
        }
    }
    #terminal_panel {
        height: 1fr;
        min-height: 6;
        overflow: auto;
        display: none;
        border: round $accent;
        background: $panel;
        padding: 1 2;
        margin-bottom: 1;
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
        padding: 1 2;
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
        width: 74%;
        max-width: 94;
        height: auto;
        max-height: 78%;
        padding: 1 3;
        background: $panel;
        border: none;
        overflow: auto;
    }
    #modal_card {
        width: 64%;
        max-width: 84;
        height: auto;
        max-height: 82%;
        padding: 1 2;
        background: $panel;
        border: round $accent;
        overflow: auto;
    }
    #diff_content {
        height: 1fr;
        overflow: auto;
        background: $panel;
        border: round $accent;
        padding: 1 2;
        margin-top: 1;
    }
    #palette_header {
        layout: horizontal;
        width: 100%;
        height: 1;
        margin-bottom: 1;
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
        margin-bottom: 1;
    }
    .modal_title {
        padding-top: 1;
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
        margin-top: 1;
    }
    .field_row {
        height: auto;
        margin: 0 0 1 0;
    }
    .field_label {
        width: 15;
        color: $text-muted;
        padding-top: 1;
    }
    .field_input {
        width: 1fr;
    }
    .setup_agents {
        height: auto;
        margin-bottom: 1;
    }
    .modal_actions {
        margin-top: 1;
        padding-top: 1;
        height: auto;
    }
    #setup_status, #model_api_hint, #model_error, #settings_error {
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
