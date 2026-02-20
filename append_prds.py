import json

with open("/Users/natecard/OnHere/Repos/self-docs/agent_recall/ralph/prd.json") as f:
    data = json.load(f)

# Find the next available ID for today
prefix = "AR-260220"
existing_indices = []
for item in data.get("items", []):
    if item["id"].startswith(prefix):
        try:
            idx = int(item["id"].split("-")[-1])
            existing_indices.append(idx)
        except ValueError:
            pass

next_idx = max(existing_indices) + 1 if existing_indices else 1

new_item_1 = {
    "id": f"{prefix}-{next_idx:02d}",
    "priority": 1,
    "title": "TUI: Fix Arrow Key Navigation in Iteration Views",
    "description": (
        "The up and down arrow keys currently do not function correctly when navigating "
        "within the interactive iterations views. Users are unable to scroll or select "
        "items using the keyboard as expected."
    ),
    "user_story": (
        "As a TUI user, I want the arrow keys to work seamlessly in the interactive "
        "iterations views so I can navigate through iterations without taking my hands "
        "off the keyboard."
    ),
    "steps": [
        (
            "Identify the widget responsible for the interactive iterations view "
            "(likely `InteractiveTimelineWidget` or related list view)."
        ),
        "Inspect the key binding and focus management logic for arrow keys within this widget.",
        (
            "Implement or fix the `on_key` or `BINDINGS` handlers to properly intercept "
            "arrow keys and update the focused or highlighted item."
        ),
        (
            "Ensure the view scrolls automatically if the newly selected item is out "
            "of the current viewport."
        ),
    ],
    "acceptance_criteria": [
        (
            "Pressing the Down arrow key moves the selection/focus to the next item "
            "in the iteration view."
        ),
        (
            "Pressing the Up arrow key moves the selection/focus to the previous item "
            "in the iteration view."
        ),
        "The view scrolls correctly to keep the selected item visible.",
        "No other keybindings are negatively affected by this change.",
    ],
    "validation_commands": [
        "uv run pytest tests/ -k iteration_view -v",
        "uv run agent-recall open",
    ],
    "passes": False,
}

new_item_2 = {
    "id": f"{prefix}-{(next_idx + 1):02d}",
    "priority": 2,
    "title": "TUI: Remove Non-Functional LLM View Option",
    "description": (
        "There is currently an option to select an 'LLM view' in the UI, but selecting it "
        "results in no view being displayed. Since this view is not functional or required, "
        "the option should be completely removed from the UI."
    ),
    "user_story": (
        "As a TUI user, I want the UI to only present functional view options, so I don't "
        "get confused by selecting an 'LLM view' that doesn't actually exist."
    ),
    "steps": [
        (
            "Locate the UI component that renders the list of available views "
            "(likely in a sidebar, tab bar, or command palette)."
        ),
        (
            "Find the reference to the 'llm' view or 'LLM view' option in the "
            "configuration or hardcoded list."
        ),
        "Remove the 'llm' option from the list of selectable views.",
        (
            "Ensure that any routing or action logic that expected the 'llm' view "
            "gracefully handles its absence (or remove that logic entirely)."
        ),
    ],
    "acceptance_criteria": [
        "The 'LLM view' (or similarly named) option is no longer visible anywhere in the TUI.",
        "Users cannot navigate to an empty LLM view via keyboard shortcuts or command palette.",
        (
            "Removing the option does not break the rendering or functionality "
            "of the remaining valid views."
        ),
    ],
    "validation_commands": ["uv run pytest tests/ -v", "uv run agent-recall open"],
    "passes": False,
}

data["items"].extend([new_item_1, new_item_2])

with open("/Users/natecard/OnHere/Repos/self-docs/agent_recall/ralph/prd.json", "w") as f:
    json.dump(data, f, indent=2)

print("Added new PRDs")
