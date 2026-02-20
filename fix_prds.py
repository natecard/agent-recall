import json

with open("/Users/natecard/OnHere/Repos/self-docs/agent_recall/ralph/prd.json") as f:
    data = json.load(f)

for item in data["items"]:
    # Remove 'type'
    if "type" in item:
        del item["type"]

    # Rename 'requirements' to 'steps'
    if "requirements" in item:
        item["steps"] = item.pop("requirements")

    # Rename 'acceptance' to 'acceptance_criteria'
    if "acceptance" in item:
        item["acceptance_criteria"] = item.pop("acceptance")

    # Rename 'validation' to 'validation_commands'
    if "validation" in item:
        item["validation_commands"] = item.pop("validation")

    # Ensure user_story exists
    if "user_story" not in item:
        item["user_story"] = "As a user, I want this feature implemented."
        if item["id"] == "AR-111":
            item["user_story"] = (
                "As a developer using the TUI, I want to clearly see the iteration identifier "
                "and resize the file tree in the Iteration Diff Viewer so I can easily "
                "contextualize and review code changes."
            )
        elif item["id"] == "AR-113":
            item["user_story"] = (
                "As a developer, I want a single unified Sessions view that also shows source "
                "statuses so I can manage everything in one place without switching to a "
                "separate Sources view."
            )
        elif item["id"] == "AR-114":
            item["user_story"] = (
                "As a developer, I want standardized CSS spacing across the TUI to ensure "
                "visual consistency and make adding new UI components simpler."
            )

    # Ensure validation_commands exists
    if "validation_commands" not in item:
        if item["id"] == "AR-111":
            item["validation_commands"] = [
                "uv run pytest tests/ -k diff_viewer -v",
                "uv run agent-recall open",
            ]
        elif item["id"] == "AR-113":
            item["validation_commands"] = [
                "uv run pytest tests/ -k sessions_view -v",
                "uv run agent-recall open",
            ]
        elif item["id"] == "AR-114":
            item["validation_commands"] = ["uv run pytest tests/ -v", "uv run agent-recall open"]
        else:
            item["validation_commands"] = ["uv run pytest tests/ -v", "uv run agent-recall open"]

    # Reorder keys
    desired_order = [
        "id",
        "priority",
        "title",
        "description",
        "user_story",
        "steps",
        "acceptance_criteria",
        "validation_commands",
        "passes",
    ]

    new_item = {}
    for key in desired_order:
        if key in item:
            new_item[key] = item[key]
    for key in item:
        if key not in desired_order:
            new_item[key] = item[key]

    item.clear()
    item.update(new_item)

with open("/Users/natecard/OnHere/Repos/self-docs/agent_recall/ralph/prd.json", "w") as f:
    json.dump(data, f, indent=2)

print("Done updating")
