import json

with open("/Users/natecard/OnHere/Repos/self-docs/agent_recall/ralph/prd.json") as f:
    data = json.load(f)

# Use today's date prefix based on system time: AR-260220
prefix = "AR-260220"
for i, item in enumerate(data["items"], start=1):
    item["id"] = f"{prefix}-{i:02d}"

with open("/Users/natecard/OnHere/Repos/self-docs/agent_recall/ralph/prd.json", "w") as f:
    json.dump(data, f, indent=2)

print("Updated PRD IDs")
