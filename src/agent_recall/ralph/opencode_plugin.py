from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OPENCODE_PLUGIN_FILENAME = "agent-recall-ralph.js"


@dataclass(frozen=True)
class OpenCodePluginPaths:
    config_dir: Path
    plugins_dir: Path
    plugin_path: Path


def get_opencode_plugin_paths(project_dir: Path) -> OpenCodePluginPaths:
    config_dir = project_dir / ".opencode"
    plugins_dir = config_dir / "plugins"
    plugin_path = plugins_dir / OPENCODE_PLUGIN_FILENAME
    return OpenCodePluginPaths(
        config_dir=config_dir,
        plugins_dir=plugins_dir,
        plugin_path=plugin_path,
    )


def render_opencode_plugin() -> str:
    return """import { appendFileSync, existsSync, mkdirSync, readFileSync } from "fs";
import { dirname, join } from "path";

const SESSION_EVENTS = new Set(["session.created", "session.idle"]);

function loadCurrentReport(reportPath) {
  if (!existsSync(reportPath)) {
    return null;
  }
  try {
    const raw = readFileSync(reportPath, "utf8");
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    return {
      iteration: parsed.iteration ?? null,
      item_id: parsed.item_id ?? null,
      item_title: parsed.item_title ?? null,
    };
  } catch (error) {
    return null;
  }
}

function appendEvent(eventsPath, payload) {
  mkdirSync(dirname(eventsPath), { recursive: true });
  appendFileSync(eventsPath, JSON.stringify(payload) + "\n", "utf8");
}

function buildEnvelope(baseDir, type, payload) {
  const reportPath = join(baseDir, ".agent", "ralph", "iterations", "current.json");
  const report = loadCurrentReport(reportPath);
  return {
    timestamp: new Date().toISOString(),
    type,
    payload,
    iteration: report?.iteration ?? null,
    item_id: report?.item_id ?? null,
    item_title: report?.item_title ?? null,
  };
}

function emitEvent(baseDir, type, payload) {
  const eventsPath = join(baseDir, ".agent", "ralph", "opencode_events.jsonl");
  appendEvent(eventsPath, buildEnvelope(baseDir, type, payload));
}

function summarizeToolPayload(input, output) {
  return {
    tool: input?.tool ?? output?.tool ?? null,
    args: output?.args ?? input?.args ?? input?.arguments ?? null,
    success: output?.success ?? null,
    error: output?.error ?? null,
  };
}

export const AgentRecallRalph = async ({ directory, worktree }) => {
  const baseDir = worktree || directory;
  if (!baseDir) {
    return {};
  }
  return {
    event: async ({ event }) => {
      if (!event || !SESSION_EVENTS.has(event.type)) {
        return;
      }
      emitEvent(baseDir, event.type, event);
    },
    "tool.execute.before": async (input, output) => {
      emitEvent(baseDir, "tool.execute.before", summarizeToolPayload(input, output));
    },
    "tool.execute.after": async (input, output) => {
      emitEvent(baseDir, "tool.execute.after", summarizeToolPayload(input, output));
    },
  };
};
"""


def install_opencode_plugin(project_dir: Path) -> bool:
    paths = get_opencode_plugin_paths(project_dir)
    paths.plugins_dir.mkdir(parents=True, exist_ok=True)
    payload = render_opencode_plugin()
    if paths.plugin_path.exists():
        try:
            existing = paths.plugin_path.read_text(encoding="utf-8")
        except OSError:
            existing = ""
        if existing == payload:
            return False
    paths.plugin_path.write_text(payload, encoding="utf-8")
    return True


def uninstall_opencode_plugin(project_dir: Path) -> bool:
    paths = get_opencode_plugin_paths(project_dir)
    if not paths.plugin_path.exists():
        return False
    paths.plugin_path.unlink()
    return True
