from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agent_recall.storage.normalize import normalize_limit


def sync_rule_confidence(
    storage: Any,
    rules: list[dict[str, Any]],
    *,
    default_confidence: float = 0.6,
    reinforcement_factor: float = 0.15,
) -> dict[str, int]:
    now = storage._now_iso()
    default_value = max(0.0, min(1.0, float(default_confidence)))
    factor = max(0.0, min(1.0, float(reinforcement_factor)))
    inserted = 0
    updated = 0
    seen_rule_ids: set[str] = set()
    with storage._connect() as conn:
        for rule in rules:
            rule_id = str(rule.get("rule_id", "")).strip()
            tier = str(rule.get("tier", "")).strip().upper()
            line = str(rule.get("line", "")).strip()
            if not rule_id or not tier or not line:
                continue
            if rule_id in seen_rule_ids:
                continue
            seen_rule_ids.add(rule_id)
            row = conn.execute(
                """
                SELECT confidence, reinforcement_count
                FROM rule_confidence
                WHERE rule_id = ? AND tenant_id = ? AND project_id = ?
                LIMIT 1
                """,
                (rule_id, storage.tenant_id, storage.project_id),
            ).fetchone()
            if row:
                current = max(0.0, min(1.0, float(row["confidence"])))
                reinforced = current + ((1.0 - current) * factor)
                reinforcement_count = int(row["reinforcement_count"]) + 1
                conn.execute(
                    """
                    UPDATE rule_confidence
                    SET tier = ?, line = ?, confidence = ?, reinforcement_count = ?,
                        last_reinforced_at = ?, is_stale = 0, updated_at = ?
                    WHERE rule_id = ? AND tenant_id = ? AND project_id = ?
                    """,
                    (
                        tier,
                        line,
                        max(0.0, min(1.0, reinforced)),
                        reinforcement_count,
                        now,
                        now,
                        rule_id,
                        storage.tenant_id,
                        storage.project_id,
                    ),
                )
                updated += 1
            else:
                conn.execute(
                    """
                    INSERT INTO rule_confidence (
                        rule_id, tenant_id, project_id, tier, line, confidence,
                        reinforcement_count, last_reinforced_at, last_decayed_at, is_stale,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule_id,
                        storage.tenant_id,
                        storage.project_id,
                        tier,
                        line,
                        default_value,
                        1,
                        now,
                        None,
                        0,
                        now,
                        now,
                    ),
                )
                inserted += 1
    return {"inserted": inserted, "updated": updated, "total": inserted + updated}


def list_rule_confidence(
    storage: Any,
    *,
    limit: int = 200,
    stale_only: bool = False,
) -> list[dict[str, Any]]:
    query = (
        "SELECT rule_id, tier, line, confidence, reinforcement_count, "
        "last_reinforced_at, last_decayed_at, is_stale, created_at, updated_at "
        "FROM rule_confidence WHERE tenant_id = ? AND project_id = ?"
    )
    params: list[Any] = [storage.tenant_id, storage.project_id]
    if stale_only:
        query += " AND is_stale = 1"
    query += " ORDER BY confidence ASC, updated_at DESC LIMIT ?"
    params.append(normalize_limit(limit))
    with storage._connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "rule_id": str(row["rule_id"]),
            "tier": str(row["tier"]),
            "line": str(row["line"]),
            "confidence": float(row["confidence"]),
            "reinforcement_count": int(row["reinforcement_count"]),
            "last_reinforced_at": str(row["last_reinforced_at"])
            if row["last_reinforced_at"]
            else None,
            "last_decayed_at": str(row["last_decayed_at"]) if row["last_decayed_at"] else None,
            "is_stale": bool(int(row["is_stale"])),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    ]


def decay_rule_confidence(
    storage: Any,
    *,
    half_life_days: float = 45.0,
    stale_after_days: float = 60.0,
) -> dict[str, int]:
    now_dt = datetime.now(UTC)
    half_life = max(1.0, float(half_life_days))
    stale_after = max(1.0, float(stale_after_days))
    decayed = 0
    stale_marked = 0
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT rule_id, confidence, is_stale, last_reinforced_at, created_at
            FROM rule_confidence
            WHERE tenant_id = ? AND project_id = ?
            """,
            (storage.tenant_id, storage.project_id),
        ).fetchall()
        for row in rows:
            anchor_raw = row["last_reinforced_at"] or row["created_at"]
            try:
                anchor_dt = datetime.fromisoformat(str(anchor_raw))
            except ValueError:
                anchor_dt = now_dt
            elapsed_days = max(0.0, (now_dt - anchor_dt).total_seconds() / 86_400.0)
            current_confidence = max(0.0, min(1.0, float(row["confidence"])))
            decayed_confidence = current_confidence * (0.5 ** (elapsed_days / half_life))
            stale = elapsed_days >= stale_after
            previous_stale = bool(int(row["is_stale"]))
            if stale and not previous_stale:
                stale_marked += 1
            if abs(decayed_confidence - current_confidence) > 0.0001 or stale != previous_stale:
                conn.execute(
                    """
                    UPDATE rule_confidence
                    SET confidence = ?, is_stale = ?, last_decayed_at = ?, updated_at = ?
                    WHERE rule_id = ? AND tenant_id = ? AND project_id = ?
                    """,
                    (
                        max(0.0, min(1.0, decayed_confidence)),
                        1 if stale else 0,
                        now_dt.isoformat(),
                        now_dt.isoformat(),
                        str(row["rule_id"]),
                        storage.tenant_id,
                        storage.project_id,
                    ),
                )
                decayed += 1
    return {"decayed": decayed, "stale_marked": stale_marked}


def archive_and_prune_rule_confidence(
    storage: Any,
    *,
    max_confidence: float = 0.35,
    stale_only: bool = True,
    dry_run: bool = True,
    limit: int = 500,
) -> list[dict[str, Any]]:
    threshold = max(0.0, min(1.0, float(max_confidence)))
    query = (
        "SELECT rule_id, tier, line, confidence, reinforcement_count, "
        "last_reinforced_at, last_decayed_at, is_stale "
        "FROM rule_confidence "
        "WHERE tenant_id = ? AND project_id = ? AND confidence <= ?"
    )
    params: list[Any] = [storage.tenant_id, storage.project_id, threshold]
    if stale_only:
        query += " AND is_stale = 1"
    query += " ORDER BY confidence ASC, updated_at DESC LIMIT ?"
    params.append(normalize_limit(limit))

    with storage._connect() as conn:
        rows = conn.execute(query, params).fetchall()
        candidates = [
            {
                "rule_id": str(row["rule_id"]),
                "tier": str(row["tier"]),
                "line": str(row["line"]),
                "confidence": float(row["confidence"]),
                "reinforcement_count": int(row["reinforcement_count"]),
                "last_reinforced_at": (
                    str(row["last_reinforced_at"]) if row["last_reinforced_at"] else None
                ),
                "last_decayed_at": str(row["last_decayed_at"]) if row["last_decayed_at"] else None,
                "is_stale": bool(int(row["is_stale"])),
            }
            for row in rows
        ]
        if dry_run or not candidates:
            return candidates

        archived_at = datetime.now(UTC).isoformat()
        for candidate in candidates:
            conn.execute(
                """
                INSERT INTO rule_confidence_archive (
                    rule_id, tenant_id, project_id, tier, line, confidence,
                    reinforcement_count, last_reinforced_at, last_decayed_at,
                    is_stale, archived_at, reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate["rule_id"],
                    storage.tenant_id,
                    storage.project_id,
                    candidate["tier"],
                    candidate["line"],
                    candidate["confidence"],
                    candidate["reinforcement_count"],
                    candidate["last_reinforced_at"],
                    candidate["last_decayed_at"],
                    1 if candidate["is_stale"] else 0,
                    archived_at,
                    "low-confidence prune",
                ),
            )
            conn.execute(
                """
                DELETE FROM rule_confidence
                WHERE rule_id = ? AND tenant_id = ? AND project_id = ?
                """,
                (candidate["rule_id"], storage.tenant_id, storage.project_id),
            )
    return candidates


def get_rule_confidence_summary(storage: Any) -> dict[str, Any]:
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_rules,
                SUM(CASE WHEN is_stale = 1 THEN 1 ELSE 0 END) AS stale_rules,
                SUM(CASE WHEN confidence <= 0.35 THEN 1 ELSE 0 END) AS low_confidence_rules,
                AVG(confidence) AS average_confidence,
                MIN(COALESCE(last_reinforced_at, created_at)) AS oldest_signal_at
            FROM rule_confidence
            WHERE tenant_id = ? AND project_id = ?
            """,
            (storage.tenant_id, storage.project_id),
        ).fetchone()
    if row is None:
        return {
            "total_rules": 0,
            "stale_rules": 0,
            "low_confidence_rules": 0,
            "average_confidence": 0.0,
            "oldest_signal_at": None,
        }
    return {
        "total_rules": int(row["total_rules"] or 0),
        "stale_rules": int(row["stale_rules"] or 0),
        "low_confidence_rules": int(row["low_confidence_rules"] or 0),
        "average_confidence": float(row["average_confidence"] or 0.0),
        "oldest_signal_at": str(row["oldest_signal_at"]) if row["oldest_signal_at"] else None,
    }
