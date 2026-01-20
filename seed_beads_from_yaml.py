#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml  # pyyaml


# ----------------------------
# Minimal Beads JSONL schema
# ----------------------------

@dataclasses.dataclass(frozen=True)
class DepEdge:
    issue_id: str
    depends_on_id: str
    dep_type: str
    created_at: str
    created_by: str = "seed-script"
    metadata: str = ""
    thread_id: str = ""

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "issue_id": self.issue_id,
            "depends_on_id": self.depends_on_id,
            "type": self.dep_type,
            "created_at": self.created_at,
        }
        if self.created_by:
            out["created_by"] = self.created_by
        if self.metadata:
            out["metadata"] = self.metadata
        if self.thread_id:
            out["thread_id"] = self.thread_id
        return out


@dataclasses.dataclass(frozen=True)
class IssueObj:
    id: str
    title: str
    priority: int
    created_at: str
    updated_at: str

    # Optional (Beads supports these JSON fields) :contentReference[oaicite:14]{index=14}
    description: str = ""
    design: str = ""
    acceptance_criteria: str = ""
    notes: str = ""
    status: str = ""          # open/in_progress/blocked/deferred/closed/pinned/... :contentReference[oaicite:15]{index=15}
    issue_type: str = ""      # bug/feature/task/epic/chore/... :contentReference[oaicite:16]{index=16}
    labels: Optional[List[str]] = None
    dependencies: Optional[List[DepEdge]] = None

    created_by: str = "seed-script"
    updated_by: str = "seed-script"  # Not a native Issue field; stored in notes if desired.

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "priority": int(self.priority),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.description:
            out["description"] = self.description
        if self.design:
            out["design"] = self.design
        if self.acceptance_criteria:
            out["acceptance_criteria"] = self.acceptance_criteria
        if self.notes:
            out["notes"] = self.notes
        if self.status:
            out["status"] = self.status
        if self.issue_type:
            out["issue_type"] = self.issue_type
        if self.labels:
            out["labels"] = list(self.labels)
        if self.dependencies:
            out["dependencies"] = [d.to_json() for d in self.dependencies]
        if self.created_by:
            out["created_by"] = self.created_by
        return out


# ----------------------------
# ID generation helpers
# ----------------------------

def _sha6(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:6]


def gen_root_id(prefix: str, key: str, title: str) -> str:
    # GenerateHashID in Beads yields prefix-{6-8 hex}; we mimic stable 6-hex here. :contentReference[oaicite:17]{index=17}
    return f"{prefix}-{_sha6(f'{key}:{title}')}"


def gen_child_id(parent_id: str, child_number: int) -> str:
    # Matches Beads documented hierarchical format: parent.N :contentReference[oaicite:18]{index=18}
    return f"{parent_id}.{child_number}"


def utc_now_rfc3339() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------
# Planner loading + expansion
# ----------------------------

class PlanError(RuntimeError):
    pass


def load_plan(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    raise PlanError(f"Expected list, got {type(v)}")


def normalize_labels(issue: Dict[str, Any], meta_labels: List[str]) -> List[str]:
    labels = []
    labels.extend(meta_labels)
    labels.extend(ensure_list(issue.get("labels")))
    # de-dup while preserving order
    seen = set()
    out = []
    for x in labels:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_issue_graph(plan: Dict[str, Any]) -> Tuple[List[IssueObj], Dict[str, str]]:
    meta = plan.get("meta", {})
    prefix = meta.get("id_prefix", "bd")
    created_by = meta.get("created_by", "seed-script")

    global_labels = plan.get("labels", {}).get("project", [])
    if not isinstance(global_labels, list):
        raise PlanError("labels.project must be a list")

    now = utc_now_rfc3339()
    # In Beads, status/issue_type may default if omitted, but priority must be explicit for P0 (0). :contentReference[oaicite:19]{index=19}

    key_to_id: Dict[str, str] = {}
    issues_out: List[IssueObj] = []

    def emit_issue_node(node: Dict[str, Any], parent_key: Optional[str], sibling_index: int) -> str:
        key = node["key"]
        title = node["title"]
        issue_type = node.get("issue_type", "")
        status = node.get("status", "")
        priority = int(node.get("priority", meta.get("default_priority", 2)))

        # ID assignment
        if parent_key is None:
            issue_id = gen_root_id(prefix, key, title)
        else:
            parent_id = key_to_id[parent_key]
            issue_id = gen_child_id(parent_id, sibling_index)

        if key in key_to_id:
            raise PlanError(f"Duplicate issue key: {key}")
        key_to_id[key] = issue_id

        # Gather fields
        desc = node.get("description", "") or ""
        design = node.get("design", "") or ""
        ac = node.get("acceptance_criteria", "") or ""
        notes = node.get("notes", "") or ""

        labels = normalize_labels(node, global_labels)

        issues_out.append(
            IssueObj(
                id=issue_id,
                title=title,
                priority=priority,
                created_at=now,
                updated_at=now,
                description=desc,
                design=design,
                acceptance_criteria=ac,
                notes=notes,
                status=status,
                issue_type=issue_type,
                labels=labels,
                dependencies=[],
                created_by=created_by,
            )
        )

        # Recurse children
        children = ensure_list(node.get("children"))
        for i, child in enumerate(children, start=1):
            emit_issue_node(child, parent_key=key, sibling_index=i)

        return issue_id

    # Emit all roots in order
    roots = ensure_list(plan.get("issues"))
    for idx, root in enumerate(roots, start=1):
        # sibling_index unused for roots
        emit_issue_node(root, parent_key=None, sibling_index=idx)

    # Build dependency edges (including implicit parent-child edges)
    issues_by_id = {iss.id: iss for iss in issues_out}
    mutable_deps: Dict[str, List[DepEdge]] = {iss.id: [] for iss in issues_out}

    def add_dep(from_key: str, to_key: str, dep_type: str) -> None:
        from_id = key_to_id[from_key]
        to_id = key_to_id[to_key]
        mutable_deps[from_id].append(
            DepEdge(
                issue_id=from_id,
                depends_on_id=to_id,
                dep_type=dep_type,
                created_at=now,
                created_by=created_by,
            )
        )

    # explicit deps
    def walk(node: Dict[str, Any], parent_key: Optional[str]) -> None:
        key = node["key"]

        # implicit hierarchy edge: child depends on parent (parent-child) :contentReference[oaicite:20]{index=20}
        if parent_key is not None:
            add_dep(key, parent_key, "parent-child")

        for dep in ensure_list(node.get("deps")):
            dep_type = dep["type"]
            for on_key in ensure_list(dep.get("on")):
                add_dep(key, on_key, dep_type)

        for child in ensure_list(node.get("children")):
            walk(child, parent_key=key)

    for root in roots:
        walk(root, parent_key=None)

    # finalize issues with deps
    finalized: List[IssueObj] = []
    for iss in issues_out:
        deps = mutable_deps.get(iss.id, [])
        finalized.append(dataclasses.replace(iss, dependencies=deps))

    return finalized, key_to_id


def write_beads_files(issues: List[IssueObj], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    issues_jsonl = out_dir / "issues.jsonl"
    beads_jsonl = out_dir / "beads.jsonl"

    # Write primary file
    with issues_jsonl.open("w", encoding="utf-8") as f:
        for iss in issues:
            f.write(json.dumps(iss.to_json(), ensure_ascii=False))
            f.write("\n")

    # Also write secondary filename for compatibility (some versions historically used beads.jsonl). :contentReference[oaicite:21]{index=21}
    with beads_jsonl.open("w", encoding="utf-8") as f:
        for iss in issues:
            f.write(json.dumps(iss.to_json(), ensure_ascii=False))
            f.write("\n")


def write_default_beads_config(beads_dir: Path) -> None:
    # Minimal repo-local defaults; config docs show json/no-daemon style options. :contentReference[oaicite:22]{index=22}
    cfg = {
        "json": True,
        "no-daemon": True,
        "sync-branch": "beads-sync",
    }
    path = beads_dir / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=str, required=True, help="Path to embodied_datakit.bdplan.yaml")
    ap.add_argument("--repo-root", type=str, default=".", help="Repo root (default: .)")
    args = ap.parse_args()

    plan_path = Path(args.plan)
    repo_root = Path(args.repo_root).resolve()
    beads_dir = repo_root / ".beads"

    plan = load_plan(plan_path)
    issues, key_to_id = build_issue_graph(plan)

    write_beads_files(issues, beads_dir)
    write_default_beads_config(beads_dir)

    # Emit a mapping file for convenience
    mapping_path = beads_dir / "seed_key_to_id.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(key_to_id, f, indent=2, sort_keys=True)

    print(f"Wrote {len(issues)} issues to {beads_dir / 'issues.jsonl'}")
    print(f"Wrote config to {beads_dir / 'config.yaml'}")
    print(f"Wrote key→id mapping to {mapping_path}")
    print("")
    print("Next:")
    print("  bd init --quiet   # imports existing JSONL into bd cache (if needed)")
    print("  bd ready --json   # list unblocked work")


if __name__ == "__main__":
    main()
