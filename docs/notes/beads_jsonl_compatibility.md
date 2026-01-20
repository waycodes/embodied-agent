# Beads JSONL Filename Compatibility

## Background

The `bd` (beads) issue tracking tool stores issues in a JSONL file. The default filename has changed across versions:

- **Older versions**: `beads.jsonl`
- **Recent versions (v0.4+)**: `issues.jsonl`

## This Repository

This repository uses `issues.jsonl` as the default filename, located at `.beads/issues.jsonl`.

## Compatibility Notes

1. **Git sync**: When syncing with git, ensure all collaborators use the same beads version to avoid filename conflicts.

2. **Migration**: If migrating from an older beads installation:
   ```bash
   mv .beads/beads.jsonl .beads/issues.jsonl
   bd sync --import-only
   ```

3. **Configuration**: The filename can be configured in `.beads/config.yaml`:
   ```yaml
   jsonl_file: issues.jsonl
   ```

4. **Prefix mismatch**: If you see warnings about prefix mismatch (e.g., `EDK-` vs `bd-`), this indicates the database and JSONL file have diverged. Use `bd sync --import-only` to reconcile.

## Troubleshooting

If `bd list` shows "Database out of sync with JSONL":
```bash
bd sync --import-only
```

If issues have wrong prefixes after git pull:
```bash
bd sync --rename-on-import
```
