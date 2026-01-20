# Agent Instructions

This project uses **bd** (beads) for AI-supervised issue tracking.

## Beads Workflow

1. **Find ready work**: `bd ready`
2. **Claim your task**: `bd update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work**: Create issues for bugs/TODOs found during work
5. **Complete**: `bd close <id> "Done: <summary>"`
6. **Repeat**: Check for newly unblocked tasks

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details  
bd create "title" <type> <priority>  # Create issue (bug/feature/task/chore, 0-4)
bd update <id> --status in_progress  # Claim work
bd close <id> "reason"  # Complete work
bd dep <id> --blocks <other>  # Add dependency
bd sync               # Sync with git
```

## Issue Types & Priority

- **bug** (0-critical, 1-high, 2-medium, 3-low, 4-backlog)
- **feature** - New functionality
- **task** - Work item (tests, docs, refactoring)
- **chore** - Maintenance work

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

