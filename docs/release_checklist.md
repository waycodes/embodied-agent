# Release Checklist

## Pre-Release

- [ ] All tests pass: `pytest tests/`
- [ ] Type checking passes: `mypy embodied_datakit/`
- [ ] Linting passes: `ruff check embodied_datakit/`
- [ ] Documentation is up to date
- [ ] CHANGELOG.md updated with release notes
- [ ] Version bumped in `pyproject.toml`

## Code Quality

- [ ] No TODO/FIXME comments in release code
- [ ] All public APIs have docstrings
- [ ] No debug print statements
- [ ] No hardcoded paths or credentials

## Testing

- [ ] Unit tests cover core functionality
- [ ] Integration tests pass
- [ ] Manual smoke test on sample dataset

## Documentation

- [ ] README.md is current
- [ ] API documentation generated
- [ ] Examples are runnable
- [ ] CITATION.cff is correct

## Release

- [ ] Create git tag: `git tag -a v0.x.x -m "Release v0.x.x"`
- [ ] Push tag: `git push origin v0.x.x`
- [ ] Build package: `python -m build`
- [ ] Verify package contents: `tar -tzf dist/*.tar.gz`

## Post-Release

- [ ] Verify installation: `pip install dist/*.whl`
- [ ] Run smoke test with installed package
- [ ] Update any dependent projects
