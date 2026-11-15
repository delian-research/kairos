# Releasing

How to cut a new release of **kairos** to PyPI.

- **Distribution name:** `kairos-analytics` (what users `pip install`)
- **Import name:** `kairos` (what users `import`)
- **Publishing:** automated via [`.github/workflows/release.yml`](.github/workflows/release.yml)
  using PyPI [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).
  **No API tokens** are stored anywhere.

The release is triggered by **publishing a GitHub Release** — not by pushing a
tag alone. Publishing the Release fires the `release: published` event that the
workflow listens for.

> **PyPI versions are write-once.** Once `X.Y.Z` is published it can never be
> re-uploaded or overwritten — a mistake means burning the number and shipping
> `X.Y.Z+1`. When in doubt, do the [TestPyPI dry run](#optional-testpypi-dry-run)
> first.

## One-time setup (NOT yet done)

These must be configured once before the first release:

- A **Trusted Publisher** on PyPI bound to repo `delian-research/kairos`, workflow
  `release.yml`, environment `pypi`.
- A GitHub **Environment** named `pypi` (Settings → Environments). It holds no
  secrets — OIDC needs none.

If you ever move the repo, rename the workflow, or change the environment name,
update the Trusted Publisher on PyPI to match, or publishing will fail auth.

## Release steps

### 1. Pick the version

Follow [SemVer](https://semver.org/): `MAJOR.MINOR.PATCH`. Pre-1.0, breaking
changes bump MINOR and everything else bumps PATCH.

### 2. Bump the version and update the changelog

The version lives in two tracked files (the package `__version__` is read from
installed metadata at runtime, so there is nothing to edit in `src/`):

1. `pyproject.toml` → `[project].version`
2. `uv.lock` → regenerate so it matches (do not hand-edit):

   ```bash
   uv lock
   ```

3. `CHANGELOG.md` → move items from `## [Unreleased]` into a new
   `## [X.Y.Z]` section and update the compare/tag links at the bottom.

Commit on a branch and merge via PR as usual. Make sure `main` is green in CI
before releasing.

### 3. Sanity-check locally (optional but recommended)

```bash
git checkout main && git pull
uv run ruff check . && uv run mypy src && uv run pytest
uv build && uvx twine check dist/*
```

### 4. Tag and publish the GitHub Release

From an up-to-date `main` (replace `X.Y.Z`):

```bash
gh release create vX.Y.Z --target main --title "vX.Y.Z" --notes-file - <<'EOF'
<summary of the release — or point at CHANGELOG.md>
EOF
```

This creates the `vX.Y.Z` tag at `main` and publishes the Release, which starts
`release.yml`.

### 5. Watch the publish

```bash
gh run watch "$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
```

The `build` job builds the sdist + wheel and runs `twine check`; the
`publish-pypi` job uploads to PyPI over OIDC.

### 6. Verify it's live

```bash
# Metadata visible on the index
curl -s https://pypi.org/pypi/kairos-analytics/json | python3 -c "import sys,json;print(json.load(sys.stdin)['info']['version'])"

# Clean-room install + import check
uv run --with kairos-analytics --no-project python -c "import kairos; print(kairos.__version__)"
```

Both should print `X.Y.Z`.

## Optional: TestPyPI dry run

TestPyPI is a separate registry, so it needs its own one-time setup:

- A Trusted Publisher on **test.pypi.org** for `delian-research/kairos`, workflow
  `release.yml`, environment `testpypi`.
- A GitHub Environment named `testpypi`.

Then trigger the manual path (the `workflow_dispatch` branch of `release.yml`
publishes to TestPyPI):

```bash
gh workflow run release.yml --ref main
gh run watch "$(gh run list --workflow=release.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
```

Install from TestPyPI to confirm (deps still resolve from real PyPI):

```bash
uv pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ kairos-analytics
```

> TestPyPI versions are also write-once. Use a throwaway version suffix (e.g.
> `X.Y.Z.devN`) when iterating there so you don't burn the real number.

## Troubleshooting

- **`Unable to resolve action ... unable to find version`** — an action tag in a
  workflow doesn't exist as a *moving* tag. Some actions (e.g. `astral-sh/setup-uv`)
  only ship full tags like `v9.0.0`, not a bare `v9`. Pin to the full version.
- **Auth / `invalid-publisher` on upload** — the repo, workflow filename, or
  environment name no longer matches the Trusted Publisher registered on PyPI.
- **Publish job skipped** — you pushed a tag but did not publish a *Release*.
  The workflow triggers on `release: published`, not on tag push.
- **`File already exists` from PyPI** — that version was already published;
  bump to the next version.
