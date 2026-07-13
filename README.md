<!--
SPDX-FileCopyrightText: 2026 CoreWeave

SPDX-License-Identifier: MIT
-->

# translocate

`translocate` moves an installed Python package from where it lives now
to somewhere else. The two destinations it knows about are a wheel file
(for transporting the package around) and another interpreter's
site-packages (for projecting the package into a different environment
without going through the index).

It recovers the package's logical contents from the install's RECORD,
classifies each file by the install scheme it came from, and then
either serializes the result back into a wheel archive or projects it
through a target interpreter's scheme using hardlinks, symlinks, or
copies. The output advertises itself accurately: a regenerated wheel
reïnstalls cleanly but is not bit-identical to the upstream artifact,
and a cloned prefix carries a regenerated RECORD describing the files
at their new locations so `pip` and `uv` see the cloned distribution
as a first-class install.

## Clone installed system packages into a uv venv

```sh
uv venv ~/envs/inference
translocate clone torch numpy Jinja2 \
    --target ~/envs/inference \
    --link-mode hardlink \
    --include-deps \
    --smoke-test-imports
```

Pass one or more distribution names. Each becomes a root of the clone;
the union deduplicates by PEP 503 canonical name, so naming a package
that also turns up as a dep of another root materializes it once.

`--link-mode hardlink` shares inodes with the source install, so the
clone takes near-zero extra disk and survives the source being
uninstalled. `--link-mode symlink` is appropriate when the source is a
stable shared store and the clone is read-mostly; `--link-mode copy`
is the universal fallback.

> [!IMPORTANT]
> In a container build, clone with `--link-mode symlink`. A hardlink
> cannot cross OCI image layers, so the build filesystem copies the
> full file data into the new layer and the image ends up carrying
> two copies of every cloned package. Symlinks stay a few bytes each,
> and the base image's site-packages is a stable target for them.

`--include-deps` walks the source environment's installed dependency
closure and clones each member with the same link mode. Members that
cannot be translocated (editable installs, legacy egg-info layouts)
are skipped and reported.

`--smoke-test-imports` in this example runs `import torch` (and the
import for each other root) through the target interpreter once the
structural checks pass, then reports the outcome separately. A failed
import after the structural checks succeed usually points at a runtime
gap, like an untranslocatable dep.

### Pair a translocated root with its deps from an index

`--deps-file=<path>` (or `--deps-file=-` for stdout) writes a
`name==version` list of each translocated root and that root's direct
`Requires-Dist`, every entry pinned to the source-environment version:

```sh
translocate clone torch \
    --target ~/envs/inference \
    --deps-file ~/envs/inference/requirements.txt
```

The resulting file feeds `uv pip install -r` to satisfy the direct
deps from an index:

```sh
uv pip install -r ~/envs/inference/requirements.txt \
    --python ~/envs/inference/bin/python
```

### Capture the full closure as a lockfile

With `--include-deps`, the file expands from direct deps to the full
transitive closure, every entry still pinned:

```sh
translocate clone torch \
    --target ~/envs/inference \
    --include-deps \
    --deps-file ~/envs/inference/requirements.txt
```

The result is a pip-style lockfile of every package in the cloned
venv, suitable for rebuilding the same set elsewhere.

## Repackage installed packages as wheels

```sh
translocate repackage torch numpy --out ./wheels --include-deps
pip install ./wheels/torch-*.whl ./wheels/numpy-*.whl
```

`--out` specifies a directory. Wheels are emitted into that directory with
filenames generated per PEP 427 from the source distribution's metadata
(`{name}-{version}-{tag}.whl`). The CLI prints each produced
filename and ends with a summary listing every wheel written to the
`--out` directory.

The output wheels reïnstall faithfully but are not bit-for-bit
reproductions of the upstream artifacts. Any file the source install
rewrote in place (script shebangs, post-install patches) and the
RECORD file itself will differ from the original. The wheels are
suitable for reïnstall and redistribution; they are not suitable for
hash-pinning against an upstream hash.

## Lock uv projects against translocated packages

A `uv.lock` can pin the exact build of a translocated package, even a
version that no index carries, such as a special `torch 2.10.0+cu130`
build compiled into a container base image. To do this, use uv's
*flat index* feature, which allows you to treat a directory of wheels
as an index source. Point `uv lock` at a flat directory generated via
`translocate repackage` and it records each package at its exact
built version. During `uv sync --frozen`, a translocated-in package
installation that already matches the locked version will be left
as-is, and everything else resolves normally from PyPI or
other indices.

`translocate repackage --uv-config=-` prints the `pyproject.toml`
snippet that configures `uv` to treat its `--out` directory
as a flat index:

```toml
[[tool.uv.index]]
name = "translocated"
url = "file:///abs/path/to/wheels"
format = "flat"
explicit = true

[tool.uv.sources]
torch = { index = "translocated" }
```

`explicit = true` confines the index to the packages that name it
under `[tool.uv.sources]`. Every other dependency resolves from PyPI
as usual. Use `--uv-config=PATH` instead of `-` to write the snippet
to a file. The emitted `url` is the absolute path of `--out`.
uv also accepts a path relative to the project root, like
`url = "./wheels"`, which may travel better when the snippet is
committed, depending on your development setup.

Two ways to populate the index:

### Real wheels

```sh
translocate repackage torch --out ./wheels --mkdir --uv-config=-
```

Run this where the package is installed. For a base image, that
means inside a container. Carry `./wheels` to wherever you develop,
merge the snippet into `pyproject.toml`, and run `uv lock`. The
index holds installable wheels, so `uv sync` can populate a fresh
venv from it directly.

### Metadata-only stubs

```sh
translocate repackage torch --out ./stubs --mkdir --metadata-only --uv-config=-
```

`--metadata-only` writes stub wheels holding just dist-info, which typically
makes for a few KiB per wheel. `uv lock` reads dependency metadata from them
exactly as it would from real wheels. Installing one would yield an empty
package, so they're mainly useful when paired with `translocate clone`:
the stub gives the resolver something to lock against, the clone puts the
real files in the venv, and `uv sync --frozen` accepts the pair as the
locked version already installed.

> [!TIP]
> `--metadata-only` wheels are small enough to be checked into version
> control. When paired with `--compression-level=0`, they can even benefit
> from delta-compression on changes.

> [!NOTE]
> `--on-hash-mismatch` has no effect with `--metadata-only`. The stub
> path never compares files against the source RECORD.

### Building a container against the lockfile

With `pyproject.toml` and `uv.lock` in the build context, the image
build is:

```sh
uv venv /app/.venv
translocate clone my-package --target /app/.venv --link-mode symlink
uv sync --frozen
```

Symlink mode keeps the image small: the links resolve into the base
image's site-packages, which is always present at runtime. Clone
before sync: `uv sync --frozen` skips a package only when the venv
already holds the locked version. A frozen sync never consults the
index directory, so it does not need to exist in the image.
Regenerate it only when you need to run `uv lock` or `uv add` again,
in an environment that has the packages.

## Common flags

| flag                    | clone | repackage |
|-------------------------|-------|-----------|
| `--include-deps`        | yes   | yes       |
| `--strict-deps`         | yes   | yes       |
| `--deps-file <path\|->` | yes   | yes       |
| `--dry-run`             | yes   | yes       |
| `--target <prefix>`     | yes   | no        |
| `--link-mode <mode>`    | yes   | no        |
| `--smoke-test-imports`  | yes   | no        |
| `--out <dir>`           | no    | yes       |
| `--metadata-only`       | no    | yes       |
| `--uv-config <path\|->` | no    | yes       |

`--strict-deps` upgrades the default skip-and-warn for an
untranslocatable dep into a hard error, for callers who want
all-or-nothing.

`--dry-run` resolves packages, runs the up-front checks, and reports
what the command would write or clone. Side outputs requested by
flag, like `--deps-file` and `--uv-config`, are still written.

## Exit codes

| code | meaning                                                                       |
|------|-------------------------------------------------------------------------------|
| 0    | Operation completed and verification passed                                   |
| 1    | Unexpected error                                                              |
| 2    | Invalid CLI invocation                                                        |
| 10   | Distribution not found in source environment                                  |
| 11   | Distribution is editable or otherwise untranslocatable                        |
| 12   | Target is not a Python install, or source and target are the same interpreter |
| 13   | Target ABI is incompatible with the source                                    |
| 20   | Verification failed                                                           |
