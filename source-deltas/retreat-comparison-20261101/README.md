# Training and evaluation source snapshot

This directory contains source files reached by a static import scan from the later training and evaluation scripts. It is provided for inspection; a complete end-to-end reproduction has not been verified.

Dynamic imports, execution-time scripts, datasets, model checkpoints, dependencies and assets may be missing. Some existing public files may differ despite sharing a name. Paths in the source reflect the development layout and are not a turnkey installation.

See [SOURCE_PROVENANCE.md](../../SOURCE_PROVENANCE.md) for the origin and verification status of the files and [CREDITS.md](../../CREDITS.md) for asset credits.

The remote-transfer helper and dataset collector were adapted to require `EYEDROP_SSH_HOST` and `EYEDROP_REMOTE_ROOT` instead of the original machine-specific settings. The remote root must include its trailing slash. These changes affect source hashes; the original collection and evaluation have not been rerun with the adapted helpers.

## Runtime source variants

`runtime-snapshots/` contains source versions found in the execution environment that differ from the workspace copies. Paths preserve the environment layout for comparison. The evaluation specified the search order `outputs/diagnostics/...`, `outputs/runtime/review-sklearn-v1/src`, `outputs/runtime/guarded-eval-person-20261029-01/scripts`, `scripts`, then `src`; an observer subprocess used a separate runtime source path. The snapshots were collected afterward and matching their current hashes does not prove which bytes were loaded at evaluation time. The full dynamic dependencies and required data remain unverified.
