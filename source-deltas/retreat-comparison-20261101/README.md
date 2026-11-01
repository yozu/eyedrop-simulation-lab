# Code review candidate

This directory is a local review candidate containing the source files reached by a static import scan from the latest training and evaluation scripts. It is not a complete or tested release.

Dynamic imports, execution-time scripts, datasets, model checkpoints, dependencies and assets may be missing. Some existing public files may differ despite sharing a name. Paths in the source reflect the development layout and are not a turnkey installation.

Review licenses, personal information, public history, completeness and runtime behavior before merging these files into the existing public repository. Keep the original project and previous publication intact.

The remote-transfer helper and dataset collector were adapted to require `EYEDROP_SSH_HOST` and `EYEDROP_REMOTE_ROOT` instead of the original machine-specific settings. The remote root must include its trailing slash. These changes affect source hashes; the original collection and evaluation have not been rerun with the adapted helpers.

## Runtime source variants

`runtime-snapshots/` contains source versions found in the execution environment that differ from the workspace copies. Paths preserve the environment layout for comparison. The evaluation specified the search order `outputs/diagnostics/...`, `outputs/runtime/review-sklearn-v1/src`, `outputs/runtime/guarded-eval-person-20261029-01/scripts`, `scripts`, then `src`; an observer subprocess used a separate runtime source path. The snapshots were collected afterward and matching their current hashes does not prove which bytes were loaded at evaluation time. The full dynamic dependencies and required data remain unverified.
