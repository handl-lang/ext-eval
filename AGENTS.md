# ext-eval

`hnl-ext-eval` — `hnl run` / `hnl eval` as a dynamic `hnl` extension:
program evaluation packaged as a `libhnl_ext_eval.{so,dylib,dll}` cdylib
the dumb `hnl` host discovers at runtime.

## Layout

| path | role |
| --- | --- |
| `src/` | extension ABI exports + `run`/`eval` command wiring |
| `xtask/` | vendored gate crate (`fmt`, `clippy`, `test`, `doc`, `gate`) |
| `workflows/gate.yaml` | per-push gate — `handl-gates` rust campaign via bupha |

## Contract

- Commands: `hnl run [--engine <name>] <file.hnl>` (typecheck, optimize
  `~[N]~>` rules, evaluate) and `hnl eval '<expression>'`.
- `run` typechecks canonical-dialect sources through the embedded Handl
  model, strips host-module import declarations, applies the module's own
  `~[N]~>` rules (`n_dim_opt`), then evaluates. `--engine` selects the
  eval engine via `Kit::discover_with_eval` — a `run`-local flag.
- Exit codes: `0` ok, `1` eval/typecheck/kit errors, `2` usage/read.
- `src/transform_categorical.rs` is the canonical two-file
  (`rules.hnl` × `program.hnl`) runner also copied into `ext-transform`.

## Verification

```bash
cargo run --manifest-path xtask/Cargo.toml -- gate   # fmt → clippy → test → doc
```

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
