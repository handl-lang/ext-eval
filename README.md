# ext-eval — `hnl run` / `hnl eval` as a dynamic `hnl` extension

Program evaluation packaged as a `libhnl_ext_eval.{so,dylib,dll}` shared
library for the dumb `hnl` host (ADR-0149). The host discovers it via
`--ext-dir`, `$HANDL_EXT_DIR`, `~/.handl/ext/`, or a dev `target/debug/` next
to the `hnl` binary.

## Commands

```text
hnl run [--engine <name>] <file.hnl>   — typecheck, optimize (~[N]~> rules), evaluate
hnl eval '<expression>'                — evaluate an expression
```

`run` typechecks canonical-dialect sources through the embedded Handl model,
strips host-module import declarations, applies the module's own `~[N]~>`
rules (`n_dim_opt`), then evaluates — matching the pre-ADR-0149 `hnl run`
exactly. `--engine <name>` selects the eval engine (`Kit::discover_with_eval`);
it is a `run`-local flag now that the host owns no global flags.

Exit codes: `0` ok, `1` eval/typecheck/kit errors, `2` usage/read errors.

## Dev

```sh
cargo build        # target/debug/libhnl_ext_eval.so
cargo test         # unit tests link the rlib statically
HANDL_EXT_DIR=$PWD/target/debug ../host-cli/target/debug/hnl eval '1 + 2'
```

Sibling-repo deps resolve from sibling checkouts via the `[patch]` tables
(ADR-0128 §Resolution); without them the `branch = "main"` git sources are used.
