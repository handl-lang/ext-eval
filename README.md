# ext-eval

`hnl-ext-eval` — `hnl run` / `hnl eval` as a dynamic `hnl` extension.

An `ext-*` crate: program evaluation packaged as a
`libhnl_ext_eval.{so,dylib,dll}` shared library for the dumb `hnl` host. The
host discovers it via `--ext-dir`, `$HANDL_EXT_DIR`, `~/.handl/ext/`, or a dev
`target/debug/` next to the `hnl` binary.

## Commands

```text
hnl run [--engine <name>] <file.hnl>   — typecheck, optimize (~[N]~> rules), evaluate
hnl eval '<expression>'                — evaluate an expression
```

`run` typechecks canonical-dialect sources through the embedded Handl model,
strips host-module import declarations, applies the module's own `~[N]~>`
rules (`n_dim_opt`), then evaluates. `--engine <name>` selects the eval engine
(`Kit::discover_with_eval`); it is a `run`-local flag.

Exit codes: `0` ok, `1` eval/typecheck/kit errors, `2` usage/read errors.

## Development

```bash
cargo xtask gate        # fmt --check → clippy -D warnings → test → doc
```

`cargo xtask` wraps the repo's `xtask/` crate (`fmt`, `clippy`, `test`, `doc`,
`check`, `gate`); `ci.sh` is the gate entry. The crate builds a `cdylib` +
`rlib`; `cargo test` links the `rlib` statically.

```sh
cargo build        # target/debug/libhnl_ext_eval.so
HANDL_EXT_DIR=$PWD/target/debug ../host-cli/target/debug/hnl eval '1 + 2'
```

Sibling-repo deps resolve from sibling checkouts via the `[patch]` tables;
without them the `branch = "main"` git sources are used.

## License

MIT OR Apache-2.0.
