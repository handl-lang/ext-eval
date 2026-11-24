//! `hnl-ext-eval` — the `run`/`eval` extension for `hnl` (ADR-0149).
//!
//! Evaluation lives in the canonical kit (`handl-kit-canonical`'s
//! `Kit::eval`/`Runtime::eval` over the data-driven engine chain); this crate
//! is the *packaging*: it exports the ADR-0149 ABI (`handl_ext_manifest` /
//! `handl_ext_dispatch` / `handl_ext_free`) and implements the
//! `hnl run [--engine <name>] <file.hnl>` and `hnl eval '<expression>'`
//! command surfaces — byte-for-byte the semantics the built-in `run`/`eval`
//! arms had before the host went dumb.
//!
//! Dialect routing matches the old dispatch: canonical-dialect sources are
//! typechecked through the embedded Handl model (`Kit::typecheck_with_prelude`),
//! then the module's own `~[N]~>` rules optimize the AST (`n_dim_opt`,
//! [`transform_categorical`]) before eval; contract/arrow dialects skip the
//! canonical check, same as before.
//!
//! Note on `--engine`: the old host consumed it as a *global* flag
//! (`hnl --engine jit run f.hnl`); a dumb host owns no global flags, so it is
//! now a `run`-local flag (`hnl run --engine jit f.hnl`) — same option,
//! owned by the extension that uses it.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use handl_ext::{ExtRequest, ExtResponse, ExtensionManifest};
use handl_kit_pkg::KitProjectExt;
use handl_kit_prelude::Kit;
use handl_language::{AstRef, Language};
use handl_plugin_ext_detect::{Dialect, detect_dialect};
use serde_json::Value;

/// The `~[N]~>` categorical arrow optimizer — the same native-rewrite module
/// the built-in `run` arm linked (verbatim); `hnl run` optimizes a module's
/// own 1-cell `flow`s before eval. Its two-file `run`/`pipeline_value`
/// surface belongs to `ext-transform` and is unused here.
#[allow(dead_code)]
mod transform_categorical;

handl_ext::export!(
    ExtensionManifest::new("eval")
        .version(env!("CARGO_PKG_VERSION"))
        .command(
            "run",
            "run [--engine <name>] [--name <expr>] [--kit <dir>] <file.hnl> | run [--plan] [--cache|--no-cache] [--json] <script> [args…]",
        )
        .command("eval", "eval [--kit <dir>] '<expression>'"),
    dispatch
);

/// ABI dispatch: `run` executes a file, `eval` evaluates an expression.
fn dispatch(cmd: &str, req: &ExtRequest) -> ExtResponse {
    match cmd {
        "run" => run_command(&req.args, Path::new(&req.cwd)),
        "eval" => eval_command(&req.args, Path::new(&req.cwd)),
        other => ExtResponse::error(2, format!("eval: unknown command `{other}`\n")),
    }
}

// ---------------------------------------------------------------------------
// `run [--engine <name>] <file.hnl>`
// ---------------------------------------------------------------------------

/// Output accumulator: the extension renders one `ExtResponse` instead of
/// printing, so stdout/stderr collect here and the code is returned.
#[derive(Default)]
struct Out {
    stdout: String,
    stderr: String,
}

impl Out {
    fn err(&mut self, msg: impl std::fmt::Display) {
        self.stderr.push_str(&format!("{msg}\n"));
    }
}

/// `hnl run` — shape-dispatched (P-003):
///
/// - first positional resolving to an existing path (or carrying a
///   `.hnl` suffix) → file execution, unchanged;
/// - a bare name → `pkg.hnl` `scripts:` resolution via `handl-run`
///   (walk-up from cwd), with runner flags `--plan`/`--cache`/
///   `--no-cache`/`--json` recognized **only before the name** —
///   everything after it is verbatim script argv (`--` is an optional
///   npm-style separator);
/// - no name → list the discovered package's scripts (`npm run`), or
///   the usage error when no `pkg.hnl`/`scripts` exist.
fn run_command(args: &[String], cwd: &Path) -> ExtResponse {
    let mut engine: Option<String> = None;
    let mut name: Option<String> = None;
    let mut kit: Option<String> = None;
    let mut plan = false;
    let mut cache_flag = false;
    let mut cache_disabled = false;
    let mut json = false;
    let mut positional: Vec<String> = Vec::new();
    // Runner/file-exec flags are recognized only before the first
    // positional (the file or script name); `--` ends flag parsing.
    // Everything after the name is verbatim script argv.
    let mut flags_done = false;

    let mut i = 0;
    while i < args.len() {
        let t = args[i].as_str();
        if flags_done {
            positional.push(t.to_string());
        } else {
            match t {
                "--" => flags_done = true,
                "--plan" => plan = true,
                "--cache" => cache_flag = true,
                "--no-cache" => {
                    cache_flag = true;
                    cache_disabled = true;
                }
                "--json" => json = true,
                "--engine" => {
                    i += 1;
                    if i >= args.len() {
                        return err("run: --engine requires a value\n");
                    }
                    engine = Some(args[i].clone());
                }
                "--name" => {
                    i += 1;
                    if i >= args.len() {
                        return err("run: --name requires a value\n");
                    }
                    name = Some(args[i].clone());
                }
                "--kit" => {
                    i += 1;
                    if i >= args.len() {
                        return err("run: --kit requires a value\n");
                    }
                    kit = Some(args[i].clone());
                }
                s if s.starts_with("--engine=") => {
                    engine = Some(s["--engine=".len()..].to_string());
                }
                s if s.starts_with("--name=") => {
                    name = Some(s["--name=".len()..].to_string());
                }
                s if s.starts_with("--kit=") => {
                    kit = Some(s["--kit=".len()..].to_string());
                }
                // An unrecognized `--flag`/`-x` before the name is a
                // usage error — script args belong after the name.
                s if s.starts_with('-') && s.len() > 1 => {
                    return err(format!(
                        "run: unknown flag `{s}` before the script/file name (script args go after the name)\n"
                    ));
                }
                _ => {
                    flags_done = true;
                    positional.push(t.to_string());
                }
            }
        }
        i += 1;
    }

    let runner_flag_used = plan || json || cache_flag;
    let file_flags_used = engine.is_some() || name.is_some() || kit.is_some();

    let Some(first) = positional.first() else {
        // `hnl run` with no name: list the nearest pkg.hnl's scripts;
        // file-exec flags or no package at all keep today's usage error.
        if file_flags_used {
            return run_file(
                None,
                engine.as_deref(),
                name.as_deref(),
                kit.as_deref(),
                cwd,
            );
        }
        return match handl_run::PkgScripts::discover(cwd) {
            Some(pkg) if !pkg.scripts.is_empty() => {
                let mut out = Out::default();
                for line in handl_run::list_scripts(&pkg) {
                    out.stdout.push_str(&line);
                    out.stdout.push('\n');
                }
                code(0, out)
            }
            _ => {
                let mut out = Out::default();
                out.err("usage: hnl run <file.handl>");
                code(2, out)
            }
        };
    };

    // Name-vs-file precedence: an existing filesystem path always wins
    // (a `.hnl` suffix counts even when missing — the read error names
    // it); anything else is a script name.
    let disk = resolve(cwd, first);
    if first.ends_with(".hnl") || disk.is_file() {
        if runner_flag_used {
            return err(
                "run: --plan/--cache/--no-cache/--json apply to pkg.hnl script names, not file execution\n",
            );
        }
        return run_file(
            Some(first),
            engine.as_deref(),
            name.as_deref(),
            kit.as_deref(),
            cwd,
        );
    }

    run_script(
        first,
        &positional[1..],
        RunInvocation {
            plan,
            json,
            cache_disabled,
        },
        engine.is_some() || name.is_some() || kit.is_some(),
        cwd,
    )
}

/// The runner flags that survived header parsing for a script run.
struct RunInvocation {
    plan: bool,
    json: bool,
    cache_disabled: bool,
}

/// `hnl run <script> [args…]` — delegate to `handl-run`'s `Runner`.
/// Everything after the name is verbatim argv; a single leading `--`
/// is the optional npm-style separator and is stripped.
fn run_script(
    script: &str,
    args: &[String],
    inv: RunInvocation,
    file_flags_used: bool,
    cwd: &Path,
) -> ExtResponse {
    if file_flags_used {
        return err(
            "run: --engine/--name/--kit do not apply to script names (they are file-execution flags)\n",
        );
    }
    // Script-resolution failures are runtime errors (exit 1), not usage
    // errors — npm exits 1 for a missing script too.
    let fail = |msg: String| ExtResponse::error(1, msg);
    let mut script_args = args;
    if script_args.first().is_some_and(|a| a == "--") {
        script_args = &script_args[1..];
    }

    let pkg = match handl_run::PkgScripts::discover(cwd) {
        Some(p) => p,
        None => {
            return fail(format!(
                "run: no pkg.hnl found walking up from {} — cannot resolve script `{script}`\n",
                cwd.display()
            ));
        }
    };

    let options = handl_run::RunOptions {
        cache: if inv.cache_disabled {
            handl_run::CacheMode::Disabled
        } else {
            handl_run::CacheMode::Enabled
        },
        toolchain: run_toolchain(),
        depth: None,
        // Streams return through ExtResponse — keep the process streams
        // clean so `--json` output stays machine-readable.
        echo: false,
    };
    let runner = handl_run::Runner::with_options(options);

    if inv.plan {
        return match runner.plan(&pkg, script, script_args) {
            Ok(report) => plan_response(&report, inv.json),
            Err(e) => fail(format!("run: {e}\n")),
        };
    }
    match runner.run(&pkg, script, script_args) {
        Ok(report) => run_report_response(&report, inv.json),
        Err(e) => fail(format!("run: {e}\n")),
    }
}

/// The cache-key toolchain: `hnl` host version via the environment
/// ([`handl_run::Toolchain::detect`]) and this extension's own
/// `CARGO_PKG_VERSION` — the code that actually runs `.hnl` files.
fn run_toolchain() -> handl_run::Toolchain {
    let mut t = handl_run::Toolchain::detect();
    t.ext_eval = env!("CARGO_PKG_VERSION").to_string();
    t
}

/// Render a [`handl_run::RunReport`]: child streams to their own
/// channels, cache messages to stderr, exit code propagated. `--json`
/// emits the whole report as one JSON value on stdout instead.
fn run_report_response(report: &handl_run::RunReport, json: bool) -> ExtResponse {
    if json {
        let tasks: Vec<Value> = report
            .tasks
            .iter()
            .map(|t| {
                let (status, code) = match &t.status {
                    handl_run::TaskStatus::Ran { code } => ("ran", Some(*code)),
                    handl_run::TaskStatus::CacheHit { code } => ("cache_hit", Some(*code)),
                    handl_run::TaskStatus::Skipped => ("skipped", None),
                };
                serde_json::json!({
                    "name": t.name,
                    "status": status,
                    "code": code,
                    "miss_reason": t.miss_reason,
                })
            })
            .collect();
        let value = serde_json::json!({
            "script": report.script,
            "order": report.order,
            "tasks": tasks,
            "exit_code": report.exit_code,
            "messages": report.messages,
            "stdout": String::from_utf8_lossy(&report.stdout),
            "stderr": String::from_utf8_lossy(&report.stderr),
        });
        return ExtResponse {
            code: report.exit_code,
            stdout: format!(
                "{}\n",
                serde_json::to_string_pretty(&value).unwrap_or_default()
            ),
            stderr: String::new(),
        };
    }
    let mut out = Out::default();
    for m in &report.messages {
        out.err(m);
    }
    out.stderr
        .push_str(&String::from_utf8_lossy(&report.stderr));
    out.stdout
        .push_str(&String::from_utf8_lossy(&report.stdout));
    ExtResponse {
        code: report.exit_code,
        stdout: out.stdout,
        stderr: out.stderr,
    }
}

/// `--plan` output: per-task cache verdicts without executing —
/// text lines, or the [`handl_run::PlanReport`] as JSON under `--json`.
fn plan_response(report: &handl_run::PlanReport, json: bool) -> ExtResponse {
    if json {
        let tasks: Vec<Value> = report
            .tasks
            .iter()
            .map(|t| {
                let (verdict, reason) = match &t.verdict {
                    handl_run::PlanVerdict::Hit => ("hit", Value::Null),
                    handl_run::PlanVerdict::Miss(r) => ("miss", Value::from(r.clone())),
                    handl_run::PlanVerdict::Uncached => ("uncached", Value::Null),
                };
                serde_json::json!({
                    "name": t.name,
                    "verdict": verdict,
                    "reason": reason,
                })
            })
            .collect();
        let value = serde_json::json!({
            "script": report.script,
            "order": report.order,
            "tasks": tasks,
        });
        return ExtResponse {
            code: 0,
            stdout: format!(
                "{}\n",
                serde_json::to_string_pretty(&value).unwrap_or_default()
            ),
            stderr: String::new(),
        };
    }
    let mut out = Out::default();
    out.stdout.push_str(&format!(
        "plan for `{}` (order: {})\n",
        report.script,
        report.order.join(", ")
    ));
    for t in &report.tasks {
        let verdict = match &t.verdict {
            handl_run::PlanVerdict::Hit => "hit".to_string(),
            handl_run::PlanVerdict::Miss(r) => format!("miss: {r}"),
            handl_run::PlanVerdict::Uncached => "uncached".to_string(),
        };
        out.stdout.push_str(&format!("  {}: {verdict}\n", t.name));
    }
    code(0, out)
}

fn run_file(
    path: Option<&str>,
    engine: Option<&str>,
    name: Option<&str>,
    kit: Option<&str>,
    cwd: &Path,
) -> ExtResponse {
    let mut out = Out::default();
    let path = match path {
        Some(p) => p,
        None => {
            out.err("usage: hnl run <file.handl>");
            return code(2, out);
        }
    };

    let disk = resolve(cwd, path);
    let source = match std::fs::read_to_string(&disk) {
        Ok(s) => s,
        Err(e) => {
            out.err(format!("run: failed to read {path}: {e}"));
            return code(2, out);
        }
    };

    // `--kit <dir-or-name>`: resolve to a kit directory before building
    // the kit so its `libs` deps merge into the project deps below.
    // Without the flag, T-122 discovery applies: the nearest ancestor
    // `.handl.toml` of the target file declaring `kit = "<spec>"` wins,
    // falling back to `~/.config/handl/.handl.toml`; the spec resolves
    // against its manifest's directory exactly like a `--kit` value.
    let kit_dir = match kit {
        Some(v) => match resolve_kit_dir(cwd, v, "run", "--kit") {
            Ok(d) => Some(d),
            Err(msg) => {
                out.err(msg);
                return code(2, out);
            }
        },
        None => match handl_platform::discover_kit_spec(&disk) {
            Some(dk) => match resolve_kit_dir(&dk.manifest_dir, &dk.spec, "run", "kit") {
                Ok(d) => Some(d),
                Err(msg) => {
                    out.err(msg);
                    return code(2, out);
                }
            },
            None => None,
        },
    };

    // Project-aware kit: discover `.handl.toml` by walking up from the
    // source file's directory; its `[dependencies]` table feeds the module
    // loader and its directory becomes the module root for `import "path"`.
    // A `--kit` directory adds its `kit.hnl` `libs` (+ transitive `pkg.hnl`
    // deps) as fill-missing entries on top.
    let kit = match (engine, kit_dir.as_deref()) {
        (Some(name), Some(kd)) => Kit::discover_with_kit_and_eval(&disk, name, kd),
        (Some(name), None) => Kit::discover_with_eval(&disk, name),
        (None, Some(kd)) => Kit::discover_with_kit(&disk, kd),
        (None, None) => Kit::discover(&disk),
    };
    let kit = match kit {
        Ok(k) => k,
        Err(e) => {
            out.err(format!("run: {e}"));
            return code(1, out);
        }
    };

    // Resolution bases for `from "<spec>"` declaration lines, mirroring
    // `ModuleLoader` order: the entry file's directory, the module root,
    // then the process working directory. A `from` line whose specifier
    // resolves to a file under one of these is kept (it is the executable
    // import op); one that names a host module is stripped.
    let bases = spec_resolution_bases(&disk, kit.module_root());

    // Typecheck before running. Canonical-dialect sources go through the
    // embedded Handl model; non-canonical dialects (Contract/Arrow) are
    // checked by their own loaders, same routing as `hnl typecheck`.
    // Splice-form file imports (`import "<spec>"`, `from "<spec>" import *`)
    // bind the module's exports into scope at eval time — seed those names
    // so the scope check sees them.
    // The run pipeline's own dep view must see the same names the kit's
    // ModuleLoader does: project `[dependencies]` plus the selected
    // kit's `libs` (+ transitive `pkg.hnl` deps) as fill-missing entries
    // — otherwise `import "<kit-lib>/…"` lines can't be told apart from
    // host-module declarations for stripping/splice-prelude purposes.
    let mut deps = handl_platform::discover_dependencies(&disk).unwrap_or_default();
    // P-013 T-150: the consuming package's `pkg.hnl` dep scopes — the
    // scope-check prelude must seed exactly the declared surface, matching
    // what the `ModuleLoader` scope filter exposes at eval time.
    let dep_scopes = handl_platform::discover_dep_scopes(&disk);
    if let Some(kd) = &kit_dir
        && let Ok(kit_deps) = handl_platform::resolve_kit_dependencies(kd)
    {
        for (name, dir) in kit_deps {
            deps.entry(name).or_insert(dir);
        }
    }
    if detect_dialect(&source) == Dialect::Canonical {
        let spliced = spliced_import_names(&source, &bases, &deps, &dep_scopes);
        if let Err(e) = kit.typecheck_with_prelude(&source, &spliced) {
            out.err(format!("run: typecheck failed: {e}"));
            return code(1, out);
        }
    }

    let source = strip_import_declarations(&source, &bases, &deps);
    let wrapped = wrap_run_source(&source, name);

    // Parse, apply the module's own ~[2]~> rules, then eval the optimized AST.
    let ast = match kit.parse(&wrapped) {
        Ok(a) => a,
        Err(e) => {
            out.err(format!("run: parse failed: {e}"));
            return code(1, out);
        }
    };
    let lang = kit.runtime().lang();
    // Bound names = module bindings + the wire names the module chain
    // supplies at eval time (stripped `from "pkg" import ...` lines are
    // not in the AST, but their names are still literal references, not
    // pattern variables).
    let mut bound = transform_categorical::bound_names(&ast, lang);
    bound.extend(typecheck_wire_prelude().iter().cloned());
    let env = transform_categorical::dim_env(&ast, lang);
    let rules = transform_categorical::collect_rules(&ast, &bound, &env, lang);
    let protected = Arc::new(transform_categorical::dim_arrow_subtrees(&ast, lang));
    let engine = hnl_tool_transform_rewrite::RewriteEngine::new().with_native_registry(
        transform_categorical::native(rules, protected, Arc::new(env)),
    );
    let out_pass = match engine.run_pass(lang, &ast, AstRef::new(ast.len() - 1), "n_dim_opt") {
        Ok(o) => o,
        Err(e) => {
            out.err(format!("run: optimization pass failed: {e}"));
            return code(1, out);
        }
    };

    match kit.runtime().eval(&out_pass.ast) {
        Ok(mut value) => {
            out.stdout
                .push_str(&print_value(&mut value, lang, Some(wrapped.as_str())));
            code(0, out)
        }
        Err(e) => {
            out.err(format!("run: {e}"));
            code(1, out)
        }
    }
}

// ---------------------------------------------------------------------------
// `eval <expr>`
// ---------------------------------------------------------------------------

fn eval_command(args: &[String], cwd: &Path) -> ExtResponse {
    // `eval [--kit <dir>] '<expression>'` — only `--kit`/`--kit=` are
    // intercepted; every other token (including a leading `--`) stays part
    // of the expression, preserving the old first-arg-is-the-expr shape.
    let mut kit: Option<String> = None;
    let mut positional: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--kit" => {
                i += 1;
                if i >= args.len() {
                    return err("eval: --kit requires a value\n");
                }
                kit = Some(args[i].clone());
            }
            s if s.starts_with("--kit=") => {
                kit = Some(s["--kit=".len()..].to_string());
            }
            s => positional.push(s),
        }
        i += 1;
    }
    eval_expression(positional.first().copied(), kit.as_deref(), cwd)
}

fn eval_expression(expr: Option<&str>, kit: Option<&str>, cwd: &Path) -> ExtResponse {
    let mut out = Out::default();
    let expr = match expr {
        Some(e) => e,
        None => {
            out.err("usage: hnl eval [--kit <dir>] <expression>");
            return code(2, out);
        }
    };

    // T-122: no target file for `eval` — discovery walks up from the
    // request cwd, same as the project-manifest discovery below.
    let kit_dir = match kit {
        Some(v) => match resolve_kit_dir(cwd, v, "eval", "--kit") {
            Ok(d) => Some(d),
            Err(msg) => {
                out.err(msg);
                return code(2, out);
            }
        },
        None => match handl_platform::discover_kit_spec(cwd) {
            Some(dk) => match resolve_kit_dir(&dk.manifest_dir, &dk.spec, "eval", "kit") {
                Ok(d) => Some(d),
                Err(msg) => {
                    out.err(msg);
                    return code(2, out);
                }
            },
            None => None,
        },
    };

    // Project-aware kit: discover `.handl.toml` from the CWD upward so
    // `import "dep/..."` resolves against `[dependencies]`; `--kit` merges
    // a `kit.hnl` kit's `libs` (+ transitive `pkg.hnl` deps) on top.
    let kit = match kit_dir {
        Some(kd) => Kit::discover_with_kit(cwd, &kd),
        None => Kit::discover(cwd),
    };
    let kit = match kit {
        Ok(k) => k,
        Err(e) => {
            out.err(format!("eval: {e}"));
            return code(1, out);
        }
    };

    // The Pratt parser handles one top-level statement per call; wrap the
    // user expression in a `do` block so semicolon-separated commands such as
    // `from "transform" import * as transform; transform.map(...)` evaluate
    // as a sequence.
    let wrapped = format!("do {{ {} }}", expr);
    let lang = kit.runtime().lang();
    match kit.eval(&wrapped) {
        Ok(mut value) => {
            out.stdout
                .push_str(&print_value(&mut value, lang, Some(wrapped.as_str())));
            code(0, out)
        }
        Err(e) => {
            out.err(format!("eval: {e}"));
            code(1, out)
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The process-wide unbound-wire prelude: the names the canonical kit's
/// module chain supplies at eval time (ADR-0096).
fn typecheck_wire_prelude() -> &'static std::collections::HashSet<String> {
    static PRELUDE: std::sync::OnceLock<std::collections::HashSet<String>> =
        std::sync::OnceLock::new();
    PRELUDE.get_or_init(handl_kit_prelude::prelude_wire_names)
}

/// Resolve `path` (possibly relative) against the request's `cwd`.
fn resolve(cwd: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

/// Resolve a `--kit` value — or a `.handl.toml` `kit =` spec — to a kit
/// directory: the value itself (a path to the kit repo root or its
/// `kit.hnl`, relative to `base`), then — for a bare name with no path
/// separator — the workspace conventions `base/kit-<name>` and
/// `base/../kit-<name>` (kits are `kit-<name>` sibling checkouts, e.g.
/// `--kit web-server` → `kit-web-server`). `base` is the request cwd for
/// an explicit `--kit` and the declaring manifest's directory for a
/// discovered spec.
///
/// `cmd` prefixes the error (`run:`/`eval:`); `via` names the source in
/// the message (`--kit` or `kit`). Returns `Err` when no candidate
/// contains a `kit.hnl`.
fn resolve_kit_dir(base: &Path, value: &str, cmd: &str, via: &str) -> Result<PathBuf, String> {
    let is_kit = |dir: &Path| dir.join("kit.hnl").is_file();
    let direct = resolve(base, value);
    if is_kit(&direct) || (direct.is_file() && direct.file_name().is_some_and(|n| n == "kit.hnl")) {
        return Ok(direct);
    }
    if !value.contains('/') && !value.contains('\\') {
        let mut bases = vec![base.to_path_buf()];
        if let Some(parent) = base.parent() {
            bases.push(parent.to_path_buf());
        }
        for base in bases {
            let candidate = base.join(format!("kit-{value}"));
            if is_kit(&candidate) {
                return Ok(candidate);
            }
        }
    }
    Err(format!(
        "{cmd}: {via} {value}: no kit.hnl found (expected a kit directory or a sibling kit-<name> checkout)"
    ))
}

fn err(msg: impl Into<String>) -> ExtResponse {
    ExtResponse::error(2, msg.into())
}

/// Wrap `source` in `do { … }` for `hnl run`. With `--name <expr>` the
/// block's last statement becomes the named expression instead of the
/// file's own tail — exports and bindings evaluate first, then `name`
/// resolves in that scope. A `do`-leading source can't be re-wrapped
/// (the inner block's scope would swallow its bindings), so `name` is
/// spliced before the matching close of the leading `do {…}`.
fn wrap_run_source(source: &str, name: Option<&str>) -> String {
    let trimmed = source.trim_start();
    let starts_do = trimmed.starts_with("do");
    match name {
        None if starts_do => source.to_string(),
        None => format!("do {{\n{}\n}}", source),
        Some(n) => {
            if starts_do && let Some(close) = do_block_close(trimmed) {
                return format!("{}\n{}\n{}", &trimmed[..close], n, &trimmed[close..]);
            }
            format!("do {{\n{}\n{}\n}}", source, n)
        }
    }
}

/// Position of the `}` that closes the `do {…}` block `src` opens with
/// (`src` must start with `do`). Braces inside `"`/`'` strings and
/// `--` line comments don't count. `None` when unmatched.
fn do_block_close(src: &str) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut i = 2; // skip `do`
    while i < bytes.len() && bytes[i] != b'{' {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    let mut depth = 0usize;
    let mut j = i;
    while j < bytes.len() {
        match bytes[j] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(j);
                }
            }
            b'"' | b'\'' => {
                let quote = bytes[j];
                j += 1;
                while j < bytes.len() && bytes[j] != quote {
                    j += if bytes[j] == b'\\' { 2 } else { 1 };
                }
            }
            b'-' if bytes.get(j + 1) == Some(&b'-') => {
                while j < bytes.len() && bytes[j] != b'\n' {
                    j += 1;
                }
                continue;
            }
            _ => {}
        }
        j += 1;
    }
    None
}

fn code(c: i32, out: Out) -> ExtResponse {
    ExtResponse {
        code: c,
        stdout: out.stdout,
        stderr: out.stderr,
    }
}
fn strip_import_declarations(
    src: &str,
    bases: &[PathBuf],
    deps: &std::collections::HashMap<String, PathBuf>,
) -> String {
    src.lines()
        .filter(|line| !is_import_line(line.trim_start(), bases, deps))
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_import_line(
    trimmed: &str,
    bases: &[PathBuf],
    deps: &std::collections::HashMap<String, PathBuf>,
) -> bool {
    if let Some(rest) = trimmed.strip_prefix("import") {
        if rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\t') {
            // `import "spec"` / `import 'spec'` is the executable form —
            // keep it. `import <name> ...` (e.g. `import insert from
            // "collections/vec"`) is a module declaration — strip it.
            let inner = rest.trim_start();
            return !inner.starts_with('"') && !inner.starts_with('\'');
        }
        // `import` directly followed by a non-space char is an identifier
        // (`imported_and`) or the quoted executable form (`import"x"` —
        // kept either way): never a strippable declaration.
        return false;
    }
    if let Some(rest) = trimmed.strip_prefix("from")
        && (rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\t'))
    {
        // Strip iff the specifier names a host module — i.e. resolves to
        // neither a file under `bases` nor a `[dependencies]` package
        // (a dep-backed `from` is an executable import op and must be
        // kept, exactly like a relative-path one). An unparseable line
        // is kept so the parser reports the error.
        return match first_quoted_spec(rest) {
            Some(spec) => !spec_resolves_to_file(spec, bases, deps),
            None => false,
        };
    }
    false
}

/// Whether `<spec>` resolves to an existing file — first as a
/// `.handl.toml` `[dependencies]` specifier (a dep name shadows a
/// same-named relative directory, mirroring `ModuleLoader::resolve_path`),
/// then under any of `bases` (checked in order — mirroring `ModuleLoader`
/// relative resolution: importing module's directory, module root, working
/// directory).
///
/// A directory is treated as a package if it contains an `index.hnl` or
/// a `<dir>/<name>.hnl` main file; a bare name without `.hnl` is tried
/// with `.hnl` appended.
fn spec_resolves_to_file(
    spec: &str,
    bases: &[PathBuf],
    deps: &std::collections::HashMap<String, PathBuf>,
) -> bool {
    resolve_spec_path(spec, bases, deps).is_some()
}

/// Resolution bases for `import "<spec>"` specifiers in a file at `path`,
/// mirroring `ModuleLoader` order: the file's directory, the module root,
/// then the process working directory. Used by `run`, `typecheck`, and
/// `lint` alike so each command resolves imports the way eval will.
fn spec_resolution_bases(path: &Path, module_root: Option<&Path>) -> Vec<PathBuf> {
    let mut bases: Vec<PathBuf> = Vec::new();
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => bases.push(dir.to_path_buf()),
        _ => bases.push(PathBuf::from(".")),
    }
    if let Some(root) = module_root
        && !bases.iter().any(|b| b.as_path() == root)
    {
        bases.push(root.to_path_buf());
    }
    if let Ok(cwd) = std::env::current_dir()
        && !bases.contains(&cwd)
    {
        bases.push(cwd);
    }
    bases
}

/// Resolve a `[dependencies]` specifier: when the first `/`-separated
/// segment of `spec` names a declared dependency, the remainder resolves
/// inside that package's directory (`dep/lib/file` →
/// `<deps["dep"]>/lib/file`; a bare `dep` name resolves the package dir's
/// `index.hnl`/`<name>.hnl` main file). Returns `None` for non-dep specs.
fn resolve_dep_spec_path(
    spec: &str,
    deps: &std::collections::HashMap<String, PathBuf>,
) -> Option<PathBuf> {
    let (first, rest) = match spec.split_once('/') {
        Some((f, r)) => (f, Some(r)),
        None => (spec, None),
    };
    let dep_dir = deps.get(first)?;
    let joined = match rest {
        Some(r) if !r.is_empty() => dep_dir.join(r),
        _ => dep_dir.clone(),
    };
    file_candidate(&joined, spec)
}

/// Resolve `<spec>` to the file `ModuleLoader` would load: dependency-name
/// resolution first, then `bases` (same order as [`spec_resolves_to_file`]).
fn resolve_spec_path(
    spec: &str,
    bases: &[PathBuf],
    deps: &std::collections::HashMap<String, PathBuf>,
) -> Option<PathBuf> {
    if Path::new(spec).is_absolute() {
        return file_candidate(Path::new(spec), spec);
    }
    resolve_dep_spec_path(spec, deps).or_else(|| {
        bases
            .iter()
            .find_map(|base| file_candidate(&base.join(spec), spec))
    })
}

/// The specifier of a *splice-form* import — `import "<spec>"` or
/// `from "<spec>" import *` with no alias — whose exports land directly in
/// the importer's scope at eval time. Named/aliased `from` forms bind
/// statically known names and need no spliced scope.
fn splice_import_spec(trimmed: &str) -> Option<String> {
    if let Some(rest) = trimmed.strip_prefix("import") {
        if !rest.starts_with(' ') && !rest.starts_with('\t') {
            return None;
        }
        let inner = rest.trim_start();
        if !inner.starts_with('"') && !inner.starts_with('\'') {
            return None;
        }
        return first_quoted_spec(inner).map(str::to_string);
    }
    let rest = trimmed.strip_prefix("from")?;
    if !rest.starts_with(' ') && !rest.starts_with('\t') {
        return None;
    }
    // `from "<spec>" <tail>` — splice iff the tail is `import *` exactly.
    let bytes = rest.as_bytes();
    let q = bytes.iter().position(|b| *b == b'"' || *b == b'\'')?;
    let quote = bytes[q] as char;
    let after_q = &rest[q + 1..];
    let end = after_q.find(quote)?;
    let spec = &after_q[..end];
    let tail = after_q[end + quote.len_utf8()..].trim_start();
    let tail = tail.strip_prefix("import")?.trim_start();
    let tail = tail
        .split("--")
        .next()
        .unwrap_or("")
        .trim_end()
        .trim_end_matches(';')
        .trim_end();
    (tail == "*").then(|| spec.to_string())
}

/// Names a file's splice-form imports bind into scope at eval time: for
/// each resolvable `import "<spec>"` / `from "<spec>" import *`, the
/// module's declared exports (collected without evaluating it). The
/// bootstrap typechecker's module scope is source-only, so `hnl run`,
/// `hnl typecheck`, and `hnl lint` seed these into their scope-check
/// prelude — otherwise every use of a spliced export is reported as
/// `unbound wire` even though eval binds it.
fn spliced_import_names(
    src: &str,
    bases: &[PathBuf],
    deps: &std::collections::HashMap<String, PathBuf>,
    scopes: &std::collections::HashMap<String, handl_platform::PkgDepScope>,
) -> HashSet<String> {
    let mut names = HashSet::new();
    for line in src.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("--") {
            continue;
        }
        let Some(spec) = splice_import_spec(trimmed) else {
            continue;
        };
        let Some(path) = resolve_spec_path(&spec, bases, deps) else {
            continue;
        };
        let Ok(mod_src) = std::fs::read_to_string(&path) else {
            continue;
        };
        let mut exported = handl_runtime::module_prep::module_export_names(&mod_src);
        // P-013 T-150: a dep-scoped specifier seeds only the declared
        // `module.field` surface — a scoped-out export stays unbound.
        if let Some((scope, _, module)) = handl_platform::dep_spec_scope(scopes, &spec) {
            if !scope.allows_module(&module) {
                continue;
            }
            exported.retain(|n| scope.allows_field(&module, n));
        }
        names.extend(exported);
    }
    names
}

fn file_candidate(joined: &Path, spec: &str) -> Option<PathBuf> {
    if joined.is_dir() {
        let index = joined.join("index.hnl");
        if index.is_file() {
            return Some(index);
        }
        if let Some(name) = joined.file_name().and_then(|s| s.to_str()) {
            let main = joined.join(format!("{}.hnl", name));
            if main.is_file() {
                return Some(main);
            }
        }
    }
    if !spec.ends_with(".hnl") {
        let with_ext = joined.with_extension("hnl");
        if with_ext.is_file() {
            return Some(with_ext);
        }
    }
    if joined.is_file() {
        Some(joined.to_path_buf())
    } else {
        None
    }
}

/// Extract the first `'`- or `"`-quoted string literal's contents from a
/// source fragment (e.g. the specifier in `from "..." import ...`).
fn first_quoted_spec(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == b'"' || b == b'\'')?;
    let quote = bytes[start];
    let rest = &s[start + 1..];
    let end = rest.as_bytes().iter().position(|&b| b == quote)?;
    Some(&rest[..end])
}
fn print_value(value: &mut Value, lang: &Language, source: Option<&str>) -> String {
    // ADR-0179: tensors with `numel >= TENSOR_BIN_THRESHOLD` cross the
    // engine boundary as `{"__tensor__": {…, "data": {"__tensor_bin__":
    // "<base64 LE>"}}}` — expand them back to the array `data` spelling
    // before printing, the same egress step `hnl-run` performs, so
    // `hnl run`/`hnl eval` stdout consumers always see `data` as a list.
    handl_value::materialize_tensor_bin(value);
    if let Some(s) = value.as_str() {
        format!("{s}\n")
    } else if let Some(s) = arrow_display(value, lang, source, None, 0) {
        format!("{s}\n")
    } else {
        format!(
            "{}\n",
            serde_json::to_string_pretty(&compact_arrows(value, lang, source)).unwrap_or_default()
        )
    }
}

/// Recursively replace arrow values nested inside records/arrays with
/// their compact term display, so `hnl run` output stays readable.
fn compact_arrows(value: &Value, lang: &Language, source: Option<&str>) -> Value {
    if let Some(s) = arrow_display(value, lang, source, None, 0) {
        return Value::String(s);
    }
    // A serialized `Term` is a single-key JSON object tagged by its
    // combinator (`{"Op": …}`, `{"Seq": …}`…). Module records embed them
    // under `binds`/`body_term`; printing the raw JSON dumps the full
    // program — collapse to a marker like the arrow collapse.
    if let Value::Object(o) = value
        && o.len() == 1
    {
        let k = o.keys().next().unwrap().as_str();
        if matches!(
            k,
            "Op" | "Seq"
                | "Choice"
                | "First"
                | "Loop"
                | "App"
                | "Arr"
                | "Const"
                | "Handle"
                | "Let"
                | "Lit"
        ) {
            return Value::String("<term>".to_string());
        }
    }
    match value {
        Value::Object(o) => Value::Object(
            o.iter()
                .map(|(k, v)| (k.clone(), compact_arrows(v, lang, source)))
                .collect(),
        ),
        Value::Array(a) => {
            Value::Array(a.iter().map(|v| compact_arrows(v, lang, source)).collect())
        }
        _ => value.clone(),
    }
}

/// Compact display for an arrow value.
///
/// - A composite built by `>>>` records its components in `binds` as
///   `__left__`/`__right__` and prints as `left >>> right`. Components that
///   were unbound names evaluate to their name string (first-class name
///   references), so `fetch >>> serialize >>> parse` prints as written.
/// - A graded arrow (`{"__tag__": "Arrow", "grade": N, "inputs", "output"}`)
///   prints as `(inputs) ~[N]~> (output)`.
/// - A leaf arrow prints as the name it is bound under when `scope` still
///   holds that name for the same definition, otherwise as
///   `params ~> body` rendered from `body_def` (source text when spans are
///   available, else the data-driven formatter).
fn arrow_display(
    value: &Value,
    lang: &Language,
    source: Option<&str>,
    scope: Option<&serde_json::Map<String, Value>>,
    depth: usize,
) -> Option<String> {
    if depth > 16 {
        return Some("...".to_string());
    }
    // First-class name references: unbound vars evaluate to their name string.
    if let Value::String(s) = value {
        return Some(s.clone());
    }
    let obj = value.as_object()?;
    let is_plain = obj.get("__arrow__").and_then(|v| v.as_bool()) == Some(true);
    let is_graded = obj.get("__tag__").and_then(|v| v.as_str()) == Some("Arrow")
        && obj.get("grade").is_some()
        && obj.get("inputs").is_some()
        && obj.get("output").is_some();
    if !is_plain && !is_graded {
        return None;
    }
    // `binds` holds the arrow's residual named values (ADR-0106) — the
    // names a composite or recursive arrow still refers to.
    let binds_storage = obj.get("binds").and_then(|b| b.as_object());
    let env = binds_storage;
    // A name bound in scope for this same definition wins over expansion.
    for e in scope.into_iter().chain(env.iter().copied()) {
        for (name, val) in e {
            if !name.starts_with("__") && def_eq(val, value, 0) {
                return Some(name.clone());
            }
        }
    }
    if is_graded {
        // Graded arrows carry no binds of their own; the evaluated `inputs`
        // side does — use it so named arrows on either side print by name.
        let inputs_env_storage = obj
            .get("inputs")
            .and_then(|i| i.get("binds"))
            .and_then(|b| b.as_object());
        let inner_scope = env.or(inputs_env_storage);
        // The `dim` slot is a normal expression — `~[a >>> b]~>`, `~[N]~>`,
        // `~[f(x)]~>` — so the grade may be any value, not just a literal.
        let grade = term_display(
            obj.get("grade").unwrap(),
            lang,
            source,
            inner_scope,
            depth + 1,
        );
        let i = term_display(&obj["inputs"], lang, source, inner_scope, depth + 1);
        let o = term_display(&obj["output"], lang, source, inner_scope, depth + 1);
        return Some(format!(
            "{} ~[{grade}]~> {}",
            paren_if_term(&i),
            paren_if_term(&o)
        ));
    }
    let left = env.and_then(|e| e.get("__left__"));
    let right = env.and_then(|e| e.get("__right__"));
    if let (Some(l), Some(r)) = (left, right) {
        let ls = term_display(l, lang, source, env, depth + 1);
        let rs_raw = term_display(r, lang, source, env, depth + 1);
        return Some(format!("{ls} >>> {}", paren_if_term(&rs_raw)));
    }
    // Leaf arrow: `params ~> body`.
    let params = obj
        .get("params")
        .and_then(|p| p.as_array())
        .map(|p| {
            p.iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    if let Some(def) = obj.get("body_def").and_then(handl_language::as_ast)
        && !def.is_empty()
    {
        let root = AstRef::new(def.len() - 1);
        let body = source
            .zip(def.at(root).ok().and_then(|n| n.span))
            .and_then(|(src, span)| src.get(span.start..span.end))
            .map(|s| s.trim().to_string())
            .or_else(|| {
                hnl_tool_fmt::FormatEngine::new()
                    .format(lang, &def, root)
                    .ok()
                    .map(|s| s.trim().to_string())
            });
        if let Some(b) = body {
            return Some(format!("{params} ~> {b}"));
        }
    }
    Some("<arrow>".to_string())
}

/// Display a term-position value: arrows get [`arrow_display`]; a bare
/// string is a name reference and prints raw; other values resolve to a
/// bound name via the captured scope (e.g. `a` for `2` in `a >>> b`), or
/// fall back to a compact literal.
fn term_display(
    value: &Value,
    lang: &Language,
    source: Option<&str>,
    scope: Option<&serde_json::Map<String, Value>>,
    depth: usize,
) -> String {
    if let Some(s) = arrow_display(value, lang, source, scope, depth) {
        return s;
    }
    if let Value::String(s) = value {
        return s.clone();
    }
    if let Some(env) = scope {
        for (name, val) in env {
            if !name.starts_with("__") && def_eq(val, value, 0) {
                return name.clone();
            }
        }
    }
    match value {
        Value::Null => "null".to_string(),
        v => serde_json::to_string(v).unwrap_or_else(|_| "<value>".to_string()),
    }
}

/// Parenthesize a displayed subterm when it is a composite or arrow literal
/// (contains `>>>` or `~>`); names and atoms print bare.
fn paren_if_term(s: &str) -> String {
    if s.contains(">>>") || s.contains("~>") {
        format!("({s})")
    } else {
        s.to_string()
    }
}

/// Structural equality of arrow definitions, ignoring `binds` (context,
/// not identity) except for the `__left__`/`__right__` composite components,
/// which are compared recursively.
fn def_eq(a: &Value, b: &Value, depth: usize) -> bool {
    if depth > 16 {
        return true;
    }
    match (a, b) {
        (Value::Object(ao), Value::Object(bo)) => {
            for (k, av) in ao {
                if k == "binds" || k == "span" {
                    continue;
                }
                if !bo.get(k).is_some_and(|bv| def_eq(av, bv, depth + 1)) {
                    return false;
                }
            }
            for k in bo.keys() {
                if k != "binds" && k != "span" && !ao.contains_key(k) {
                    return false;
                }
            }
            let aenv = ao.get("binds").and_then(|b| b.as_object());
            let benv = bo.get("binds").and_then(|b| b.as_object());
            let al = aenv.as_ref().and_then(|e| e.get("__left__"));
            let ar = aenv.as_ref().and_then(|e| e.get("__right__"));
            let bl = benv.as_ref().and_then(|e| e.get("__left__"));
            let br = benv.as_ref().and_then(|e| e.get("__right__"));
            (al.is_none() && bl.is_none()
                || def_eq(
                    al.unwrap_or(&Value::Null),
                    bl.unwrap_or(&Value::Null),
                    depth + 1,
                ))
                && (ar.is_none() && br.is_none()
                    || def_eq(
                        ar.unwrap_or(&Value::Null),
                        br.unwrap_or(&Value::Null),
                        depth + 1,
                    ))
        }
        (Value::Array(aa), Value::Array(ba)) => {
            aa.len() == ba.len()
                && aa
                    .iter()
                    .zip(ba.iter())
                    .all(|(av, bv)| def_eq(av, bv, depth + 1))
        }
        _ => a == b,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    /// A unique temp dir per test.
    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hnl-ext-eval-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn run_without_a_file_is_usage_error() {
        let res = run_command(&[], Path::new("."));
        assert_eq!(res.code, 2);
        assert!(res.stderr.contains("usage: hnl run <file.handl>"));
    }

    #[test]
    fn eval_without_an_expression_is_usage_error() {
        let res = eval_command(&[], Path::new("."));
        assert_eq!(res.code, 2);
        assert!(res.stderr.contains("usage: hnl eval"));
    }

    #[test]
    fn run_missing_file_is_read_error() {
        let dir = tmpdir("missing");
        let res = run_command(&["nope.hnl".into()], &dir);
        assert_eq!(res.code, 2);
        assert!(res.stderr.contains("run: failed to read"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn eval_evaluates_an_expression() {
        let dir = tmpdir("eval");
        let res = eval_command(&["1 + 2".into()], &dir);
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert_eq!(res.stdout, "3\n");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_evaluates_a_file() {
        let dir = tmpdir("run");
        let file = dir.join("main.hnl");
        std::fs::write(&file, "40 + 2").unwrap();
        let res = run_command(&[file.to_string_lossy().into_owned()], &dir);
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert_eq!(res.stdout, "42\n");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_binds_export_decls_for_later_code() {
        let dir = tmpdir("export");
        let file = dir.join("main.hnl");
        std::fs::write(&file, "export x := 41\nexport y := x + 1\ny").unwrap();
        let res = run_command(&[file.to_string_lossy().into_owned()], &dir);
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert_eq!(res.stdout, "42\n");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_name_evaluates_the_named_binding() {
        let dir = tmpdir("name");
        let file = dir.join("main.hnl");
        // The block tail (`x + y` = 3) is ignored: `--name y` wins.
        std::fs::write(&file, "export x := 1\nexport y := x + 41\nx + y").unwrap();
        let res = run_command(
            &[
                "--name".into(),
                "y".into(),
                file.to_string_lossy().into_owned(),
            ],
            &dir,
        );
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert_eq!(res.stdout, "42\n");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_name_splices_into_a_do_leading_file() {
        let dir = tmpdir("name-do");
        let file = dir.join("main.hnl");
        // (an `export` inside `do {…}` is a separate checker gap —
        // unbound wire — so the splice is exercised with a plain
        // binding here; top-level exports are covered above.)
        std::fs::write(&file, "do {\n  x := 41\n  x + 1\n}").unwrap();
        let res = run_command(
            &[
                "--name".into(),
                "x".into(),
                file.to_string_lossy().into_owned(),
            ],
            &dir,
        );
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert_eq!(res.stdout, "41\n");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_kit_requires_a_value() {
        let res = run_command(&["--kit".into()], Path::new("."));
        assert_eq!(res.code, 2);
        assert!(res.stderr.contains("run: --kit requires a value"));
    }

    #[test]
    fn eval_kit_requires_a_value() {
        let res = eval_command(&["--kit".into()], Path::new("."));
        assert_eq!(res.code, 2);
        assert!(res.stderr.contains("eval: --kit requires a value"));
    }

    #[test]
    fn run_missing_kit_dir_is_usage_error() {
        let dir = tmpdir("kit-missing");
        let file = dir.join("main.hnl");
        std::fs::write(&file, "1").unwrap();
        let res = run_command(
            &[
                "--kit".into(),
                "no-such-kit".into(),
                file.to_string_lossy().into_owned(),
            ],
            &dir,
        );
        assert_eq!(res.code, 2);
        assert!(res.stderr.contains("no kit.hnl"), "stderr: {}", res.stderr);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn eval_missing_kit_dir_is_usage_error() {
        let dir = tmpdir("eval-kit-missing");
        let res = eval_command(
            &["--kit".into(), "no-such-kit".into(), "1 + 2".into()],
            &dir,
        );
        assert_eq!(res.code, 2);
        assert!(res.stderr.contains("no kit.hnl"), "stderr: {}", res.stderr);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `resolve_kit_dir`: a path to the kit dir (or its `kit.hnl`) wins;
    /// a bare name probes `cwd/kit-<name>` then `cwd/../kit-<name>`.
    #[test]
    fn resolve_kit_dir_path_and_name_forms() {
        let dir = tmpdir("kit-name");
        let kit = dir.join("kit-demo");
        std::fs::create_dir_all(&kit).unwrap();
        std::fs::write(kit.join("kit.hnl"), "{ libs: [] }\n").unwrap();
        let proj = dir.join("proj");
        std::fs::create_dir_all(&proj).unwrap();

        // Path form (relative to cwd) and the kit.hnl file itself.
        assert_eq!(
            resolve_kit_dir(&dir, "kit-demo", "run", "--kit").unwrap(),
            kit
        );
        assert_eq!(
            resolve_kit_dir(&dir, "kit-demo/kit.hnl", "run", "--kit").unwrap(),
            kit.join("kit.hnl")
        );
        // Bare name: sibling probe from cwd and from a child dir.
        assert_eq!(resolve_kit_dir(&dir, "demo", "run", "--kit").unwrap(), kit);
        assert_eq!(resolve_kit_dir(&proj, "demo", "run", "--kit").unwrap(), kit);
        // Names containing a separator skip the sibling probe.
        assert!(resolve_kit_dir(&dir, "sub/dir", "run", "--kit").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// T-122: `hnl run` picks up the `kit =` of the nearest ancestor
    /// `.handl.toml` when `--kit` is absent — the kit's `libs` deps merge
    /// into the project deps, and an explicit `--kit` still wins.
    #[test]
    fn run_discovers_kit_from_handl_toml() {
        let dir = tmpdir("kit-disc");
        let kit = dir.join("kit-web");
        let lib_js = dir.join("lib-js");
        std::fs::create_dir_all(&kit).unwrap();
        std::fs::create_dir_all(&lib_js).unwrap();
        std::fs::write(kit.join("kit.hnl"), "{ libs: [\"js\"] }\n").unwrap();
        // `lib-js` exports a binding the file uses; without the kit the
        // import would not resolve. `export` (not `::`) so the spliced
        // name reaches the typecheck prelude.
        std::fs::write(lib_js.join("answer.hnl"), "export answer := 42\n").unwrap();

        let proj = dir.join("proj");
        let nested = proj.join("src/deep");
        std::fs::create_dir_all(&nested).unwrap();
        // Project manifest two levels up declares the kit by bare *name* —
        // the sibling `kit-<name>` probe anchors at the manifest dir
        // (`proj/../kit-web`).
        std::fs::write(proj.join(".handl.toml"), "kit = \"web\"\n").unwrap();
        let file = nested.join("main.hnl");
        std::fs::write(&file, "import \"js/answer.hnl\"\nanswer").unwrap();

        // Discovered kit: `import "js/answer.hnl"` resolves through the
        // kit's libs.
        let res = run_command(&[file.to_string_lossy().into_owned()], &dir);
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert_eq!(res.stdout, "42\n");

        // `--kit` overrides discovery: point it at an empty kit and the
        // `js` dep disappears → the import fails.
        let empty = dir.join("kit-empty");
        std::fs::create_dir_all(&empty).unwrap();
        std::fs::write(empty.join("kit.hnl"), "{ libs: [] }\n").unwrap();
        let res = run_command(
            &[
                "--kit".into(),
                empty.to_string_lossy().into_owned(),
                file.to_string_lossy().into_owned(),
            ],
            &dir,
        );
        assert_ne!(res.code, 0, "expected failure without the kit dep");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// T-122: `hnl eval` discovers `kit =` walking up from the cwd.
    #[test]
    fn eval_discovers_kit_from_handl_toml() {
        let dir = tmpdir("eval-kit-disc");
        let kit = dir.join("kit-x");
        let lib_x = dir.join("lib-x");
        std::fs::create_dir_all(&kit).unwrap();
        std::fs::create_dir_all(&lib_x).unwrap();
        std::fs::write(kit.join("kit.hnl"), "{ libs: [\"x\"] }\n").unwrap();
        std::fs::write(lib_x.join("answer.hnl"), "answer := 42\n").unwrap();
        let proj = dir.join("proj");
        let nested = proj.join("sub");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(proj.join(".handl.toml"), "kit = \"../kit-x\"\n").unwrap();

        // From a nested cwd the walk finds the project manifest.
        let res = eval_command(&["import \"x/answer.hnl\"\nanswer".into()], &nested);
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert_eq!(res.stdout, "42\n");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A discovered `kit =` spec that resolves to no `kit.hnl` fails
    /// like an explicit `--kit` does.
    #[test]
    fn run_missing_discovered_kit_is_error() {
        let dir = tmpdir("kit-disc-missing");
        let proj = dir.join("proj");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::write(proj.join(".handl.toml"), "kit = \"no-such-kit\"\n").unwrap();
        let file = proj.join("main.hnl");
        std::fs::write(&file, "1").unwrap();
        let res = run_command(&[file.to_string_lossy().into_owned()], &dir);
        assert_eq!(res.code, 2);
        assert!(res.stderr.contains("no kit.hnl"), "stderr: {}", res.stderr);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn eval_with_kit_flag_evaluates() {
        let dir = tmpdir("eval-kit");
        let kit = dir.join("kit-x");
        std::fs::create_dir_all(&kit).unwrap();
        std::fs::write(kit.join("kit.hnl"), "{ libs: [] }\n").unwrap();
        let res = eval_command(
            &[
                "--kit".into(),
                kit.to_string_lossy().into_owned(),
                "1 + 2".into(),
            ],
            &dir,
        );
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert_eq!(res.stdout, "3\n");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wrap_run_source_variants() {
        assert_eq!(wrap_run_source("1 + 2", None), "do {\n1 + 2\n}");
        assert_eq!(wrap_run_source("do { 1 }", None), "do { 1 }");
        assert_eq!(wrap_run_source("1 + 2", Some("x")), "do {\n1 + 2\nx\n}");
        assert_eq!(
            wrap_run_source("do { a := \"}\" -- c\n a }", Some("f")),
            "do { a := \"}\" -- c\n a \nf\n}"
        );
    }

    /// ADR-0179 egress: a tensor at/above `TENSOR_BIN_THRESHOLD` reaches
    /// `hnl run`/`eval` as a `{"__tensor_bin__": "<base64 LE>"}` record;
    /// `print_value` must expand it to the array `data` spelling before
    /// serializing (the same step `hnl-run` performs).
    #[test]
    fn print_value_materializes_tensor_bin() {
        let shape = [64usize, 64];
        let payload: Vec<u8> = (0..4096u64)
            .flat_map(|x| (x as f64).to_le_bytes())
            .collect();
        let mut v = handl_value::tensor_bin_record(&shape, "f64", &payload).unwrap();
        assert!(
            serde_json::to_string(&v)
                .unwrap()
                .contains("__tensor_bin__")
        );

        let out = print_value(&mut v, &Language::new(), None);
        assert!(!out.contains("__tensor_bin__"), "stdout: {out}");
        let parsed: Value = serde_json::from_str(out.trim()).unwrap();
        let tensor = &parsed["__tensor__"];
        assert_eq!(tensor["dtype"], "f64");
        assert_eq!(tensor["shape"], serde_json::json!([64, 64]));
        let data = tensor["data"].as_array().unwrap();
        assert_eq!(data.len(), 4096);
        assert_eq!(data[0], serde_json::json!(0.0));
        assert_eq!(data[4095], serde_json::json!(4095.0));
    }

    /// The `export!`-generated manifest symbol, called directly.
    #[test]
    fn manifest_declares_run_and_eval() {
        let raw = handl_ext_manifest();
        assert!(!raw.is_null());
        let json = unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned();
        unsafe { handl_ext_free(raw) };
        let manifest: ExtensionManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest.name, "eval");
        let names: Vec<&str> = manifest.commands.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["run", "eval"]);
    }

    // -----------------------------------------------------------------
    // `hnl run` script dispatch (P-003)
    // -----------------------------------------------------------------

    /// Write a minimal `pkg.hnl` whose `scripts` table is `body`.
    fn write_pkg(dir: &Path, body: &str) {
        std::fs::write(
            dir.join("pkg.hnl"),
            format!("{{ scripts: {{ {body} }} }}\n"),
        )
        .unwrap();
    }

    #[test]
    fn run_no_args_lists_pkg_scripts() {
        let dir = tmpdir("run-list");
        write_pkg(&dir, r#"hi: "echo hello", bye: "echo bye""#);
        let res = run_command(&[], &dir);
        assert_eq!(res.code, 0, "stderr: {}", res.stderr);
        assert!(res.stdout.contains("hi"), "stdout: {}", res.stdout);
        assert!(res.stdout.contains("bye"), "stdout: {}", res.stdout);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_no_args_without_pkg_is_usage_error() {
        let dir = tmpdir("run-nopkg");
        let res = run_command(&[], &dir);
        assert_eq!(res.code, 2);
        assert!(
            res.stderr.contains("usage: hnl run"),
            "stderr: {}",
            res.stderr
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_script_name_executes() {
        let dir = tmpdir("run-script");
        write_pkg(&dir, r#"hi: "echo hello-script""#);
        let res = run_command(&["hi".into()], &dir);
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert_eq!(res.stdout, "hello-script\n");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_existing_file_beats_script_name() {
        let dir = tmpdir("run-precedence");
        // A file literally named `gate` shadows script `gate` — the
        // spec's deterministic precedence rule.
        write_pkg(&dir, r#"gate: "echo script-wins""#);
        std::fs::write(dir.join("gate"), "40 + 2").unwrap();
        let res = run_command(&["gate".into()], &dir);
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert_eq!(res.stdout, "42\n", "file exec must win over the script");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_script_args_are_verbatim_argv() {
        let dir = tmpdir("run-argv");
        write_pkg(&dir, r#"hi: "echo hi""#);
        // Runner-looking flags after the name are script argv.
        let res = run_command(&["hi".into(), "--json".into(), "--flag".into()], &dir);
        assert_eq!(res.code, 0, "stderr: {}", res.stderr);
        assert_eq!(res.stdout, "hi --json --flag\n");
        // A leading `--` is the optional separator and is stripped.
        let res = run_command(
            &["--".into(), "hi".into(), "--".into(), "--json".into()],
            &dir,
        );
        assert_eq!(res.code, 0, "stderr: {}", res.stderr);
        assert_eq!(res.stdout, "hi --json\n");
        // `--` mid-argv is verbatim (belongs to the script).
        let res = run_command(&["hi".into(), "a".into(), "--".into(), "b".into()], &dir);
        assert_eq!(res.code, 0, "stderr: {}", res.stderr);
        assert_eq!(res.stdout, "hi a -- b\n");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_json_before_name_emits_run_report() {
        let dir = tmpdir("run-json");
        write_pkg(&dir, r#"hi: "echo json-out""#);
        let res = run_command(&["--json".into(), "hi".into()], &dir);
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        let report: Value = serde_json::from_str(res.stdout.trim()).unwrap();
        assert_eq!(report["script"], "hi");
        assert_eq!(report["exit_code"], 0);
        assert_eq!(report["stdout"], "json-out\n");
        assert_eq!(report["tasks"][0]["status"], "ran");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_plan_prints_verdicts_without_executing() {
        let dir = tmpdir("run-plan");
        write_pkg(&dir, r#"t: { run: "echo should-not-print", cache: true }"#);
        let res = run_command(&["--plan".into(), "t".into()], &dir);
        assert_eq!(res.code, 0, "stdout: {} stderr: {}", res.stdout, res.stderr);
        assert!(res.stdout.contains("t:"), "stdout: {}", res.stdout);
        assert!(res.stdout.contains("miss"), "stdout: {}", res.stdout);
        assert!(
            !res.stdout.contains("should-not-print"),
            "plan must not execute: {}",
            res.stdout
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_script_not_found_is_error_naming_available() {
        let dir = tmpdir("run-unknown");
        write_pkg(&dir, r#"hi: "echo hi""#);
        let res = run_command(&["nope".into()], &dir);
        assert_eq!(res.code, 1, "stderr: {}", res.stderr);
        assert!(
            res.stderr.contains("unknown script"),
            "stderr: {}",
            res.stderr
        );
        assert!(res.stderr.contains("hi"), "stderr: {}", res.stderr);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_script_name_without_pkg_is_error() {
        let dir = tmpdir("run-nopkg-name");
        let res = run_command(&["hi".into()], &dir);
        assert_eq!(res.code, 1, "stderr: {}", res.stderr);
        assert!(res.stderr.contains("no pkg.hnl"), "stderr: {}", res.stderr);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_file_exec_flags_with_script_name_is_error() {
        let dir = tmpdir("run-flag-mix");
        write_pkg(&dir, r#"hi: "echo hi""#);
        let res = run_command(&["--kit".into(), "somewhere".into(), "hi".into()], &dir);
        assert_eq!(res.code, 2, "stderr: {}", res.stderr);
        assert!(
            res.stderr.contains("do not apply to script names"),
            "stderr: {}",
            res.stderr
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_runner_flag_with_file_is_error() {
        let dir = tmpdir("run-flag-file");
        let file = dir.join("main.hnl");
        std::fs::write(&file, "1").unwrap();
        let res = run_command(
            &["--json".into(), file.to_string_lossy().into_owned()],
            &dir,
        );
        assert_eq!(res.code, 2, "stderr: {}", res.stderr);
        assert!(
            res.stderr.contains("not file execution"),
            "stderr: {}",
            res.stderr
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn run_unknown_flag_before_name_is_usage_error() {
        let dir = tmpdir("run-badflag");
        write_pkg(&dir, r#"hi: "echo hi""#);
        let res = run_command(&["--bogus".into(), "hi".into()], &dir);
        assert_eq!(res.code, 2, "stderr: {}", res.stderr);
        assert!(
            res.stderr.contains("unknown flag"),
            "stderr: {}",
            res.stderr
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
