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
//! typechecked through the embedded Handl model (`Kit::typecheck_handl`),
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
use handl_kit_canonical::Kit;
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
        .command("run", "run [--engine <name>] <file.hnl>")
        .command("eval", "eval '<expression>'"),
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

fn run_command(args: &[String], cwd: &Path) -> ExtResponse {
    let mut engine: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--engine" => {
                i += 1;
                if i >= args.len() {
                    return err("run: --engine requires a value\n");
                }
                engine = Some(args[i].clone());
            }
            s => positional.push(s.to_string()),
        }
        i += 1;
    }

    run_file(positional.first().map(String::as_str), engine.as_deref(), cwd)
}

fn run_file(path: Option<&str>, engine: Option<&str>, cwd: &Path) -> ExtResponse {
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

    // Project-aware kit: discover `.handl.toml` by walking up from the
    // source file's directory; its `[dependencies]` table feeds the module
    // loader and its directory becomes the module root for `import "path"`.
    let kit = match engine {
        Some(name) => match Kit::discover_with_eval(&disk, name) {
            Ok(k) => k,
            Err(e) => {
                out.err(format!("run: {e}"));
                return code(1, out);
            }
        },
        None => match Kit::discover(&disk) {
            Ok(k) => k,
            Err(e) => {
                out.err(format!("run: {e}"));
                return code(1, out);
            }
        },
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
    let deps = handl_platform::discover_dependencies(&disk).unwrap_or_default();
    if detect_dialect(&source) == Dialect::Canonical {
        let spliced = spliced_import_names(&source, &bases, &deps);
        if let Err(e) = kit.typecheck_handl_with_prelude(&source, &spliced) {
            out.err(format!("run: typecheck failed: {e}"));
            return code(1, out);
        }
    }

    let source = strip_import_declarations(&source, &bases, &deps);
    let trimmed = source.trim_start();
    let wrapped = if trimmed.starts_with("do") {
        source.to_string()
    } else {
        format!("do {{\n{}\n}}", source)
    };

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
    let engine = handl_plugin_engine_transform_rewrite::RewriteEngine::new()
        .with_native_registry(transform_categorical::native(
            rules,
            protected,
            Arc::new(env),
        ));
    let out_pass = match engine.run_pass(lang, &ast, AstRef::new(ast.len() - 1), "n_dim_opt") {
        Ok(o) => o,
        Err(e) => {
            out.err(format!("run: optimization pass failed: {e}"));
            return code(1, out);
        }
    };

    match kit.runtime().eval(&out_pass.ast) {
        Ok(value) => {
            out.stdout.push_str(&print_value(&value, lang, Some(wrapped.as_str())));
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
    eval_expression(args.first().map(String::as_str), cwd)
}

fn eval_expression(expr: Option<&str>, cwd: &Path) -> ExtResponse {
    let mut out = Out::default();
    let expr = match expr {
        Some(e) => e,
        None => {
            out.err("usage: hnl eval <expression>");
            return code(2, out);
        }
    };

    // Project-aware kit: discover `.handl.toml` from the CWD upward so
    // `import "dep/..."` resolves against `[dependencies]`.
    let kit = match Kit::discover(cwd) {
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
        Ok(value) => {
            out.stdout.push_str(&print_value(&value, lang, Some(wrapped.as_str())));
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
    PRELUDE.get_or_init(handl_kit_canonical::canonical_wire_names)
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

fn err(msg: impl Into<String>) -> ExtResponse {
    ExtResponse::error(2, msg.into())
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
    if let Some(rest) = trimmed.strip_prefix("from") {
        if rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\t') {
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
    if let Some(root) = module_root {
        if !bases.iter().any(|b| b.as_path() == root) {
            bases.push(root.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        if !bases.iter().any(|b| *b == cwd) {
            bases.push(cwd);
        }
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
    resolve_dep_spec_path(spec, deps)
        .or_else(|| bases.iter().find_map(|base| file_candidate(&base.join(spec), spec)))
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
        names.extend(handl_runtime::module_prep::module_export_names(&mod_src));
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
fn print_value(value: &Value, lang: &Language, source: Option<&str>) -> String {
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
    if let Value::Object(o) = value {
        if o.len() == 1 {
            let k = o.keys().next().unwrap().as_str();
            if matches!(
                k,
                "Op" | "Seq" | "Choice" | "First" | "Loop" | "App" | "Arr" | "Const"
                    | "Handle" | "Let" | "Lit"
            ) {
                return Value::String("<term>".to_string());
            }
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
    if let Some(def) = obj.get("body_def").and_then(handl_language::as_ast) {
        if !def.is_empty() {
            let root = AstRef::new(def.len() - 1);
            let body = source
                .zip(def.at(root).ok().and_then(|n| n.span))
                .and_then(|(src, span)| src.get(span.start..span.end))
                .map(|s| s.trim().to_string())
                .or_else(|| {
                    handl_plugin_engine_fmt::FormatEngine::new()
                        .format(lang, &def, root)
                        .ok()
                        .map(|s| s.trim().to_string())
                });
            if let Some(b) = body {
                return Some(format!("{params} ~> {b}"));
            }
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
        assert!(res.stderr.contains("usage: hnl eval <expression>"));
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
}
