#!/usr/bin/env python3
"""Fail if a registered model's decode goes off the production runner/decode seam.

The THIRD "MUST route through, not re-implement" seam (alongside the glue
`vt::FusedChain` catalog and the merged-GEMM `MlpGateUpMethodBase` family, both
policed by scripts/check-fusion-consistency.py): the **decode/runtime path**. A new
model's decode MUST enter the production runner (`ModelRegistry::Forward`, the
registry factory `.forward` hook) and return device-resident logits
(`ForwardLogits.on_device()==true`) on the default `gather_logits` path — handed
straight to the on-GPU sampler — NOT a private generate/argmax loop with a host
logit download. A model born off the runner inherits none of the parity-enablers
(paged bf16-KV attention, on-GPU sampling, shared MoE builders, device RoPE,
decode-graph capture) and forces per-model rediscovery — the "Laguna anti-pattern"
(AGENTS.md, the decode/runtime seam added in `c5c872e6`).

Three invariants, checked per `REGISTER_VLLM_MODEL` (`.forward = &Fn`):

(a) **ON-DEVICE LOGITS** — the model's DEFAULT (`gather_logits`) production forward
    returns a device-resident `ForwardLogits` (reaches the device-logits seam
    `WrapDeviceLogits` / `ViewDeviceLogits` / a `device_storage`/`device_tensor`
    assignment). A registered model whose production forward instead returns
    `HostLogits` / a host `logits.Download` off the on-GPU sampler is DRIFT.

(b) **RUNNER-ROUTED (no private host generate loop)** — the same model must not
    ALSO ship a private `*GenerateCore` host greedy-argmax decode loop as its real
    decode path (the Qwen3-VL escape: `VLGenerateCore` + host `ArgMax`). This is an
    ENRICHMENT of (a): it is only reported for a model already classified HOST, so a
    CLEAN device-resident model that merely keeps a `*GenerateCore` example helper
    around (e.g. qwen3_5's `VLGenerateCoreGdn`) is NOT flagged — the runner path,
    not the presence of a helper, is what makes a model "born on the runner".

(c) **BF16-RESIDENT ACTIVATIONS** — the model's decode path must keep the residual/
    activation stream in bf16 **device** buffers (the shared `dense_attn::AttnBlock`
    preamble + `DBuf` glue, mirroring vLLM's per-op bf16 stores), NOT hand-roll a
    private `std::vector<float>` host residual stream threaded through the per-token
    decode with a `CastBf16`/`GemmBf16`-before-every-projection anti-pattern (AGENTS.md
    born-on-the-runner: "`DBuf`s, not an f32 host residual stream with a `CastBf16`
    before every projection"). A model whose decode files declare >= MIN_F32_RESID
    private f32 residual-stream buffers (`std::vector<float> {hidden,residual,resid,
    hn,hs,x,...}`) AND route the stream through NEITHER the shared bf16 attn preamble
    (`dense_attn::AttnBlock(`) NOR a bf16-`DBuf` residual (`DBuf hidden/res ... kBF16`)
    is classified **F32_STREAM** (drift); an on-framework `DBuf`-based decode is
    **BF16_RESIDENT** (clean). This axis is ORTHOGONAL to (a): deepseek_v4 returns
    device-resident logits (DEVICE, clean on (a)) yet computes its whole decode in an
    f32 host vector stream and only uploads the final logits (F32_STREAM, drift on (c));
    qwen3_vl is HOST on (a)/(b) yet its per-token transformer decode residual is a bf16
    `DBuf` (BF16_RESIDENT, clean on (c)). Supporting per-projection host casts
    (`CastBf16(`/`GemmBf16(`) are REPORTED but not required: laguna casts hn->bf16 before
    each projection (18 sites) while deepseek_v4 keeps everything f32 and quantizes
    per-GEMM (0 CastBf16) — both are the same f32-host-stream escape, so the load-bearing
    signal is the f32 residual-stream declaration + the ABSENCE of a bf16-resident stream,
    not the cast-kernel name. This invariant is REFUSE-skipped like (a): a
    VT_CHECK(false) stub decodes nothing.

Like check-fusion-consistency.py this is a coarse FLOOR, not a per-site proof, and it
resolves through the codebase's real seams so the false-positive/negative rate stays
low:
  * the registered `.forward` hook usually DELEGATES on its `gather_logits` branch to
    `SomeModel::ForwardDevice(...)`; the classification follows that call into the
    ForwardDevice IMPL body (so laguna/qwen3_vl, whose ForwardDevice is a HOST stub
    returning `HostLogits`/`out.host`, are caught even though their registry hook looks
    identical to a clean model's);
  * and ONE further hop into a file-local `ForwardLogits`-returning helper the impl
    calls, so a model that builds its device carrier in its OWN wrapper rather than
    the shared `WrapDeviceLogits` still reads as DEVICE (deepseek_v4's
    `WrapV4DeviceLogits`). Without the hop such a model matched neither seam and
    classified NONE — not an error state, so it dropped out of the drift check
    entirely and the gate stayed green while silently exempting it;
  * `using LlamaModel = Qwen3DenseModel;` aliases are resolved (llama/mistral/internlm2
    reuse the qwen3 dense device forward);
  * a REFUSE-by-name stub (`KimiK3Model::ForwardDevice` is `VT_CHECK(false)`) decodes
    nothing and is SKIPPED, not flagged;
  * a multimodal model with a device text-decode path plus a host mm-PREFILL path
    (gemma4) is CLEAN — its default decode still reaches ForwardDevice.

A model that legitimately cannot route yet is a CONSCIOUS, reviewable allowlist entry
with a reason + fold-plan tier — never a silent landing. The (a)/(b) logits/generate
seam has its own allowlist (scripts/runner-routing-allowlist.txt: laguna, qwen3_vl);
the (c) bf16-activation seam has a SIBLING allowlist
(scripts/runner-bf16-activation-allowlist.txt: laguna, deepseek_v4) — kept separate
because the two axes have DIFFERENT membership (deepseek_v4 is clean on (a) but drifts
on (c); qwen3_vl drifts on (a)/(b) but is clean on (c)), so a shared flat list would
silently cross-suppress. Removing an (a)/(b) entry after the model returns
device-resident logits, or a (c) entry after its decode routes onto the shared bf16
`AttnBlock`/`DBuf` glue (byte-exact / token-exact-or-near-tie gated), is the enforcement
gate closing.

The validation logic is pure functions (`classify_body`, `classify_activation_stream`,
`drift_models`, `f32_stream_drift_models`) so it is unit- and mutation-testable
(tests/scripts/test_check_runner_routing_consistency.py), mirroring
check-fusion-consistency.py.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# `scripts/` is sys.path[0] when this file is RUN, but not when a test loads it by
# path, so pin it either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checker_text import strip_comments  # noqa: E402,F401  (re-exported by name)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "src/vllm/model_executor/models"
INCLUDE_DIR = ROOT / "include/vllm/model_executor/models"
ALLOWLIST = ROOT / "scripts/runner-routing-allowlist.txt"
# Invariant (c) has its OWN allowlist — different membership than (a)/(b).
BF16_ACT_ALLOWLIST = ROOT / "scripts/runner-bf16-activation-allowlist.txt"

# Invariant (c) threshold: how many private f32 residual-stream declarations a decode
# file set must carry before it counts as a hand-rolled f32 stream. laguna declares 12,
# deepseek_v4 declares 8; every on-framework (DBuf) model declares 0 — so any floor in
# [1, 8] separates them. 3 is a conservative margin that ignores a lone f32 scratch var.
MIN_F32_RESID = 3

# --- Seam / signal regexes ----------------------------------------------------

# The DEVICE-resident-logits seam: a ForwardLogits carrying a pool-backed device
# buffer (on_device()==true). WrapDeviceLogits/ViewDeviceLogits are the shared
# helpers every clean model funnels through; the raw field assignments cover a
# model that builds the carrier inline.
_DEVICE_SEAM = re.compile(
    r"\bWrapDeviceLogits\b|\bViewDeviceLogits\b"
    r"|\.device_storage\s*=|\.device_tensor\s*="
)
# The HOST-logits producer: a ForwardLogits with only a host [rows,vocab] buffer
# (on_device()==false) handed off the on-GPU sampler, or a host download/argmax.
_HOST_SEAM = re.compile(
    r"\bHostLogits\s*\(|\bout\.host\b|\.host\s*=\s*std::move|\.Download\s*\("
)
# A REFUSE-by-name stub decodes nothing (VT_CHECK(false)); out of scope.
_REFUSE = re.compile(r"VT_CHECK\(\s*false")

# `.forward = &Fn` — the registry factory decode hook (ModelRegistry::Forward).
_FORWARD_FIELD = re.compile(r"\.forward\s*=\s*&\s*([A-Za-z_]\w*)")
# `using Alias = Target;` — model-class aliases (llama/mistral/internlm2).
_ALIAS = re.compile(r"\busing\s+([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*;")
# `SomeModel::ForwardDevice(...)` (+ tap/mm/step variants) the hook delegates to.
_DELEGATE = re.compile(
    r"\b([A-Za-z_]\w*)::(?:ForwardDevice|ForwardDeviceTap|ForwardDeviceMultiTap"
    r"|ForwardMm|Step)\b"
)
# A private greedy host generate loop (the runner-bypass anti-pattern).
_GENERATE_CORE_CALL = re.compile(r"\b([A-Za-z_]\w*GenerateCore)\s*\(")
_GENERATE_CORE_DEF = re.compile(
    r"\b(?:std::vector<[^;{}]+>|ForwardLogits|void)\s+([A-Za-z_]\w*GenerateCore)\s*\("
)
# A free function the hook calls whose definition file carries the real decode body
# (Qwen3VLForwardStepLastLogits -> qwen3_vl.cpp, which also holds VLGenerateCore).
_FREE_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# --- Invariant (c): bf16-resident activations vs a hand-rolled f32 host stream -------

# A PRIVATE f32 residual/activation-stream declaration: `std::vector<float> <name>` where
# <name> is a residual/hidden/activation carrier threaded through the per-token decode.
# This is the load-bearing tell of the anti-pattern AGENTS.md names ("an f32 host residual
# stream ... with a CastBf16 before every projection"). Only the FIRST identifier of a
# multi-declaration is required to match (e.g. `std::vector<float> x, resA, resB` counts
# once) — the count is a coarse floor, not an exact buffer census.
_F32_RESID_DECL = re.compile(
    r"std::vector<\s*float\s*>\s+(?:hidden|residual|resid|res|hstate|hs|hn|x|cur|act)\b"
)
# The BF16-RESIDENT exemption: the decode keeps its stream in bf16 device buffers. Either
# it routes through the shared bf16 attention preamble (`dense_attn::AttnBlock(` / an
# `AttnBlock(` call under `using namespace dense_attn`) OR it binds a bf16-`DBuf` residual
# stream (`DBuf hidden/res/residual(... kBF16 ...)`). NB: this is deliberately NARROW —
# it matches a residual-NAMED bf16 DBuf, not a per-op bf16 scratch DBuf (laguna's `DBuf
# dh(... kBF16)` cast-scratch must NOT exempt it, since its residual is still `std::vector
# <float> hidden`).
_BF16_ATTN_PREAMBLE = re.compile(r"(?:dense_attn::)?\bAttnBlock\s*\(")
_BF16_DBUF_RESID = re.compile(
    r"\bDBuf\s+(?:hidden|residual|resid|res|hstate|hs)\s*\([^;{}]*\bkBF16\b"
)
# Supporting (reported, not required) evidence: a per-projection host cast/GEMM helper.
_PER_PROJ_HOST_CAST = re.compile(r"\b(?:CastBf16|GemmBf16(?:Into)?)\s*\(")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


# `strip_comments` — a comment mentioning a seam name must never flip a
# classification — is imported above from scripts/checker_text.py, which is the one
# copy this family shares. It blanks comments in place instead of collapsing them,
# so offsets and line numbers survive; for the `findall`/`search` uses below that is
# indistinguishable from the private copy it replaces.


def extract_fn_body(text: str, fn: str) -> str | None:
    """Return the brace-matched body `{...}` of a `ForwardLogits <fn>(...)` DEFINITION
    in `text`, or None. Matches the definition (return type ForwardLogits) rather than
    a call site, and brace-balances params then body."""
    for m in re.finditer(r"\b" + re.escape(fn) + r"\s*\(", text):
        head = text[max(0, m.start() - 48):m.start()]
        if "ForwardLogits" not in head:
            continue
        # Balance the parameter list starting at the '(' we matched.
        i = m.end() - 1
        depth = 0
        while i < len(text):
            c = text[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        j = text.find("{", i)
        semi = text.find(";", i)
        if j < 0 or (0 <= semi < j):
            continue  # a forward-declaration, not a definition
        depth = 0
        k = j
        while k < len(text):
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[j:k + 1]
            k += 1
    return None


def classify_body(body: str | None) -> str:
    """Classify a production-forward body: DEVICE (device-resident logits, clean),
    HOST (host logits off the sampler, drift), REFUSE (VT_CHECK(false) stub, skip),
    or NONE (no recognizable logit producer). DEVICE wins over HOST wins over REFUSE."""
    if body is None:
        return "NONE"
    body = strip_comments(body)
    if _DEVICE_SEAM.search(body):
        return "DEVICE"
    if _HOST_SEAM.search(body):
        return "HOST"
    if _REFUSE.search(body):
        return "REFUSE"
    return "NONE"


def classify_with_helpers(body: str | None, defining_text: str) -> str:
    """`classify_body`, but following ONE level of file-local `ForwardLogits`-returning
    helper that the body CALLS.

    A model may build its device-resident carrier in its own privately-named wrapper
    instead of the shared `WrapDeviceLogits` — deepseek_v4.cpp's `WrapV4DeviceLogits`
    is the case that motivated this. Without the hop, `ForwardDevice` matched NEITHER
    seam and classified NONE, which is not an error state: the model dropped out of
    the HOST-drift check entirely and the gate went green while claiming it had "1
    no-logit-producer". A hole that silently EXEMPTS a model is worse than a red gate,
    so the resolution follows the call the same way the delegate hop already follows
    `Class::ForwardDevice`. Conservative by construction: `extract_fn_body` only
    matches a definition whose return type is `ForwardLogits`, so a helper that
    produces something else is skipped rather than guessed at."""
    direct = classify_body(body)
    if direct == "DEVICE" or body is None:
        return direct
    ranked = {direct}
    for called in _FREE_CALL.findall(strip_comments(body)):
        helper = extract_fn_body(defining_text, called)
        if helper is not None:
            ranked.add(classify_body(helper))
    for level in ("DEVICE", "HOST", "REFUSE"):
        if level in ranked:
            return level
    return "NONE"


def classify_hook(
    body: str | None,
    text: str,
    file_text: dict[str, str],
    free_fn_file: dict[str, str],
) -> str:
    """Classify a registry hook's body with the file-local one-level hop AND one
    cross-TU hop: a free function the hook CALLS that is only DECLARED in the
    registry TU but DEFINED in a sibling models/*.cpp is resolved there (the same
    one-level semantics, one file further).

    mimo_v2 is the case: `mimo_v2_registry.cpp` declares `ForwardMiMoV2Device` and
    defines the hook in the additive registry TU, while the definition lives in
    `mimo_v2.cpp` and returns the shared `WrapDeviceLogits` (the dots3_note
    pattern). Without the cross-TU hop the declaration matches neither seam, the
    model lands in the silently-exempt NONE bucket, and the gate stays green while
    exempting a registered model. `free_fn_file` already maps every free-function
    DEFINITION to its file, so the resolution is exact, not guessed: a name with
    no definition anywhere still classifies NONE, and a cross-TU helper that does
    not return `ForwardLogits` is skipped by `extract_fn_body` like any other.

    The hop only RESOLVES an unclassified hook (NONE). A hook that already
    classifies HOST or REFUSE keeps that verdict: qwen3_vl's registered forward
    carries a direct `HostLogits` off-gather path next to a device last-logits
    helper, and a cross-TU DEVICE hit must not launder that drift — the direct
    seam is authoritative over anything the delegation resolves to."""
    got = classify_with_helpers(body, text)
    if got != "NONE" or body is None:
        return got
    ranked: set[str] = set()
    for called in _FREE_CALL.findall(strip_comments(body)):
        f = free_fn_file.get(called)
        if f is None or f not in file_text:
            continue
        other_body = extract_fn_body(file_text[f], called)
        if other_body is not None:
            ranked.add(classify_with_helpers(other_body, file_text[f]))
    for level in ("DEVICE", "HOST", "REFUSE"):
        if level in ranked:
            return level
    return "NONE"


def count_f32_resid_decls(text: str) -> int:
    """Number of private f32 residual/activation-stream declarations in a decode TU."""
    return len(_F32_RESID_DECL.findall(strip_comments(text)))


def is_bf16_resident(text: str) -> bool:
    """True if the decode TU keeps its residual stream bf16-resident on device — via the
    shared `dense_attn::AttnBlock(` preamble or a bf16-`DBuf` residual (`DBuf hidden ...
    kBF16`). A per-op bf16 scratch DBuf does NOT count (the regex is residual-named)."""
    t = strip_comments(text)
    return bool(_BF16_ATTN_PREAMBLE.search(t) or _BF16_DBUF_RESID.search(t))


def classify_activation_stream(text: str) -> str:
    """Classify a decode file set's activation stream (invariant c): F32_STREAM if it
    hand-rolls >= MIN_F32_RESID private f32 residual-stream buffers AND is NOT
    bf16-resident (no shared AttnBlock preamble, no bf16-DBuf residual); else
    BF16_RESIDENT. Pure over the concatenated decode-body text so it is mutation-testable.

    A bf16-resident decode wins even if it also declares some f32 scratch — the exemption
    is checked FIRST so a `DBuf`-based model that keeps an f32 helper is never mislabeled."""
    if is_bf16_resident(text):
        return "BF16_RESIDENT"
    if count_f32_resid_decls(text) >= MIN_F32_RESID:
        return "F32_STREAM"
    return "BF16_RESIDENT"


def resolve_alias(cls: str, alias: dict[str, str]) -> str:
    seen: set[str] = set()
    while cls in alias and cls not in seen:
        seen.add(cls)
        cls = alias[cls]
    return cls


_IS_POOLING = re.compile(r"\.is_pooling_model\s*=\s*true")


@dataclass(frozen=True)
class ModelRoute:
    """The decode-routing verdict for one REGISTER_VLLM_MODEL registration."""
    name: str                      # allowlist key (registry stem minus _registry)
    reg_file: str
    forward_fn: str
    classification: str            # DEVICE | HOST | REFUSE | POOLING | NONE
    private_generate_loop: bool    # invariant (b): ships a *GenerateCore host loop
    device_source: str = ""        # which delegated class supplied the device seam
    activation: str = "BF16_RESIDENT"  # invariant (c): F32_STREAM | BF16_RESIDENT
    f32_resid_decls: int = 0       # invariant (c): private f32 residual-stream decls
    per_proj_host_casts: int = 0   # invariant (c): supporting CastBf16/GemmBf16 sites


def build_alias_map(files: list[Path]) -> dict[str, str]:
    alias: dict[str, str] = {}
    for p in files:
        for a, tgt in _ALIAS.findall(strip_comments(read(p))):
            alias[a] = tgt
    return alias


def collect_forwarddevice_bodies(cpp_files: list[Path]) -> dict[str, tuple[str, str]]:
    """Map ModelClass -> (defining file name, ForwardDevice body) for every
    `ForwardLogits Class::ForwardDevice(...) {...}` definition."""
    out: dict[str, tuple[str, str]] = {}
    for p in cpp_files:
        text = read(p)
        for m in re.finditer(
            r"ForwardLogits\s+([A-Za-z_]\w*)::ForwardDevice\s*\(", text
        ):
            cls = m.group(1)
            j = text.find("{", m.end())
            semi = text.find(";", m.end())
            if j < 0 or (0 <= semi < j):
                continue
            depth = 0
            k = j
            while k < len(text):
                c = text[k]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        out.setdefault(cls, (p.name, text[j:k + 1]))
                        break
                k += 1
    return out


def _model_name(reg_stem: str) -> str:
    return reg_stem[:-len("_registry")] if reg_stem.endswith("_registry") else reg_stem


def scan_registrations(
    models_dir: Path, include_dir: Path
) -> dict[str, ModelRoute]:
    """Scan every registered model and classify its decode routing. Returns a map
    name -> ModelRoute."""
    routes: dict[str, ModelRoute] = {}
    if not models_dir.is_dir():
        return routes
    cpp_files = sorted(models_dir.glob("*.cpp"))
    header_files = sorted(models_dir.glob("*.h"))
    if include_dir.is_dir():
        header_files += sorted(include_dir.glob("*.h"))

    alias = build_alias_map(cpp_files + header_files)
    fd_bodies = collect_forwarddevice_bodies(cpp_files)
    # free-function definition -> file (for resolving the hook's decode body file).
    # DEFINITIONS only: a registry TU's forward DECLARATION (mimo_v2_registry.cpp's
    # `ForwardLogits ForwardMiMoV2Device(...);`) matches the same regex but must not
    # claim the name, or the cross-TU hop would resolve to a file with no body.
    free_fn_file: dict[str, str] = {}
    file_text: dict[str, str] = {}
    for p in cpp_files:
        file_text[p.name] = read(p)
        for m in re.finditer(
            r"\b(?:std::vector<[^;{}]+>|ForwardLogits)\s+([A-Za-z_]\w*)\s*\(", file_text[p.name]
        ):
            if extract_fn_body(file_text[p.name], m.group(1)) is not None:
                free_fn_file.setdefault(m.group(1), p.name)

    for p in cpp_files:
        text = file_text[p.name]
        fm = _FORWARD_FIELD.search(strip_comments(text))
        if not fm:
            continue
        fn = fm.group(1)
        name = _model_name(p.stem)
        body = extract_fn_body(text, fn)
        clean_body = strip_comments(body) if body else ""

        # Files that carry this model's real decode body: the registry hook file,
        # the ForwardDevice impl files of delegated classes, and the def files of
        # free functions the hook calls.
        impl_files: set[str] = {p.name}
        delegated_classes = {
            resolve_alias(c, alias) for c in _DELEGATE.findall(clean_body)
        }
        device_source = ""
        classification = "NONE"
        # POOLING registrations (ARCH-ONE-SURFACE ROW 6): a registry TU that
        # declares `.is_pooling_model = true` registers a NON-GENERATIVE model -
        # its forward is a HIDDEN-STATE producer for the PoolingRunner (the
        # engine pools instead of sampling, gpu/model_runner.py:1586-1607), so
        # the device-resident-LOGITS seam does not apply BY DESIGN, exactly as
        # a refuse-by-name stub decodes nothing. Classified explicitly (never
        # the silently-exempt NONE bucket) and excluded from the HOST-drift
        # gate; the bf16-activation invariant still applies to its body.
        if _IS_POOLING.search(strip_comments(text)):
            classification = "POOLING"
        elif classify_hook(body, text, file_text, free_fn_file) == "DEVICE":
            classification, device_source = "DEVICE", fn
        for cls in delegated_classes:
            impl = fd_bodies.get(cls)
            if impl:
                impl_files.add(impl[0])
                impl_class = classify_with_helpers(impl[1], file_text.get(impl[0], ""))
                if impl_class == "DEVICE" and classification != "DEVICE":
                    classification, device_source = "DEVICE", cls
        if classification not in ("DEVICE", "POOLING"):
            # Not device-reachable: rank the delegated ForwardDevice impls, else the
            # hook body itself, as HOST > REFUSE > NONE.
            impl_classes = [
                classify_with_helpers(fd_bodies[c][1], file_text.get(fd_bodies[c][0], ""))
                for c in delegated_classes
                if c in fd_bodies
            ]
            if "HOST" in impl_classes:
                classification = "HOST"
            elif not impl_classes and _HOST_SEAM.search(clean_body):
                classification = "HOST"
            elif "REFUSE" in impl_classes:
                classification = "REFUSE"
            elif _REFUSE.search(clean_body):
                classification = "REFUSE"

        # Invariant (c): bf16-resident activations. Scope to the decode-BODY files (the
        # registry hook file + the delegated ForwardDevice impl files) — NOT the broader
        # free-fn expansion below (which would drag in a model's mm-PREFILL / audio tower
        # helpers, whose legitimate f32 vectors are not the decode residual stream). A
        # REFUSE stub decodes nothing, so it is never an f32-stream escape.
        decode_text = "\n".join(file_text.get(f, "") for f in sorted(impl_files))
        f32_resid = count_f32_resid_decls(decode_text)
        per_proj_casts = len(_PER_PROJ_HOST_CAST.findall(strip_comments(decode_text)))
        activation = (
            classify_activation_stream(decode_text)
            if classification != "REFUSE"
            else "BF16_RESIDENT"
        )

        # Invariant (b): does this model's decode-body file set define a private
        # *GenerateCore host loop? (Only meaningful for a HOST model — a DEVICE model
        # that keeps a GenerateCore example helper is legitimately clean.)
        for called in _FREE_CALL.findall(clean_body):
            f = free_fn_file.get(called)
            if f:
                impl_files.add(f)
        private_loop = any(
            _GENERATE_CORE_DEF.search(strip_comments(file_text.get(f, "")))
            for f in impl_files
        )

        routes[name] = ModelRoute(
            name=name,
            reg_file=p.name,
            forward_fn=fn,
            classification=classification,
            private_generate_loop=private_loop,
            device_source=device_source,
            activation=activation,
            f32_resid_decls=f32_resid,
            per_proj_host_casts=per_proj_casts,
        )
    return routes


def allowlisted_names(text: str) -> set[str]:
    """Model names accepted as known off-framework / deliberately-deferred (one per
    line, # comments ignored) — mirrors check-fusion-consistency.py."""
    names: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def drift_models(
    scanned: dict[str, ModelRoute], allowlisted: set[str]
) -> list[str]:
    """Model names whose production decode returns HOST logits off the runner's
    on-GPU sampler (invariant a) and are not allowlisted. Empty == the check passes.
    REFUSE stubs and DEVICE-clean models never drift."""
    return sorted(
        name
        for name, route in scanned.items()
        if route.classification == "HOST" and name not in allowlisted
    )


def f32_stream_drift_models(
    scanned: dict[str, ModelRoute], allowlisted: set[str]
) -> list[str]:
    """Model names whose decode hand-rolls an f32 host residual/activation stream
    instead of the shared bf16 DBuf glue (invariant c: activation == F32_STREAM) and
    are not allowlisted. Empty == the check passes. REFUSE stubs and BF16_RESIDENT
    (DBuf-based) decodes never drift."""
    return sorted(
        name
        for name, route in scanned.items()
        if route.activation == "F32_STREAM"
        and route.classification != "REFUSE"
        and name not in allowlisted
    )


def _load_allowlist(path: Path) -> set[str]:
    return allowlisted_names(read(path)) if path.exists() else set()


def main() -> int:
    scanned = scan_registrations(MODELS_DIR, INCLUDE_DIR)
    allowlisted = _load_allowlist(ALLOWLIST)
    bf16_allowlisted = _load_allowlist(BF16_ACT_ALLOWLIST)
    drift = drift_models(scanned, allowlisted)
    f32_drift = f32_stream_drift_models(scanned, bf16_allowlisted)

    n_device = sum(1 for r in scanned.values() if r.classification == "DEVICE")
    n_host = sum(1 for r in scanned.values() if r.classification == "HOST")
    n_refuse = sum(1 for r in scanned.values() if r.classification == "REFUSE")
    n_pooling = sum(1 for r in scanned.values() if r.classification == "POOLING")
    n_none = sum(1 for r in scanned.values() if r.classification == "NONE")
    n_f32 = sum(1 for r in scanned.values() if r.activation == "F32_STREAM")
    n_bf16 = sum(
        1 for r in scanned.values()
        if r.activation == "BF16_RESIDENT" and r.classification != "REFUSE"
    )

    rc = 0

    # Invariant (a)+(b): on-device logits / no private host generate loop.
    if drift:
        rc = 1
        print(
            "ERROR: registered model decode(s) return HOST logits off the production "
            "runner / on-GPU sampler instead of a device-resident ForwardLogits "
            "(on_device()==true) on the default gather_logits path — the "
            "decode/runtime seam (AGENTS.md 'born on the runner') — and are not on "
            "scripts/runner-routing-allowlist.txt:",
            file=sys.stderr,
        )
        for name in drift:
            r = scanned[name]
            extra = (
                " + ships a private *GenerateCore host generate loop off the runner "
                "(invariant b)" if r.private_generate_loop else ""
            )
            print(
                f"  - {name} ({r.reg_file}: {r.forward_fn} returns HostLogits"
                f"{extra})",
                file=sys.stderr,
            )
        print(
            "Route decode through ModelRegistry::Forward returning "
            "ForwardLogits.on_device()==true (WrapDeviceLogits / ViewDeviceLogits on "
            "the gather_logits path; see qwen3_dense.cpp / gemma.cpp / opt.cpp), or "
            "add the model to scripts/runner-routing-allowlist.txt with a reason "
            "(pending framework-routing, see AGENTS.md decode/runtime seam).",
            file=sys.stderr,
        )
    else:
        print(
            f"OK (runner-routing): {len(scanned)} registered model(s); "
            f"{n_device} return device-resident logits on the runner, "
            f"{n_host} host-logits off-framework ({len(allowlisted)} allowlisted), "
            f"{n_refuse} refuse-by-name stub(s) skipped, "
            f"{n_pooling} pooling (hidden-state producer(s) for the PoolingRunner), "
            f"{n_none} no-logit-producer."
        )

    # Invariant (c): bf16-resident activations (no hand-rolled f32 host stream).
    if f32_drift:
        rc = 1
        print(
            "ERROR: registered model decode(s) hand-roll an f32 host residual/activation "
            "stream (private std::vector<float> {hidden,residual,x,...} threaded through "
            "the per-token decode with CastBf16/GemmBf16-before-every-projection) instead "
            "of the shared bf16 DBuf glue (dense_attn::AttnBlock / bf16-resident DBuf) — "
            "the bf16-resident-activations invariant (AGENTS.md 'born on the runner') — "
            "and are not on scripts/runner-bf16-activation-allowlist.txt:",
            file=sys.stderr,
        )
        for name in f32_drift:
            r = scanned[name]
            casts = (
                f", {r.per_proj_host_casts} CastBf16/GemmBf16 per-projection cast(s)"
                if r.per_proj_host_casts else ""
            )
            print(
                f"  - {name} ({r.reg_file}: {r.f32_resid_decls} f32 residual-stream "
                f"decl(s){casts}, no bf16-resident stream) -> its M=1 decode projections "
                "buy the slow cuBLAS gemvx<bf16,FLOAT> template (f32 output) where vLLM "
                "runs gemvx<bf16,bf16>; the op-contract side is policed by "
                "scripts/check-gemv-invocation-consistency.py",
                file=sys.stderr,
            )
        print(
            "Route the decode residual onto the shared bf16 dense_attn::AttnBlock preamble "
            "+ DBuf glue (bf16-resident device activations; see qwen3.cpp / qwen3_5.cpp / "
            "gemma4.cpp), or add the model to scripts/runner-bf16-activation-allowlist.txt "
            "with a reason + fold-plan tier (pending framework-routing, see AGENTS.md "
            "decode/runtime seam).",
            file=sys.stderr,
        )
    else:
        print(
            f"OK (bf16-activation): {len(scanned)} registered model(s); "
            f"{n_bf16} keep bf16-resident DBuf activations, "
            f"{n_f32} hand-roll an f32 host stream ({len(bf16_allowlisted)} allowlisted), "
            f"{n_refuse} refuse-by-name stub(s) skipped."
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
