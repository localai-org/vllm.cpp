#!/usr/bin/env python3
"""DSR ratchet — Device-Specific References in the device-agnostic shared layer.

Work row `S1` of [.agents/specs/accelerator-seam-audit.md]. Runs standalone with
no CUDA toolkit, no GPU and no build:

    python3 scripts/check-device-leakage.py            # gate against the baseline
    python3 scripts/check-device-leakage.py --report   # per-file breakdown
    python3 scripts/check-device-leakage.py --write-baseline   # after a REDUCTION

WHY THIS EXISTS. `src/vllm/` + `include/vllm/` is the layer that is supposed to
be device-agnostic: the engine, the worker, the model definitions and the
platform seam. Upstream vLLM keeps essentially no device branching in
`model_executor/models/` (14 sites across 287 files) because
`model_executor/layers/` absorbs it once for everyone. We never ported that
library, so the same branching lands in our model TUs instead — 71% of it in a
single file. The audit re-measured the leakage and found it had DRIFTED UPWARD
since the previous study with **no bad commit**: DeepSeek-V2, Qwen3-Coder and the
attention-registry work each added a device test in passing. Leakage grows
silently under well-executed work, so it needs a ratchet, not a cleanup.

WHAT IT COUNTS. Five buckets, over non-comment / non-string-literal source:

  kcuda      textual `DeviceType::kCUDA` (or bare `kCUDA`) references
  is_cuda    `is_cuda()` call sites
  dev_cast   an integer converted into a `vt::DeviceType` by a cast, in any
             spelling — the device named by ENUM VALUE instead of by the
             platform seam
  cuda_inc   `#include "vt/cuda/…"` / `<cuda_runtime…>` NOT inside a CUDA or
             VT_* preprocessor guard (i.e. a non-CUDA build cannot compile)
  vt_ifdef   `#ifdef VT_*` / `#if defined(VT_*)` build-time kernel-feature gates

Comments and string literals are stripped before matching in the first three
buckets. That is a DELIBERATE correction to the audit's composition, which
excluded comment lines from `is_cuda` but not from `kCUDA` — a prose mention of
a device name is not device leakage, and making the rule uniform is what keeps
the metric honest in both directions.

WHY `dev_cast` EXISTS (#660). `kcuda` is a TOKEN grep. `DeviceType` is
`enum class DeviceType : uint8_t { kCPU = 0, kCUDA = 1, … }`
(include/vt/device.h), so `static_cast<vt::DeviceType>(device)` on the public
0/1 ABI selector hardcodes CUDA *by enum value* and never writes the token. That
is exactly what `src/vllm/multimodal/minimax_h3_video.cpp` did: it scored ZERO in
`kcuda` while the test asserting the same mapping spelled `kCUDA` honestly and
WAS counted. The gate read the confession and missed the act. It is also an
enum-ordering hazard — reordering the enum silently re-points every such site.

The reverse direction (`static_cast<int>(type)`) is the SAFE one and is
deliberately not counted: it is how the seam itself indexes its registry.

WHAT `dev_cast` ENFORCES, EXACTLY. It is a regular expression over
comment-stripped text, so what it enforces is a SET OF SPELLINGS. It does NOT
enforce the property it was written for — "an integer becomes a DeviceType
outside the platform seam" — and it cannot: it cannot resolve a name to a type,
cannot tell an integer operand from a DeviceType one, and cannot see a conversion
whose target type is not written at the conversion site. Read a green `dev_cast`
as "none of the spellings enumerated below is present", never as "no integer
becomes a DeviceType here". #828 tracks the AST-level check (clang tooling),
where the destination type canonicalises and the source type is known, and which
is what would enforce the property. This pattern is the INTERIM, and says so.

That distinction is measured, not defensive. FOUR review rounds of this one
bucket each found a spelling the previous round's message already claimed to
cover. #660 shipped it; round 1 found the literal operand `(vt::DeviceType)1`,
and the mutant written to rule that out used an IDENTIFIER operand, so it could
not see the gap it existed to close; round 2 found nine more that all write the
target type the docstring named as its anchor; round 3 found eight more (counted,
not rounded), including a FOURTH place a cv-qualifier can sit while the mutant
asserted "all three".
Every one of them closed at ZERO hits on the scanned tree, so none was a
coverage-versus-false-positive trade — there was simply always another spelling.
So the list below is an enumeration of what was MEASURED, not a claim about a
class.

WHAT `dev_cast` DOES SEE. Each entry is pinned by its own mutant in
tests/scripts/test_device_leakage.py, asserted individually:

  * the named casts `static_cast`, `reinterpret_cast`, `const_cast`,
    `dynamic_cast` and `bit_cast`, plus the builtin spelling
    `__builtin_bit_cast(vt::DeviceType, x)`. `bit_cast` is a FUNCTION template
    with two parameters, so its second template argument (`From`) may be
    written — as long as it is on one line and contains no nested angle
    brackets;
  * the C-style cast `(vt::DeviceType)x`, the functional casts
    `vt::DeviceType(x)` and `vt::DeviceType{x}`, and the same conversion spelled
    as a DECLARATION, `vt::DeviceType d{x}`;
  * the type written `DeviceType`, `vt::DeviceType`, `::vt::DeviceType` or
    `enum vt::DeviceType`, with `const`/`volatile` on EITHER side of the name in
    ALL of those forms — the declaration form included, which is the fourth
    cv-position and the one round 3 found (`vt::DeviceType const kD{1}`);
  * an operand that is an identifier, a literal (`)1`), a signed literal (`)-1`),
    a parenthesised expression, or a unary expression (`)*cursor`, `)~mask`,
    `)!flag`);
  * a target that is a POINTER, a POINTER-TO-POINTER or a REFERENCE — the pun
    `reinterpret_cast` performs on an integer's bytes — in the named-cast form
    AND the C-style form, with cv-qualifiers anywhere inside the `*`/`&` run
    (`reinterpret_cast<vt::DeviceType* const&>(p)`);
  * a cast clang-format has wrapped across two lines.

WHAT `dev_cast` STILL CANNOT SEE, stated here because a checker's message is the
authority on what it enforces and an instrument that hides its blind spot
returns a false pass. Every entry states the reason that is true OF THAT ENTRY
ALONE, and no entry binds two cases whose reasons differ — a list whose shared
reason is false for one member is the same false claim in a smaller font, and
that is precisely what rounds 2 and 3 each found here:

  * a cast whose target is a type ALIAS (`using DT = vt::DeviceType;` then
    `static_cast<DT>(x)`), a MACRO that expands to the type name, or a TEMPLATE
    PARAMETER that resolves to DeviceType. Reason: no spelling of the target
    type appears at the conversion site, so there is nothing to anchor on;
  * `memcpy` or a UNION punning an integer onto a DeviceType. Reason: there is
    no cast EXPRESSION to anchor on. (Not "the type is not named at the site":
    the idiomatic `std::memcpy(&dt, &raw, sizeof(vt::DeviceType))` names it right
    there, and matching that would mean matching a bare `sizeof(vt::DeviceType)`,
    which converts nothing. The old entry gave `bit_cast`'s reason to `memcpy`,
    and `bit_cast` is now a keyword above.);
  * the C-style POINTER PUN whose operand begins with `&`, `*(vt::DeviceType*)&x`.
    Reason: closing it needs `&` in the trailing class, and `) &` / `) &&` are
    ref-qualifiers on a member declarator — `void note (vt::DeviceType*) &;`
    compiles, and admitting `&` turns it into a false positive (measured: 2).
    Every OTHER C-style pointer or reference cast IS caught — `(vt::DeviceType&)`
    and `(vt::DeviceType*)` both — as is the named-cast spelling of this same
    pun, so this is one operand shape rather than the class;
  * a CHARACTER-LITERAL operand, `(vt::DeviceType)'\x01'`. Reason: character
    literals are blanked to whitespace by `strip_comments_and_strings` before the
    pattern ever runs, so the operand is gone by match time and no trailing class
    could see it. `(vt::DeviceType)buf[0]` is caught, so this is that one literal
    kind, not the wire-decode shape;
  * a conversion that happens inside `src/vt/` and is merely CALLED from the
    shared layer. Reason: `src/vt/` is a device leg and is not scanned at all;
  * whether the operand really is an integer. Reason: nothing here type-checks.
    The bucket flags every cast TO DeviceType and relies on `// DSR-ALLOW(<row>)`
    for the legitimate ones (a wire-format decode is the expected case). This is
    also why a pointer or reference target counts: `const_cast<DeviceType&>(t)`
    removes const rather than converting an integer, and buys its exemption the
    same way;
  * any spelling that is not on the list above. Reason: this is a spelling list,
    and four rounds of finding a new one is the evidence that a spelling list is
    not the class. That gap does not close by widening; it closes at #828.

WHAT `dev_cast` OVER-MATCHES. The two lists above answer "what is caught" and
"what is missed". Neither answers "what is caught that is not the defect", and a
gate whose message omits its own false positives is the same instrument fault in
the other direction: the reader takes a RED as proof of leakage.

  * a plain COPY-INITIALISATION whose target is a DeviceType and whose operand
    ALREADY IS one — `vt::DeviceType d{other}`, a member default-init
    `struct S { vt::DeviceType d{kCPU}; }`, or an init from a call
    `vt::DeviceType d{platform.device_type()}`. Alternative (3) fires and nothing
    is converted. Measured on this pattern: 1 hit each; `vt::DeviceType d{}`
    scores 0, because the empty-braces lookahead already rejects a
    value-initialisation.

    This is NOT narrowed, and the reason is that narrowing it would delete a real
    catch. `vt::DeviceType d{raw}` — the declaration spelling of the conversion,
    which review rounds 1 and 3 found and M32/M36 pin — is textually identical to
    `vt::DeviceType d{other}`. A text checker cannot tell the two apart, which is
    the same statement as "nothing here type-checks", and is exactly what #828
    resolves. The `const_cast<DeviceType&>(t)` entry above is the other side of
    it and must keep firing for the same reason.

    The COST is not zero, so it is stated rather than dismissed: `dev_cast`'s
    baseline is a hard 0 outside the one allowlisted `platform.cpp` hit, so the
    FIRST such copy-init written under `src/vllm/` or `include/vllm/` fails the
    ratchet as a `DSR REGRESSION`. The remedy is `// DSR-ALLOW(<row-id>): <why>`
    on the line, which is the same remedy the wire-decode case gets, and it is
    visible in CI output rather than in the diff. Nothing in the scanned tree
    writes this form today — a repo-wide probe over the scan roots finds
    `dev_cast` at exactly its allowlisted hit — so this is a documented cost of
    the next line, not a live failure.

    M47 pins it, with `vt::DeviceType d{}` as the negative control in the same
    test: if a later change makes the copy-init form stop firing, M47 goes RED
    and this entry must be corrected in the same change. That is the M46 shape,
    applied to a false positive instead of a blind spot.

tests/scripts/test_device_leakage.py M20-M47 pin what it does catch, each
spelling asserted on its own — including, in M29, the literal operand, because a
discriminator tested only on the case it was tuned for is a guard that certifies
itself.

HOW THE RATCHET WORKS. `scripts/device-leakage-baseline.json` holds the accepted
per-bucket counts. Any bucket ABOVE its baseline fails. Any bucket BELOW its
baseline also fails, with the instruction to re-run `--write-baseline` and commit
the lowered baseline IN THE SAME COMMIT as the reduction. So the number can only
ever move down, and it moves down only deliberately. Per-bucket enforcement (not
just the total) is what stops a removed `#ifdef` from paying for a new `kCUDA`.

THE ALLOWLIST is per-file, per-bucket, with an exact expected count and a stated
reason. These sites ARE the CUDA leg — the `(kCUDA, name)` registrar keys, the
platform-priority walk, the `is_cuda()` definition — and counting them would
punish adding a backend, which is the opposite of the point. An allowlist entry
whose count no longer matches fails too, so the allowlist itself is a ratchet and
cannot silently absorb new leakage.

THE ESCAPE HATCH is `// DSR-ALLOW(<row-id>): <reason>` on the offending line or
the line directly above it. Such sites are excluded from the count but are
COUNTED AND PRINTED separately on every run, so the exception budget is visible
in CI output rather than invisible in the diff. Repair the code, do not grow this
list; and never raise a baseline to make a failing state pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "scripts/device-leakage-baseline.json"

# The layer that is supposed to be device-agnostic. `src/vt/cuda/`,
# `src/vt/metal/` and `src/vt/vulkan/` are device legs BY DEFINITION and are not
# scanned at all — they are where device-specific code is SUPPOSED to live.
SCAN_ROOTS = ("src/vllm", "include/vllm")
SOURCE_SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".cu", ".cuh"}

BUCKETS = ("kcuda", "is_cuda", "dev_cast", "cuda_inc", "vt_ifdef")

RE_KCUDA = re.compile(r"\bkCUDA\b")
RE_IS_CUDA = re.compile(r"\bis_cuda\s*\(\s*\)")

# `dev_cast`: an integer becoming a DeviceType. Four alternatives, one per C++
# spelling of the same conversion, all anchored on the TARGET TYPE:
#
#   1. a named cast          static_cast<vt::DeviceType>(x) / <DeviceType> / …
#                            and the INDIRECT forms, *reinterpret_cast<DeviceType*>(&x)
#                            and reinterpret_cast<DeviceType&>(x)
#   1b. __builtin_bit_cast   __builtin_bit_cast(vt::DeviceType, x) — the same
#                            conversion as (1)'s `bit_cast`, but a two-argument
#                            BUILTIN rather than a template-id, so (1)'s `<…>`
#                            anchor cannot reach it
#   2. a C-style cast        (vt::DeviceType)x, (vt::DeviceType)1,
#                            (vt::DeviceType&)raw and (vt::DeviceType*)vp
#   3. a functional cast     vt::DeviceType(x) / vt::DeviceType{x}, and the same
#                            conversion spelled as a DECLARATION, `DeviceType d{x}`
#
# `_DEVTYPE_QUAL` absorbs the ways the SAME type can be written before its name:
# `vt::DeviceType`, a global-scope-qualified `::vt::DeviceType`, and an
# elaborated-type-specifier `enum vt::DeviceType`. All three name one type, so a
# pattern that recognised only the first would be a spelling grep again — which
# is the exact defect (#660) this bucket exists to answer.
#
# `\b` after the type name keeps `DeviceTypeName(t)` out. The lookbehind on (3)
# keeps `X::DeviceType(` and `foo.DeviceType(` and `vector<DeviceType>(` out, so
# a qualified spelling is matched once at its `vt::`, not twice. The negative
# lookahead keeps a zero-argument `DeviceType()` value-initialisation out — that
# converts nothing. Matched over the WHOLE comment-stripped text rather than
# line by line, so a clang-format-wrapped cast cannot slip between two lines.
#
# (2) needs two guards, because `(Type)ident` and a PARAMETER LIST are textually
# the same thing. `supports_worker_transfer_on(vt::DeviceType /*device*/) const`
# in kv_connector.h is a declaration whose parameter name is commented out; after
# stripping it reads exactly like a cast applied to `const`. So the `(` must not
# be glued to an identifier (that is a call or a declarator), except after the
# keywords that legitimately precede a parenthesised expression; and what follows
# must not be a declarator suffix.
#
# (2)'s trailing class admits DIGITS, a leading SIGN and the UNARY OPERATORS
# `*`, `~` and `!`, not just identifiers. `(vt::DeviceType)1` — the device named
# by its literal enum value — is the PUREST form of what this bucket polices, and
# an identifier-only lookahead missed exactly it while catching
# `(vt::DeviceType)d`; `(vt::DeviceType)*cursor` is the WIRE-DECODE spelling this
# docstring itself names as the expected `DSR-ALLOW` case, and a class that could
# not see it was refusing to police the one site it predicted. `&` is deliberately
# NOT admitted: `) &` and `) &&` are ref-qualifiers on a member declarator, which
# is the false positive the trailing guard exists to reject. Measured over
# `src/vllm` + `include/vllm`: admitting `0-9+-*~!` adds ZERO hits.
#
# (2)'s TARGET, by contrast, does take a run of `*` and `&` — INSIDE the parens,
# where no declarator can be confused with it. That is the split the blind-spot
# entry used to get wrong: the old note bound "pointer or reference" together
# under the single reason "both are `)` followed by `&`", which is true of
# `*(vt::DeviceType*)&x` and FALSE of `(vt::DeviceType&)raw`, whose next character
# is an identifier. Widening only inside the parens catches the reference form and
# `(vt::DeviceType*)vp` at zero cost; the ref-qualifier negatives
# `void note (vt::DeviceType*) &;` and `… &&;` stay at zero because `&` is still
# out of the TRAILING class. Both halves compile-verified, both measured.
#
# `sizeof (vt::DeviceType) + 1` was ALREADY a false positive before that widening
# — `sizeof` is not glued to its paren there, so the identifier discriminator
# passes, and `+` is in the trailing class — and admitting `*` inside the parens
# would have extended it to `sizeof (vt::DeviceType*) + 1`. A `sizeof` or
# `alignof` converts nothing, so both are excluded outright rather than left as a
# gap the widening made wider. The lookbehinds are fixed-width, so they cover the
# one-space spelling clang-format produces; `sizeof(` with no space was already
# blocked by the glued-identifier discriminator. Measured: removing these two
# costs nothing on the tree (dev_cast stays at its single allowlisted hit).
#
# (1) and (2) and (3) all take the cv-qualifiers on EITHER SIDE of the type name,
# and (1) takes any run of `*`, `&` and cv-qualifiers INTERLEAVED.
# `vt::DeviceType const` is the same type as `const vt::DeviceType`;
# `reinterpret_cast<vt::DeviceType&>(raw)` is the same pun as
# `*reinterpret_cast<vt::DeviceType*>(&raw)`; and `vt::DeviceType* const&` puts a
# cv-qualifier one position further right than a `[*&]`-only run can reach — all
# COMPILE (checked, not read). (3) is where the fourth cv-position hides:
# `vt::DeviceType const d{raw}` is a declaration, and a group present on (1) and
# (2) but absent from (3) is a docstring claiming three places while the code
# covers two.
#
# `bit_cast` joins the four cast keywords because it spells the target type at the
# conversion site, so the blind-spot note below could not honestly claim it as
# unreachable. It is a FUNCTION template, not a cast operator, so `From` may also
# be written — `std::bit_cast<vt::DeviceType, std::uint8_t>(raw)` compiles — and
# the optional `,` tail is what lets (1) reach that form. `__builtin_bit_cast`
# gets its own alternative because it is not a template-id at all. `memcpy` and
# union punning stay blind, and their reason is now their own: there is no cast
# expression to anchor on.
#
# (3)'s DECLARATION form takes `{` only. `DeviceType d{x}` is a real conversion
# (C++17 permits list-initialising a scoped enum with a fixed underlying type),
# but `DeviceType d(x)` is ILL-FORMED — there is no implicit int→scoped-enum
# conversion — so nothing is lost by excluding it, while allowing `(` would match
# every FUNCTION DEFINITION whose return type is DeviceType (measured: 3 false
# positives, e.g. `vt::DeviceType MiniMaxH3VideoDeviceType(`). Adding the east
# cv-group to (3) does not reopen that: `vt::DeviceType const Frozen(int);` still
# scores 0, because `{` is still the only initialiser admitted.
_DEVTYPE_QUAL = r"(?:enum\s+)?(?:::\s*)?(?:vt\s*::\s*)?"
# cv-qualifiers, which C++ permits on either side of the type name.
_CV_WEST = r"(?:(?:const|volatile)\s+)*"
_CV_EAST = r"(?:\s*(?:const|volatile)\b)*"
# A named cast's target suffix: `*`, `&` and cv-qualifiers, any number, any order.
# `vt::DeviceType* const&` needs the cv-qualifier INSIDE the run, not before it.
_TARGET_SUFFIX = r"(?:\s*(?:const\b|volatile\b|[*&]))*"
# `bit_cast`'s second template parameter (`From`), which may be written out.
# One line, no nested angle brackets — deliberately narrow, since the scanned
# roots contain zero `bit_cast` occurrences and a greedy tail would be the only
# way this could cost a false positive.
_TEMPLATE_TAIL = r"(?:\s*,[^<>;{}\n]*)?"
RE_DEVTYPE_CAST = re.compile(
    r"(?:static_cast|reinterpret_cast|const_cast|dynamic_cast|bit_cast)\s*<\s*"
    + _CV_WEST + _DEVTYPE_QUAL + r"DeviceType\b" + _TARGET_SUFFIX + _TEMPLATE_TAIL + r"\s*>"
    r"|__builtin_bit_cast\s*\(\s*"
    + _CV_WEST + _DEVTYPE_QUAL + r"DeviceType\b" + _CV_EAST + r"\s*,"
    r"|(?:(?<![\w])|(?<=return)|(?<=case)|(?<=throw)|(?<=delete))(?<!sizeof\s)(?<!alignof\s)"
    r"\(\s*" + _CV_WEST + _DEVTYPE_QUAL + r"DeviceType\b" + _CV_EAST + r"(?:\s*[*&])*\s*\)\s*"
    r"(?!(?:const|volatile|noexcept|override|final|try)\b)(?=[A-Za-z_(0-9+\-*~!])"
    r"|(?<![\w.:>])" + _DEVTYPE_QUAL + r"DeviceType\b" + _CV_EAST
    + r"\s*(?:\w+\s*\{|[({])\s*(?![)}])"
)
RE_CUDA_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](?:vt/cuda/|cuda_runtime)')
RE_PP_IF = re.compile(r"^\s*#\s*(ifdef|ifndef|if)\b(.*)$")
RE_PP_ELIF = re.compile(r"^\s*#\s*elif\b(.*)$")
RE_PP_ELSE = re.compile(r"^\s*#\s*else\b")
RE_PP_ENDIF = re.compile(r"^\s*#\s*endif\b")
RE_VT_IFDEF = re.compile(r"^\s*#\s*(?:ifdef\s+VT_\w+|if\s+defined\s*\(\s*VT_\w+)")
# A preprocessor condition that makes an enclosed CUDA include legal in a
# non-CUDA build. `VT_*` gates are CUDA kernel-feature macros, only ever defined
# by the CUDA build (cmake/CudaArchFeatures.cmake).
RE_CUDA_GUARD = re.compile(r"\b(VLLM_CPP_CUDA|VT_\w+|__CUDACC__|CUDA_VERSION)\b")

RE_DSR_ALLOW = re.compile(r"//\s*DSR-ALLOW\(\s*([A-Za-z0-9_.\-]+)\s*\)\s*:\s*(\S.*?)\s*$")


# --- allowlist ---------------------------------------------------------------
#
# path -> {bucket: (expected_count | "*", reason)}
#
# "*" means the whole file IS a device leg and the bucket is not counted there.
# An integer means EXACTLY that many references are the platform definition; a
# 17th `kCUDA` in a registrar file is leakage and fails. Every entry states why
# the site is the CUDA leg rather than the shared layer branching on it.
ALLOWLIST: dict[str, dict[str, tuple[object, str]]] = {
    "src/vllm/platforms/cuda.cpp": {
        "kcuda": ("*", "IS the CUDA platform leg — `device_type()`, `backend()` "
                       "and the static `RegisterPlatform(kCUDA, …)`. Counting a "
                       "platform file's own device name would punish adding a "
                       "platform, which is the opposite of the metric's point."),
        "cuda_inc": ("*", "the CUDA platform leg is the one TU that may include "
                          "<cuda_runtime.h> unconditionally; it is compiled only "
                          "when VLLM_CPP_CUDA is on (see its CMake guard)."),
    },
    "src/vllm/platforms/platform.cpp": {
        "kcuda": (1, "the device PRIORITY WALK "
                     "`{kCUDA, kXPU, kVULKAN, kMETAL, kCPU}` — a data list that "
                     "names every platform equally, mirroring upstream's "
                     "`platforms/__init__.py` import probe."),
        "dev_cast": (1, "`FindPlatformByName`'s `static_cast<DeviceType>(i)`: the "
                        "registry is an array indexed BY DeviceType, so turning a "
                        "slot index back into its type is the seam's own inverse, "
                        "not a model file naming a device by enum value. Budgeted "
                        "at exactly one — a second cast in this file is not the "
                        "registry walk and fails."),
    },
    "include/vllm/platforms/interface.h": {
        "kcuda": (1, "the `is_cuda()` DEFINITION itself "
                     "(`device_type() == DeviceType::kCUDA`), mirroring "
                     "`vllm/platforms/interface.py:189-215`. The definition is "
                     "the seam; its 11 CALL SITES in the shared layer are not."),
        "is_cuda": (1, "same line — the definition, not a call site."),
    },
    "src/vllm/v1/attention/backend.cpp": {
        "kcuda": (2, "`(kCUDA, name)` REGISTRAR KEYS for TRITON_MLA and "
                     "FLASH_ATTN. Registration keys are how a backend declares "
                     "which device it serves — upstream's "
                     "`AttentionBackendEnum` (registry.py:34-120) spends a "
                     "closed enum entry on exactly the same thing."),
    },
    "src/vllm/v1/attention/cuda_legacy.cpp": {
        "kcuda": (1, "the `(kCUDA, \"CUDA_LEGACY\")` registrar key declares "
                     "the portable CUDA attention capability descriptor. This "
                     "is the platform leg, like backend.cpp above; an extra "
                     "device branch must still fail the exact-count guard."),
    },
    "src/vllm/v1/attention/backends/gdn_attn.cpp": {
        "kcuda": (1, "the `(kCUDA, \"GDN_ATTN\")` registrar key — same reason as "
                     "backend.cpp above."),
    },
    "src/vllm/v1/attention/backends/mla/sparse_swa.cpp": {
        "kcuda": (1, "the `(kCUDA, \"DEEPSEEK_SPARSE_SWA\")` registrar key — same "
                     "reason as backend.cpp above. The backend is host metadata "
                     "with no device code at all; the key exists because a "
                     "DeepSeek-V4 KV group NAMES it and the registry is keyed by "
                     "(DeviceType, name), so a missing kCUDA row would throw at "
                     "engine construction on the one device this row exists to "
                     "unblock (ISSUE-LOCAL-01M2EMPC6T63TVDPQ90GVPRC5F)."),
    },
    "src/vllm/v1/attention/backends/mla/indexer.cpp": {
        "kcuda": (1, "the `(kCUDA, \"DEEPSEEK_V4_INDEXER\")` registrar key — same "
                     "reason as sparse_swa.cpp above."),
    },
    "src/vllm/model_executor/models/deepseek_v4_compressor.cpp": {
        "kcuda": (1, "the `(kCUDA, \"CompressorBackend\")` registrar key — same "
                     "reason as sparse_swa.cpp above. It lives in the compressor "
                     "TU because upstream's `CompressorBackend` lives in the same "
                     "file as the layer that names it (`compressor.py:59-77`, "
                     ":189-190)."),
    },
    "src/vllm/model_executor/models/deepseek_v4_device.cpp": {
        "kcuda": (9, "the DeepSeek-V4 CUDA device-forward RESOLVER TU (W7-device): "
                     "4 `GetOp` + 5 `OpRegistered` lookups that fetch the "
                     "device-registered `kDeepseekV4{Mhc,Dsa,Compressor,Moe}` kernels "
                     "for `DeepseekV4Model::ForwardDevice`. This TU exists only to "
                     "resolve the device leg — a CPU build registers nothing on "
                     "`(op, kCUDA)` so `GetOp` throws a clean device-only error. "
                     "O34 (#3198) added one `OpRegistered(kDeepseekV4Mhc, kCUDA)` "
                     "probe to `MhcDevice()` so it can try kCUDA first and fall back "
                     "to kROCm; Dsa, Compressor and Moe stay kCUDA-only. "
                     "FOLLOW-UP (deferred, needs a GB10 re-gate — DGX offline "
                     "2026-07-29): thread the runner `DeviceType` through these "
                     "resolvers so they become device-parameterized lookups "
                     "(`GetOp(op, runner.device.type)`) instead of hardcoding kCUDA."),
    },
    "src/vllm/model_executor/models/glm5_next_device.cpp": {
        "kcuda": (3, "the GLM-5.3-Flash k-pool DSA indexer's device PROBE TU "
                     "(W9c-0, #2415) — the SAME shape as laguna_device.cpp below: "
                     "2 `OpRegistered` lookups asking whether cuda_glm5_next.cu "
                     "registered `kGlm5NextKpool{Compress,Select}` on kCUDA, plus "
                     "the `OpRegistered(kDeepseekV4Mhc, kCUDA)` leg of the mHC "
                     "pre/post device probe (#3229, O34 #3199), which tries kCUDA "
                     "first and falls back to kROCm. The TU "
                     "is always compiled and holds NO CUDA code. There is "
                     "deliberately no CPU provider for either op — the CPU answer "
                     "is `glm5_next_dsa.cpp`, which is the ORACLE these kernels are "
                     "gated against, and registering it again under the same ids "
                     "would make the seam its own golden — so a CPU build finds "
                     "nothing on `(op, kCUDA)`, the probe returns false, and "
                     "`vt::Glm5NextKpoolCompress` refuses a CPU queue BY NAME "
                     "through `GetOp`. `tests/vllm/models/"
                     "test_glm5_next_kpool_device.cpp` asserts that refusal and "
                     "asserts both ops are absent on kCPU. FOLLOW-UP (deferred, "
                     "shares the DeepSeek-V4 and Laguna row): thread the runner "
                     "`DeviceType` through this resolver so it becomes "
                     "`OpRegistered(op, runner.device.type)` instead of hardcoding "
                     "kCUDA. W9c-3 (#2410) owns the forward that would supply that "
                     "device, and until it lands there is no runner to read one "
                     "from."),
    },
    "src/vllm/model_executor/models/laguna_device.cpp": {
        "kcuda": (2, "the Laguna CUDA device-forward RESOLVER TU — the SAME shape as "
                     "deepseek_v4_device.cpp above: 1 `GetOp` + 1 `OpRegistered` "
                     "lookup fetching the `kLaguna` glue table that cuda_laguna.cu "
                     "registers on kCUDA, for the resident decode path. The TU is "
                     "always compiled and holds NO CUDA code; on a CPU-only build "
                     "nothing is registered for `(kLaguna, kCUDA)`, so "
                     "`LagunaDeviceKernelsAvailable()` returns false and the host "
                     "compose path runs. FOLLOW-UP (deferred, shares the DeepSeek-V4 "
                     "row above and needs the same GB10 re-gate): thread the runner "
                     "`DeviceType` through both resolvers so they become "
                     "`GetOp(op, runner.device.type)` instead of hardcoding kCUDA."),
    },
}


@dataclass
class Hit:
    path: str
    line: int
    bucket: str
    text: str


@dataclass
class Result:
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(BUCKETS, 0))
    per_file: dict[str, dict[str, int]] = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)
    allowed: dict[str, dict[str, int]] = field(default_factory=dict)
    exempt: list[Hit] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def strip_comments_and_strings(text: str) -> list[str]:
    """Blank out //, /* */ and "…" / '…' spans, preserving line structure.

    A device name inside a comment or a diagnostic message is prose, not a
    branch. Newlines are preserved so line numbers survive.
    """
    out: list[str] = []
    i, n = 0, len(text)
    state = "code"  # code | line_comment | block_comment | string | char
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line_comment"
                out.append("  ")
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block_comment"
                out.append("  ")
                i += 2
                continue
            if c == '"':
                state = "string"
                out.append(" ")
                i += 1
                continue
            if c == "'":
                state = "char"
                out.append(" ")
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        if state == "line_comment":
            if c == "\n":
                state = "code"
                out.append("\n")
            else:
                out.append(" ")
            i += 1
            continue
        if state == "block_comment":
            if c == "*" and nxt == "/":
                state = "code"
                out.append("  ")
                i += 2
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        # string / char
        if c == "\\" and nxt:
            out.append("  ")
            i += 2
            continue
        if (state == "string" and c == '"') or (state == "char" and c == "'"):
            state = "code"
            out.append(" ")
            i += 1
            continue
        out.append("\n" if c == "\n" else " ")
        i += 1
    return "".join(out).splitlines()


def cuda_guard_depth(raw_lines: list[str]) -> list[bool]:
    """Per line: is it inside a CUDA / VT_* preprocessor conditional?

    An unconditional `#include "vt/cuda/…"` breaks the non-CUDA build outright;
    the same include under `#ifdef VT_MARLIN_NVFP4` does not. Only the former is
    leakage.
    """
    guarded: list[bool] = []
    stack: list[bool] = []
    for line in raw_lines:
        m = RE_PP_IF.match(line)
        if m:
            stack.append(bool(RE_CUDA_GUARD.search(m.group(2))))
            guarded.append(any(stack))
            continue
        m = RE_PP_ELIF.match(line)
        if m and stack:
            stack[-1] = bool(RE_CUDA_GUARD.search(m.group(1)))
            guarded.append(any(stack))
            continue
        if RE_PP_ELSE.match(line) and stack:
            # The negative arm of a CUDA guard is the PORTABLE arm; a CUDA
            # include there would be a genuine break, so stop treating it as
            # guarded.
            stack[-1] = False
            guarded.append(any(stack))
            continue
        if RE_PP_ENDIF.match(line):
            if stack:
                stack.pop()
            guarded.append(any(stack))
            continue
        guarded.append(any(stack))
    return guarded


def source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in SCAN_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        files.extend(
            p for p in sorted(base.rglob("*")) if p.suffix in SOURCE_SUFFIXES and p.is_file()
        )
    return files


def dsr_allow_lines(raw_lines: list[str]) -> dict[int, tuple[str, str]]:
    """1-based line -> (row_id, reason) for the line the escape hatch covers.

    The marker covers its OWN line and the line directly below it, so it can sit
    above a statement that has no room for a trailing comment.
    """
    covered: dict[int, tuple[str, str]] = {}
    for idx, line in enumerate(raw_lines, start=1):
        m = RE_DSR_ALLOW.search(line)
        if not m:
            continue
        covered[idx] = (m.group(1), m.group(2))
        covered.setdefault(idx + 1, (m.group(1), m.group(2)))
    return covered


def scan(root: Path) -> Result:
    res = Result()
    for path in source_files(root):
        rel = path.relative_to(root).as_posix()
        raw = path.read_text(encoding="utf-8", errors="replace")
        raw_lines = raw.splitlines()
        code_lines = strip_comments_and_strings(raw)
        guarded = cuda_guard_depth(raw_lines)
        allow_map = dsr_allow_lines(raw_lines)
        entry = ALLOWLIST.get(rel, {})

        found: dict[str, list[Hit]] = {b: [] for b in BUCKETS}
        for lineno, code in enumerate(code_lines, start=1):
            for m in RE_KCUDA.finditer(code):
                del m
                found["kcuda"].append(Hit(rel, lineno, "kcuda", raw_lines[lineno - 1].strip()))
            for m in RE_IS_CUDA.finditer(code):
                del m
                found["is_cuda"].append(Hit(rel, lineno, "is_cuda", raw_lines[lineno - 1].strip()))
        # `dev_cast` matches over the WHOLE stripped text, not line by line: a
        # cast wrapped across two lines is the same conversion, and a
        # line-oriented matcher would read the halves as two innocent lines.
        # The match offset is mapped back to a 1-based line for reporting.
        joined = "\n".join(code_lines)
        for m in RE_DEVTYPE_CAST.finditer(joined):
            lineno = joined.count("\n", 0, m.start()) + 1
            text = raw_lines[lineno - 1].strip() if lineno <= len(raw_lines) else m.group(0)
            found["dev_cast"].append(Hit(rel, lineno, "dev_cast", text))
        for lineno, line in enumerate(raw_lines, start=1):
            if RE_CUDA_INCLUDE.match(line) and not guarded[lineno - 1]:
                found["cuda_inc"].append(Hit(rel, lineno, "cuda_inc", line.strip()))
            if RE_VT_IFDEF.match(line):
                found["vt_ifdef"].append(Hit(rel, lineno, "vt_ifdef", line.strip()))

        for bucket, hits in found.items():
            allow = entry.get(bucket)
            if allow is not None:
                expected, _reason = allow
                if expected == "*":
                    res.allowed.setdefault(rel, {})[bucket] = len(hits)
                    continue
                if len(hits) != expected:
                    res.errors.append(
                        f"ALLOWLIST STALE: {rel} bucket '{bucket}' has {len(hits)} "
                        f"reference(s) but the allowlist expects exactly {expected}. "
                        "The allowlist is itself a ratchet: if the platform leg "
                        "genuinely changed shape, edit ALLOWLIST in "
                        "scripts/check-device-leakage.py and say why in the same "
                        "commit — never widen it to absorb a new device branch."
                    )
                res.allowed.setdefault(rel, {})[bucket] = min(len(hits), int(expected))
                hits = hits[int(expected):]

            kept: list[Hit] = []
            for hit in hits:
                cover = allow_map.get(hit.line)
                if cover:
                    res.exempt.append(hit)
                else:
                    kept.append(hit)
            if kept:
                res.per_file.setdefault(rel, dict.fromkeys(BUCKETS, 0))
                res.per_file[rel][bucket] += len(kept)
                res.counts[bucket] += len(kept)
                res.hits.extend(kept)
    return res


def load_baseline() -> dict[str, int]:
    if not BASELINE_PATH.exists():
        return {}
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {b: int(data["buckets"][b]) for b in BUCKETS}


def write_baseline(res: Result) -> None:
    payload = {
        "_comment": [
            "DSR baseline for scripts/check-device-leakage.py (work row S1 of",
            ".agents/specs/accelerator-seam-audit.md). Device-Specific References",
            "in the device-agnostic shared layer (src/vllm/ + include/vllm/),",
            "net of the checker's per-file platform-leg allowlist.",
            "THIS NUMBER MAY ONLY EVER GO DOWN. Lower it in the SAME commit as the",
            "reduction that earned it, by running:",
            "  python3 scripts/check-device-leakage.py --write-baseline",
            "It is a leakage budget, never to be raised to make a failing check pass.",
        ],
        "total": res.total,
        "buckets": {b: res.counts[b] for b in BUCKETS},
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def report(res: Result, out=sys.stdout) -> None:
    print("Device-Specific References (DSR) in the shared layer "
          f"({' + '.join(SCAN_ROOTS)})", file=out)
    print("", file=out)
    width = max(len(f) for f in res.per_file) if res.per_file else 40
    print(f"{'file':<{width}}  " + "  ".join(f"{b:>9}" for b in BUCKETS) + "      DSR", file=out)
    for rel in sorted(res.per_file, key=lambda r: (-sum(res.per_file[r].values()), r)):
        row = res.per_file[rel]
        print(
            f"{rel:<{width}}  "
            + "  ".join(f"{row[b]:>9}" for b in BUCKETS)
            + f"  {sum(row.values()):>7}",
            file=out,
        )
    print("", file=out)
    print(f"{'TOTAL':<{width}}  "
          + "  ".join(f"{res.counts[b]:>9}" for b in BUCKETS)
          + f"  {res.total:>7}", file=out)
    print("", file=out)
    if res.allowed:
        print("Allowlisted platform-leg sites (NOT counted — these ARE the CUDA leg):",
              file=out)
        for rel in sorted(res.allowed):
            for bucket, n in sorted(res.allowed[rel].items()):
                reason = ALLOWLIST[rel][bucket][1]
                print(f"  {rel} [{bucket}] x{n}: {reason}", file=out)
        print("", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", action="store_true", help="print the per-file breakdown")
    ap.add_argument("--list", action="store_true", help="print every counted site")
    ap.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the baseline to the measured counts (only ever DOWNWARD)",
    )
    ap.add_argument("--root", default=str(ROOT), help="tree to scan (testing)")
    args = ap.parse_args(argv)

    res = scan(Path(args.root))

    if args.report:
        report(res)
    if args.list:
        for hit in res.hits:
            print(f"{hit.path}:{hit.line}: [{hit.bucket}] {hit.text}")
        print("")

    # The escape hatch is always LOUD: printed on every run, pass or fail, so a
    # growing exception budget is visible in CI output.
    print(f"DSR-ALLOW exemptions in force: {len(res.exempt)}")
    for hit in res.exempt:
        print(f"  DSR-ALLOW {hit.path}:{hit.line} [{hit.bucket}] {hit.text}")

    print("DSR by bucket: "
          + ", ".join(f"{b}={res.counts[b]}" for b in BUCKETS)
          + f"  -> total {res.total}")

    if args.write_baseline:
        prev = load_baseline()
        if prev and res.total > sum(prev.values()):
            print("REFUSING to write a HIGHER baseline "
                  f"({sum(prev.values())} -> {res.total}). The DSR ratchet only "
                  "turns one way: reduce the leakage instead.", file=sys.stderr)
            return 1
        write_baseline(res)
        try:
            shown = BASELINE_PATH.relative_to(ROOT)
        except ValueError:  # a scratch baseline under --root (mutation suite)
            shown = BASELINE_PATH
        print(f"baseline written: {shown} -> {res.total}")
        return 0

    errors = list(res.errors)
    baseline = load_baseline()
    if not baseline:
        errors.append(
            f"no baseline at {BASELINE_PATH}; run --write-baseline to establish one"
        )
    else:
        for bucket in BUCKETS:
            got, want = res.counts[bucket], baseline[bucket]
            if got > want:
                errors.append(
                    f"DSR REGRESSION in bucket '{bucket}': {got} > baseline {want}. "
                    "A device-specific reference was added to the device-agnostic "
                    "shared layer. Ask the op/provider table the question instead "
                    "(vt::OpRegistered / Platform capability), or — if the site is "
                    "genuinely the platform leg — add it to ALLOWLIST with a "
                    "reason. NEVER raise the baseline to make this pass."
                )
            elif got < want:
                errors.append(
                    f"DSR baseline STALE in bucket '{bucket}': {got} < baseline "
                    f"{want}. A reduction must lower the baseline in the SAME "
                    "commit: run `python3 scripts/check-device-leakage.py "
                    "--write-baseline` and commit the result."
                )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        print(f"\ncheck-device-leakage: FAIL ({len(errors)} error(s))", file=sys.stderr)
        return 1

    print(f"check-device-leakage: OK (DSR {res.total} == baseline "
          f"{sum(baseline.values())}, ratchet holds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
