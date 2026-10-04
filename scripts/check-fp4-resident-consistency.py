#!/usr/bin/env python3
"""Fail if an fp4 resident-upload copy drops the ENG-LOAD-DIRECT-UPLOAD post-upload step.

THE DEFECT THIS EXISTS TO PREVENT ALREADY HAPPENED, TWICE OVER. `ResidentNvfp4` is
the ONE host->device move of a compressed-tensors fp4 weight's `packed`/`scale`,
because `LoadCtNvfp4W4A16` / `LoadCtMxfp4W4A16` / `LoadCtNvfp4Raw` BORROW those
buffers from the safetensors mmap (issue #150) instead of copying them. Round 2 of
ENG-LOAD-DIRECT-UPLOAD therefore gave every fp4 upload three statements per buffer:

  1. `load_stats::AddDeviceUpload(nb)`         — account the move. These bytes are
     counted NOWHERE else: they were borrowed on the way in, so the host-copy
     counter never saw them.
  2. `w.<buf>.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); })`
     — PUBLISH the allocation on the OwnedTensor. `AdoptDeviceBytesAsHost` keys on
     `d_dev` and returns immediately when it is null, so without this the next
     statement is a silent no-op. `d_dev` is ALSO the one owner of those bytes:
     the type-specific `Nvfp4Weight::d_<handle>` fields this used to publish
     alongside are gone, because a second owning handle is what leaked a full
     packed+scale device copy per repacked weight (the repack builders reset only
     one of the two). The release clause below pins that.
  3. `AdoptDeviceBytesAsHost(d.b, w.<buf>)`    — the post-upload residency step:
     release the consumed source pages, and on a host-addressable device (GB10
     unified memory under Vulkan) adopt the device allocation as the host view so
     the model is not resident TWICE.

A fresh reviewer then reverted all six statements — three per buffer, in BOTH copies
of the function — rebuilt clean, and ran every suite touching fp4 residency or the
load counters: 20/20 still passed. `tests/vllm/test_load_direct_upload.cpp` now pins
the SHARED copy at run time (it drives `dense_nvfp4::ResidentNvfp4` over a fake
host-addressable backend and an observable stand-in mapping, and goes red under each
of those mutations). This checker exists for the OTHER copy.

WHY A SOURCE CHECK FOR THAT ONE. `src/vllm/model_executor/models/qwen3_5.cpp` keeps
its own `ResidentNvfp4` — recorded, deliberate duplication (see the preamble of
include/vllm/model_executor/models/dense_nvfp4_gemm.h: unifying the two device-glue
families is a separate refactor that would touch the 27B/35B gate models' hot path).
It lives in an ANONYMOUS namespace inside an 8.5k-line translation unit, so no test
can call it and no runtime assertion can reach it. The only mechanism that bites when
it silently diverges from the copy the runtime gate pins is a structural one.

THE INVARIANT is checked PER BUFFER, inside that buffer's own upload block — the
`if (!w.d_packed) { ... }` / `if (!w.d_scale) { ... }` guard both copies are written
as. Body-wide matching was the first version of this file and it was WRONG: one
surviving `AddDeviceUpload` anywhere in the function satisfied the "counted" clause
for BOTH buffers, so dropping exactly one of the two counters passed. Everything
below is scoped to one block, and each clause is bound to that block's buffer:

  (a) COUNTED    — `load_stats::AddDeviceUpload(nb)` where `nb` is bound in the same
                   block to `w.<buf>.bytes.size()` (so `AddDeviceUpload(0)`, or the
                   other buffer's byte count, is not a pass);
  (b) COPIED     — the upload reads `w.<buf>.bytes.data()` (not the other buffer's);
  (c) PUBLISHED  — `w.<buf>.d_dev = <device-allocation expr>`: a `std::shared_ptr`
                   construction whose deleter returns the block through the vt
                   Backend (`...Free(...)`). `= nullptr`, `= {}`, and a bare
                   `= <object>.<other>.d_dev` (which owns nothing) are not a pass;
  (d) ADOPTED    — `AdoptDeviceBytesAsHost(..., w.<buf>)` on THIS function's weight
                   parameter (so adopting another object's buffer is not a pass);
  (e) ORDERED    — (c) precedes (d), which is what makes (d) more than a no-op;
  (f) READ-FIRST — (b) precedes (d). The adoption RE-POINTS `w.<buf>.bytes` at the
                   device allocation and releases the source pages, so an upload
                   copy sequenced after it reads pages that have already been
                   handed back. SINKING the `Copy` to below the adopt is what used
                   to pass; the opposite motion — HOISTING the adopt above the
                   `Copy` — lifts it above (c) as well and was already caught by
                   (e), so only the sink shape shows what (f) adds.

  (g) RELEASED   — `Nvfp4Weight::ReleaseResident()` resets `packed.d_dev` AND
                   `scale.d_dev`, and where it drops an adopted host twin for a
                   buffer that drop precedes that buffer's reset. THIS IS THE
                   CLAUSE THE LEAK VIOLATED: the repack builders reset only the
                   type-specific handle, so the surviving `d_dev` alias kept the
                   block for the process lifetime. It is checked in
                   include/vllm/model_executor/models/qwen3_5_weights.h, the one
                   definition both upload copies release through, because the
                   release is not inside the upload function the clauses above
                   scope to.

TEXT THE COMPILER NEVER SEES IS NOT A PASS. Every clause runs against
`checker_text.normalize_source`, which blanks `//` and `/* */` comments, `#if 0` /
`#if false` regions and `if (false)` / `if (0)` branches before matching, keeping
byte offsets and line numbers intact so the ordering clauses and the `file:line`
report still describe the real file. Without that, `// AdoptDeviceBytesAsHost(d.b,
w.packed);` read as the statement being present — a commented-out step is precisely
the silent divergence this gate exists to catch, and it passed. Conditions other
than a literal false are NOT evaluated: a region under `#ifdef VT_CUTLASS_NVFP4` is
a real build configuration, not a disguised deletion.

WHAT THIS GATE DOES *NOT* DO, stated plainly so the record does not imply more. It is
a STRUCTURAL check over text: it proves the per-buffer statements and the single
release are present in code the compiler keeps, bound to the right buffer of the right
object, and that the adoption follows both the copy and the publication. It cannot
prove they are CORRECT at run
time — that the pointer published is the one that was uploaded, that the byte count
matches the allocation, or that the copy transferred the right bytes. Nor does it
model the preprocessor: a statement moved under a build-configuration `#ifdef` is
out of its reach. The qwen3_5.cpp duplicate is therefore guarded against DELETION —
including deletion disguised as a comment, an `#if 0`, or a never-taken branch —
and against gross substitution and mis-ordering, not against arbitrary corruption.
Run-time proof exists only for the shared copy, via
tests/vllm/test_load_direct_upload.cpp; extending it to this one means making the
duplicate reachable, which is the unification refactor this row explicitly does not
attempt.

Pure functions over text, so they are unit- and mutation-testable
(tests/scripts/test_check_fp4_resident_consistency.py), mirroring
check-fusion-consistency.py and check-gemv-invocation-consistency.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The shared normalization the whole check-* family needs. `scripts/` is sys.path[0]
# when this file is RUN, but not when a test loads it by path, so pin it either way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checker_text import match_braces as _match_braces  # noqa: E402
from checker_text import normalize_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The two copies of the fp4 resident upload. Both are in scope: the shared one is
# also pinned at run time, and checking it here keeps the two descriptions of the
# invariant from drifting apart in opposite directions.
SOURCES = (
    Path("include/vllm/model_executor/models/dense_nvfp4_gemm.h"),
    Path("src/vllm/model_executor/models/qwen3_5.cpp"),
)

# The ONE release of an Nvfp4Weight's device resident. Both upload copies above are
# released through it, so a release that forgets a buffer leaks from BOTH.
RELEASE_SOURCE = Path("include/vllm/model_executor/models/qwen3_5_weights.h")

# The one release function both upload copies are released through.
RELEASE_FUNCTION = "ReleaseResident"

# The buffers an Nvfp4Weight uploads. Each needs the full step, in its OWN block.
BUFFERS = ("packed", "scale")

# The function whose body carries the invariant.
FUNCTION = "ResidentNvfp4"

# A definition head `... ResidentNvfp4(Dev d, const Nvfp4Weight& w) {`. Anchored on
# the parameter list so the many CALL sites (`ResidentNvfp4(d, w)`) never match, and
# so the neighbouring `ResidentNvfp4Qkv` / `ResidentNvfp4GateUp` / `ResidentNvfp4Alpha`
# / `ResidentNvfp4ScaleSwizzled` overloads — which build MERGED device operands out of
# several borrowed tensors and are explicitly out of this row's scope — do not either.
# The weight parameter's NAME is captured: every clause below is bound to it, so a
# statement that operates on some other Nvfp4Weight cannot satisfy this one.
_DEF = re.compile(
    r"\b"
    + re.escape(FUNCTION)
    + r"\s*\(\s*Dev\s+\w+\s*,\s*const\s+Nvfp4Weight\s*&\s*(?P<recv>\w+)\s*\)\s*\{"
)

# Null-ish right-hand sides a `d_dev` publication must not have. Kept explicit
# rather than inferred: the publication is only meaningful if it hands over THIS
# buffer's device handle, which the per-buffer check below requires outright.
_NULLISH = ("nullptr", "NULL", "{}", "0")


def _line_no(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _statement_end(text: str, start: int) -> int:
    """Index of the `;` that ends the statement beginning at `start`, tracking
    bracket depth so a semicolon INSIDE a lambda body does not truncate it.

    The published right-hand side is now `std::shared_ptr<void>(p, [bk](void* q)
    { bk->Free(q); })`, whose inner brace holds a `;`. A flat `[^;]+;` capture —
    what this checker used while the rhs was a bare member name — stops there and
    sees no deleter, so the check would have failed on the real, correct shape."""
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ";" and depth == 0:
            return i
    return -1


def function_bodies(text: str) -> list[tuple[int, str, str]]:
    """Every `ResidentNvfp4(Dev, const Nvfp4Weight&)` as (line_no, recv, body),
    the body delimited by brace matching from the definition's opening brace."""
    bodies: list[tuple[int, str, str]] = []
    for m in _DEF.finditer(text):
        start = m.end()  # just past the opening brace
        end = _match_braces(text, start)
        bodies.append((_line_no(text, m.start()), m.group("recv"), text[start : end - 1]))
    return bodies


def checked_bodies(text: str) -> list[tuple[int, str, str]]:
    """`function_bodies` over the text the COMPILER keeps — the ONE place raw source
    is normalized, so no caller can reach a clause matcher with a commented-out or
    `#if 0`-ed statement still in it. `normalize_source` is position-preserving, so
    the line numbers returned here are the raw file's."""
    return function_bodies(normalize_source(text))


def buffer_block(body: str, recv: str, buffer: str) -> str | None:
    """The `if (!<recv>.<buffer>.d_dev) { ... }` upload block for ONE buffer, or
    None.

    This is the scope every clause is checked in. Checking the whole body instead
    was the original bug: a single surviving statement served both buffers. The
    guard is the single-owner slot rather than the deleted `d_<buffer>` handle:
    the upload is skipped exactly when the device resident already exists."""
    guard = re.compile(
        r"if\s*\(\s*!\s*"
        + re.escape(recv)
        + r"\s*\.\s*"
        + re.escape(buffer)
        + r"\s*\.\s*d_dev\s*\)\s*\{"
    )
    m = guard.search(body)
    if m is None:
        return None
    return body[m.end() : _match_braces(body, m.end()) - 1]


def _counted(block: str, recv: str, buffer: str) -> bool:
    """`AddDeviceUpload(nb)` with `nb` bound HERE to `<recv>.<buffer>.bytes.size()`."""
    size_expr = (
        re.escape(recv) + r"\s*\.\s*" + re.escape(buffer) + r"\s*\.\s*bytes\s*\.\s*size\s*\(\s*\)"
    )
    for m in re.finditer(r"\bload_stats::AddDeviceUpload\s*\(\s*(?P<arg>[^()]*?)\s*\)", block):
        arg = m.group("arg").strip()
        if re.fullmatch(size_expr, arg):
            return True
        if re.fullmatch(r"\w+", arg) and re.search(
            r"\b" + re.escape(arg) + r"\s*=\s*" + size_expr, block
        ):
            return True
    return False


def _copied(block: str, recv: str, buffer: str) -> int | None:
    """Offset of the upload that reads THIS buffer's bytes (not the other one's),
    or None. The OFFSET, not a bool, because clause (f) orders it against (d)."""
    m = re.search(
        r"\.\s*Copy\s*\([^;]*?"
        + re.escape(recv)
        + r"\s*\.\s*"
        + re.escape(buffer)
        + r"\s*\.\s*bytes\s*\.\s*data\s*\(\s*\)",
        block,
    )
    return None if m is None else m.start()


def _published(block: str, recv: str, buffer: str) -> int | None:
    """Offset of `<recv>.<buffer>.d_dev = <device allocation>`, or None.

    The published value must OWN the block: a `std::shared_ptr` construction whose
    deleter returns it through the vt Backend. A bare `= nullptr`, `= {}` or an
    alias of some other object's `d_dev` publishes no owner, so
    `AdoptDeviceBytesAsHost` either returns immediately or the block outlives the
    weight with nobody to free it."""
    for m in re.finditer(
        re.escape(recv)
        + r"\s*\.\s*"
        + re.escape(buffer)
        + r"\s*\.\s*d_dev\s*=\s*",
        block,
    ):
        end = _statement_end(block, m.end())
        if end < 0:
            continue
        rhs = block[m.end() : end].strip()
        if rhs in _NULLISH:
            continue
        if re.search(r"\bshared_ptr\s*<\s*void\s*>\s*\(", rhs) and re.search(
            r"\bFree\s*\(", rhs
        ):
            return m.start()
    return None


def _adopted(block: str, recv: str, buffer: str) -> int | None:
    """Offset of `AdoptDeviceBytesAsHost(<backend>, <recv>.<buffer>)`, or None."""
    m = re.search(
        r"\bAdoptDeviceBytesAsHost\s*\(\s*[^,)]+,\s*"
        + re.escape(recv)
        + r"\s*\.\s*"
        + re.escape(buffer)
        + r"\s*\)",
        block,
    )
    return None if m is None else m.start()


def body_violations(body: str, recv: str = "w") -> list[str]:
    """Every way ONE ResidentNvfp4 body breaks the invariant. Empty == it holds."""
    problems: list[str] = []

    for buffer in BUFFERS:
        block = buffer_block(body, recv, buffer)
        if block is None:
            problems.append(
                f"no `if (!{recv}.{buffer}.d_dev)` upload block for `{buffer}`. "
                f"Each buffer's post-upload step is checked inside its OWN block, "
                f"because body-wide matching lets one buffer's statements satisfy the "
                f"other's. If this copy was deliberately restructured, update "
                f"scripts/check-fp4-resident-consistency.py in the same change"
            )
            continue

        if not _counted(block, recv, buffer):
            problems.append(
                f"`{buffer}`'s host->device upload is NOT counted: no "
                f"`load_stats::AddDeviceUpload(...)` in its block over "
                f"`{recv}.{buffer}.bytes.size()`. These bytes were BORROWED from the "
                f"shard mmap, so this is the only counter that can ever see them"
            )
        copied = _copied(block, recv, buffer)
        if copied is None:
            problems.append(
                f"`{buffer}`'s block does not upload `{recv}.{buffer}.bytes.data()`: "
                f"the device buffer published for `{buffer}` is filled from somewhere "
                f"else, or from nothing"
            )

        pub = _published(block, recv, buffer)
        if pub is None:
            problems.append(
                f"`{buffer}` does not PUBLISH a backend-owned device allocation on "
                f"`{recv}.{buffer}.d_dev` (a std::shared_ptr whose deleter returns "
                f"the block through the vt Backend); AdoptDeviceBytesAsHost keys "
                f"on `d_dev` and would return immediately"
            )
        adopt = _adopted(block, recv, buffer)
        if adopt is None:
            problems.append(
                f"`{buffer}` skips the post-upload residency step "
                f"(`AdoptDeviceBytesAsHost(d.b, {recv}.{buffer})`): its consumed source "
                f"pages are never released and, on a host-addressable device, the "
                f"model stays resident twice"
            )
        else:
            if pub is not None and pub > adopt:
                problems.append(
                    f"`{buffer}` publishes `d_dev` AFTER its AdoptDeviceBytesAsHost "
                    f"call, so the adoption sees a null `d_dev` and silently does "
                    f"nothing"
                )
            if copied is not None and copied > adopt:
                problems.append(
                    f"`{buffer}` runs AdoptDeviceBytesAsHost BEFORE the upload copy "
                    f"reads `{recv}.{buffer}.bytes.data()`: the adoption re-points "
                    f"`bytes` at the device allocation and releases the consumed "
                    f"source pages, so the copy reads them after they were handed "
                    f"back"
                )
    return problems


def file_violations(text: str, label: str) -> list[str]:
    """Every violation in one file, each already prefixed with `label:line`."""
    bodies = checked_bodies(text)
    if not bodies:
        return [
            f"{label}: no `{FUNCTION}(Dev, const Nvfp4Weight&)` definition found. "
            f"If this copy was deliberately removed or renamed, update SOURCES in "
            f"scripts/check-fp4-resident-consistency.py in the same change."
        ]
    return [
        f"{label}:{line_no}: {problem}"
        for line_no, recv, body in bodies
        for problem in body_violations(body, recv)
    ]


_RELEASE_DEF = re.compile(
    r"\bvoid\s+" + re.escape(RELEASE_FUNCTION) + r"\s*\(\s*\)\s*const\s*\{"
)


def release_body(text: str) -> tuple[int, str] | None:
    """The `ReleaseResident() const` body as (line_no, normalized body), or None."""
    norm = normalize_source(text)
    m = _RELEASE_DEF.search(norm)
    if m is None:
        return None
    end = _match_braces(norm, m.end())
    return _line_no(text, m.start()), norm[m.end() : end - 1]


def release_violations(text: str, label: str) -> list[str]:
    """Every way `ReleaseResident` leaves a device resident un-owned, and the one
    ordering that makes the release a no-op on a host-addressable device."""
    found = release_body(text)
    if found is None:
        return [
            f"{label}: no `void {RELEASE_FUNCTION}() const` definition. It is the ONE "
            f"release of an Nvfp4Weight's device resident (both upload copies above "
            f"use it); if it was deliberately renamed or moved, update RELEASE_SOURCE "
            f"in scripts/check-fp4-resident-consistency.py in the same change."
        ]
    line_no, body = found
    problems: list[str] = []
    for buffer in BUFFERS:
        reset = re.search(
            r"\b" + re.escape(buffer) + r"\s*\.\s*d_dev\s*\.\s*reset\s*\(\s*\)", body
        )
        if reset is None:
            problems.append(
                f"`{buffer}.d_dev` is never reset: after a Marlin repack dropped the "
                f"fp4 originals, this block stays owned for the process lifetime — "
                f"one leaked packed+scale device copy per repacked weight and per "
                f"expert. ReleaseResident must reset BOTH buffers"
            )
            continue
        twin = re.search(r"\b" + re.escape(buffer) + r"\s*\.\s*ReleaseHost\s*\(\s*\)", body)
        if twin is None:
            problems.append(
                f"`{buffer}`'s adopted host twin is never dropped: on a "
                f"host-addressable device that view holds the same block alive, so "
                f"the reset below frees nothing and the release leaks exactly as it "
                f"did when the type-specific handle was the only one reset"
            )
        elif twin.start() > reset.start():
            problems.append(
                f"`{buffer}`'s adopted host twin is dropped AFTER its `d_dev` "
                f"reset: the view holds the block alive on a host-addressable "
                f"device, so the reset frees nothing and the later drop touches a "
                f"view whose owner is gone"
            )
    return [f"{label}:{line_no}: {p}" for p in problems]


def main() -> int:
    violations: list[str] = []
    checked = 0
    for rel in SOURCES:
        path = ROOT / rel
        if not path.exists():
            print(f"ERROR: {rel} not found", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        checked += len(checked_bodies(text))
        violations.extend(file_violations(text, str(rel)))

    rel = str(RELEASE_SOURCE)
    path = ROOT / RELEASE_SOURCE
    if not path.exists():
        print(f"ERROR: {rel} not found", file=sys.stderr)
        return 1
    violations.extend(
        release_violations(path.read_text(encoding="utf-8", errors="ignore"), rel)
    )

    if violations:
        print(
            "ERROR: an fp4 resident upload or its single release drops part of the "
            "ENG-LOAD-DIRECT-UPLOAD post-upload step (issue #150):",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            "Every ResidentNvfp4 must COUNT its upload, COPY from that buffer, "
            "PUBLISH the allocation on the OwnedTensor, and then run "
            "AdoptDeviceBytesAsHost — for BOTH `packed` and `scale`, each inside its "
            "own `if (!w.<buf>.d_dev)` block — and `ReleaseResident()` must reset BOTH "
            "`d_dev` slots, dropping an adopted twin view first. The shared copy is "
            "pinned at run time by tests/vllm/test_load_direct_upload.cpp; this gate "
            "exists because the qwen3_5.cpp duplicate sits in an anonymous namespace "
            "no test can reach.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: {checked} ResidentNvfp4 definition(s) across {len(SOURCES)} file(s) count "
        f"the upload, copy, publish d_dev, and adopt — per buffer, for both packed and "
        f"scale — and ReleaseResident() resets both d_dev slots."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
