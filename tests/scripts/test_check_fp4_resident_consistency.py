#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-fp4-resident-consistency.py.

The checker is the ONLY mechanism that bites when the `ResidentNvfp4` copy in
src/vllm/model_executor/models/qwen3_5.cpp silently drops part of the
ENG-LOAD-DIRECT-UPLOAD post-upload step (issue #150): that copy sits in an
anonymous namespace inside an 8.5k-line translation unit, so no test can call it.
These cases mutate a miniature of the real function to prove the checker actually
detects each dropped statement, and then run it against the LIVE tree so a rename
or a refactor that makes the invariant unreachable cannot pass silently.

EVERY MUTATION DROPS EXACTLY ONE THING. The first version of this suite only ever
dropped BOTH `AddDeviceUpload` calls at once, which is why it could not see that
`_COUNT` was matched body-wide: one surviving call satisfied the clause for both
buffers, so dropping exactly one of them passed. The drop-exactly-one cases below
are the regression for that.

A DELETION DOES NOT HAVE TO LOOK LIKE ONE. The second round of this file only ever
DELETED text, so it could not see that the clause matchers ran on RAW source: a
statement left behind as `// AdoptDeviceBytesAsHost(d.b, w.packed);`, inside `#if 0`,
or inside `if (false)` is gone as far as the compiler is concerned and was a PASS.
The `Disguised*` cases are the regression for that, and the `#ifdef` case pins the
opposite direction — a real build configuration is not a deletion and must stay
green, which is the limitation this gate declares rather than enforces.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-fp4-resident-consistency.py"
SPEC = importlib.util.spec_from_file_location("check_fp4_resident_consistency", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

function_bodies = mod.function_bodies
buffer_block = mod.buffer_block
body_violations = mod.body_violations
file_violations = mod.file_violations
checked_bodies = mod.checked_bodies


# A miniature of the real ResidentNvfp4 carrying the full step for both buffers,
# each inside its own `if (!w.d_<buf>)` upload block. Every mutation below changes
# exactly one thing in it.
GOOD = """
Nvfp4Dev ResidentNvfp4(Dev d, const Nvfp4Weight& w) {
  if (!w.packed.d_dev) {
    const size_t pb = w.packed.bytes.size();
    void* p = d.b.Alloc(pb);
    vllm::load_stats::AddDeviceUpload(pb);
    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);
    Backend* bk = &d.b;
    w.packed.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    AdoptDeviceBytesAsHost(d.b, w.packed);
  }
  if (!w.scale.d_dev) {
    const size_t sb = w.scale.bytes.size();
    void* p = d.b.Alloc(sb);
    vllm::load_stats::AddDeviceUpload(sb);
    d.b.Copy(d.q, p, w.scale.bytes.data(), sb);
    Backend* bk = &d.b;
    w.scale.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
    AdoptDeviceBytesAsHost(d.b, w.scale);
  }
  Nvfp4Dev r;
  return r;
}
"""

# The ONE release both upload copies above are released through. A miniature of
# `Nvfp4Weight::ReleaseResident`: the two `d_dev` resets are the ownership, the two
# guarded `ReleaseHost()` drops are the adopted host twin that holds the same block
# alive on a host-addressable device and therefore has to go FIRST.
GOOD_RELEASE = """
void ReleaseResident() const {
  if (packed.HostViewIsDeviceTwin()) packed.ReleaseHost();
  if (scale.HostViewIsDeviceTwin()) scale.ReleaseHost();
  packed.d_dev.reset();
  scale.d_dev.reset();
}
"""

PACKED_PUBLISH = "    w.packed.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });\n"
SCALE_PUBLISH = "    w.scale.d_dev = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });\n"


def body_of(text: str) -> str:
    """The single ResidentNvfp4 body, through the SAME normalization `file_violations`
    applies — so a mutation that only comments a statement out is seen here exactly
    as the checker sees it in the real file."""
    bodies = checked_bodies(text)
    assert len(bodies) == 1, f"expected one body, got {len(bodies)}"
    return bodies[0][2]


def mutate(old: str, new: str = "", text: str = GOOD) -> str:
    """Replace `old` ONCE, asserting it was unique first — a near-duplicate anchor
    silently mutating the wrong statement is how a mutation suite lies."""
    assert text.count(old) == 1, f"anchor is not unique ({text.count(old)}): {old!r}"
    return text.replace(old, new, 1)


class BodyExtractionTests(unittest.TestCase):
    def test_finds_the_definition(self) -> None:
        bodies = function_bodies(GOOD)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0][0], 2)
        self.assertEqual(bodies[0][1], "w")  # the captured weight parameter name
        self.assertIn("AdoptDeviceBytesAsHost(d.b, w.scale);", bodies[0][2])

    def test_call_sites_are_not_definitions(self) -> None:
        # `ResidentNvfp4(d, w)` appears dozens of times in qwen3_5.cpp. Matching a
        # call would give the checker a body it cannot reason about.
        self.assertEqual(function_bodies("Nvfp4Dev dw = ResidentNvfp4(d, w);"), [])

    def test_sibling_overloads_are_out_of_scope(self) -> None:
        # ResidentNvfp4Qkv / GateUp / Alpha / ScaleSwizzled build MERGED device
        # operands out of several borrowed tensors; an adoption is not representable
        # there and the row explicitly leaves them named-but-unclaimed. None of them
        # may be pulled in by the name prefix they share with ResidentNvfp4.
        for text in (
            "Nvfp4QkvDev ResidentNvfp4Qkv(Dev d, const FullAttnLayerWeights& w) { return r; }",
            "Nvfp4GateUpDev ResidentNvfp4GateUp(Dev d, const DenseMlpWeights& w) { return r; }",
            "Tensor ResidentNvfp4Alpha(Dev d, const Nvfp4Weight& w) { return t; }",
            "Tensor ResidentNvfp4ScaleSwizzled(Dev d, const Nvfp4Weight& w) { return t; }",
        ):
            self.assertEqual(function_bodies(text), [], text)

    def test_nested_braces_do_not_truncate_the_body(self) -> None:
        # The real body contains lambdas and if-blocks; brace matching must reach the
        # end, or the trailing `scale` statements would look absent.
        self.assertIn("w.scale.d_dev", body_of(GOOD))

    def test_a_renamed_weight_parameter_is_followed(self) -> None:
        # The clauses are bound to the parameter NAME, so a copy that calls it `nw`
        # must still pass on its own terms rather than going vacuously red.
        text = GOOD.replace("Nvfp4Weight& w)", "Nvfp4Weight& nw)").replace("w.", "nw.")
        bodies = function_bodies(text)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0][1], "nw")
        self.assertEqual(body_violations(bodies[0][2], bodies[0][1]), [])


class BufferBlockTests(unittest.TestCase):
    """The scoping that makes the invariant PER BUFFER rather than body-wide."""

    def test_each_buffer_gets_its_own_block(self) -> None:
        body = body_of(GOOD)
        packed = buffer_block(body, "w", "packed")
        scale = buffer_block(body, "w", "scale")
        self.assertIsNotNone(packed)
        self.assertIsNotNone(scale)
        # Neither block may contain the other buffer's statements — that overlap is
        # precisely what let one AddDeviceUpload satisfy both clauses.
        assert packed is not None and scale is not None
        self.assertIn("AdoptDeviceBytesAsHost(d.b, w.packed);", packed)
        self.assertNotIn("w.scale", packed)
        self.assertIn("AdoptDeviceBytesAsHost(d.b, w.scale);", scale)
        self.assertNotIn("w.packed", scale)

    def test_a_missing_upload_block_is_a_violation(self) -> None:
        # A restructure the checker cannot scope must go RED and say so, never pass
        # by silently falling back to body-wide matching.
        text = mutate("if (!w.scale.d_dev) {", "if (true) {")
        problems = body_violations(body_of(text))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("no `if (!w.scale.d_dev)` upload block", problems[0])


class InvariantTests(unittest.TestCase):
    def test_the_real_shape_passes(self) -> None:
        self.assertEqual(body_violations(body_of(GOOD)), [])

    # --- (a) COUNTED, per buffer ------------------------------------------------
    def test_dropping_ONLY_the_packed_upload_counter_fails(self) -> None:
        # THE REGRESSION for the body-wide `_COUNT` bug: the surviving `scale`
        # counter must not satisfy `packed`.
        text = mutate("    vllm::load_stats::AddDeviceUpload(pb);\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed`'s host->device upload is NOT counted" in p for p in problems), problems)
        self.assertFalse(any("`scale`'s host->device upload is NOT counted" in p for p in problems), problems)

    def test_dropping_ONLY_the_scale_upload_counter_fails(self) -> None:
        text = mutate("    vllm::load_stats::AddDeviceUpload(sb);\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale`'s host->device upload is NOT counted" in p for p in problems), problems)
        self.assertFalse(any("`packed`'s host->device upload is NOT counted" in p for p in problems), problems)

    def test_dropping_both_upload_counters_fails(self) -> None:
        # Mutation (1): the upload stops being accounted. These bytes were BORROWED
        # from the shard mmap, so no other counter can ever see them.
        text = mutate("    vllm::load_stats::AddDeviceUpload(pb);\n")
        text = mutate("    vllm::load_stats::AddDeviceUpload(sb);\n", "", text)
        problems = body_violations(body_of(text))
        self.assertEqual(sum("is NOT counted" in p for p in problems), 2, problems)

    def test_counting_a_literal_zero_fails(self) -> None:
        # Present-but-meaningless: the call survives review by eye and accounts for
        # nothing. The argument has to be THIS buffer's byte count.
        text = mutate(
            "    vllm::load_stats::AddDeviceUpload(sb);\n",
            "    vllm::load_stats::AddDeviceUpload(0);\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale`'s host->device upload is NOT counted" in p for p in problems), problems)

    def test_counting_the_other_buffers_size_fails(self) -> None:
        # `pb` is not bound inside the `scale` block, so charging the scale upload
        # to the packed byte count is not a pass either.
        text = mutate(
            "    vllm::load_stats::AddDeviceUpload(sb);\n",
            "    vllm::load_stats::AddDeviceUpload(pb);\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale`'s host->device upload is NOT counted" in p for p in problems), problems)

    # --- (b) COPIED, per buffer -------------------------------------------------
    def test_uploading_the_other_buffers_bytes_fails(self) -> None:
        text = mutate(
            "    d.b.Copy(d.q, p, w.scale.bytes.data(), sb);\n",
            "    d.b.Copy(d.q, p, w.packed.bytes.data(), sb);\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("does not upload `w.scale.bytes.data()`" in p for p in problems), problems)

    # --- (c) PUBLISHED, per buffer ----------------------------------------------
    def test_dropping_the_packed_publication_fails(self) -> None:
        # Mutation (2): `d_dev` is never published, so AdoptDeviceBytesAsHost returns
        # immediately and the adoption below it is a silent no-op.
        text = mutate(PACKED_PUBLISH)
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed` does not PUBLISH" in p for p in problems), problems)
        self.assertFalse(any("`scale` does not PUBLISH" in p for p in problems), problems)

    def test_dropping_the_scale_publication_fails(self) -> None:
        text = mutate(SCALE_PUBLISH)
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale` does not PUBLISH" in p for p in problems), problems)

    def test_publishing_a_null_handle_fails(self) -> None:
        # The statement is still there; it publishes nothing. AdoptDeviceBytesAsHost
        # keys on `d_dev` and returns on null, so this is the deletion in disguise.
        text = mutate(PACKED_PUBLISH, "    w.packed.d_dev = nullptr;\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed` does not PUBLISH" in p for p in problems), problems)

    def test_publishing_a_bare_alias_owns_nothing_fails(self) -> None:
        # The statement is present and non-null; it just claims some other object's
        # device resident without owning it and without a deleter, which is the
        # two-owner shape this row deleted, one indirection further along.
        text = mutate(SCALE_PUBLISH, "    w.scale.d_dev = w.packed.d_dev;\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale` does not PUBLISH" in p for p in problems), problems)

    # --- (d) ADOPTED, per buffer ------------------------------------------------
    def test_dropping_the_packed_adoption_fails(self) -> None:
        # Mutation (3): the post-upload residency step is skipped, so the consumed
        # source pages are never released and the model stays resident twice.
        text = mutate("    AdoptDeviceBytesAsHost(d.b, w.packed);\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed` skips" in p for p in problems), problems)
        self.assertFalse(any("`scale` skips" in p for p in problems), problems)

    def test_dropping_the_scale_adoption_fails(self) -> None:
        text = mutate("    AdoptDeviceBytesAsHost(d.b, w.scale);\n")
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale` skips" in p for p in problems), problems)

    def test_adopting_another_objects_buffer_fails(self) -> None:
        # The call is present and adopts a `packed` — just not THIS weight's.
        text = mutate(
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    AdoptDeviceBytesAsHost(d.b, other.packed);\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed` skips" in p for p in problems), problems)

    # --- (e) ORDERED ------------------------------------------------------------
    def test_publishing_after_the_adoption_fails(self) -> None:
        # Mutation (4): the ORDERING. Publishing d_dev after the adopt call leaves the
        # adoption looking at a null handle — present, and useless.
        text = mutate(
            PACKED_PUBLISH + "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n" + PACKED_PUBLISH,
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("publishes `d_dev` AFTER" in p for p in problems), problems)

    # --- (f) READ-FIRST ---------------------------------------------------------
    def test_adopting_before_the_upload_copy_fails(self) -> None:
        # Mutation (5): the OTHER ordering. `AdoptDeviceBytesAsHost` re-points
        # `w.packed.bytes` at the device allocation and releases the consumed source
        # pages; running it before the `Copy` means the upload reads pages that were
        # already handed back. Publication still precedes the adopt, so clause (e) is
        # satisfied and only this one may fire.
        text = mutate(
            "    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);\n"
            "    Backend* bk = &d.b;\n"
            + PACKED_PUBLISH
            + "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    Backend* bk = &d.b;\n"
            + PACKED_PUBLISH
            + "    AdoptDeviceBytesAsHost(d.b, w.packed);\n"
            "    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);\n",
        )
        problems = body_violations(body_of(text))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("BEFORE the upload copy", problems[0])
        self.assertIn("`packed`", problems[0])

    # --- the whole revert -------------------------------------------------------
    def test_dropping_all_six_statements_fails(self) -> None:
        # The exact revert a fresh reviewer applied to BOTH copies, which left every
        # existing suite green. Six distinct violations here, three per buffer.
        text = GOOD
        for line in (
            "    vllm::load_stats::AddDeviceUpload(pb);\n",
            "    vllm::load_stats::AddDeviceUpload(sb);\n",
            PACKED_PUBLISH,
            SCALE_PUBLISH,
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    AdoptDeviceBytesAsHost(d.b, w.scale);\n",
        ):
            text = mutate(line, "", text)
        self.assertEqual(len(body_violations(body_of(text))), 6)

    def test_a_missing_definition_is_a_violation_not_a_pass(self) -> None:
        # Renaming or deleting a copy must go RED, never silently vacuous.
        problems = file_violations("int main() { return 0; }", "some/file.cpp")
        self.assertEqual(len(problems), 1)
        self.assertIn("no `ResidentNvfp4", problems[0])


class DisguisedDeletionTests(unittest.TestCase):
    """A statement the COMPILER never sees is a deletion, however it reads.

    Each case leaves the statement's TEXT in place and removes only its effect. The
    round-2 checker matched raw source and returned GREEN on every one of them, which
    is the finding these pin: a commented-out post-upload step is exactly the silent
    runtime divergence the gate says it catches.
    """

    def test_a_line_commented_adoption_is_not_a_pass(self) -> None:
        text = mutate(
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    // AdoptDeviceBytesAsHost(d.b, w.packed);\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("`packed` skips" in p for p in problems), problems)
        self.assertFalse(any("`scale` skips" in p for p in problems), problems)

    def test_a_block_commented_adoption_is_not_a_pass(self) -> None:
        text = mutate(
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    /* AdoptDeviceBytesAsHost(d.b, w.packed); */\n",
        )
        self.assertTrue(any("`packed` skips" in p for p in body_violations(body_of(text))))

    def test_an_if_0_adoption_is_not_a_pass(self) -> None:
        text = mutate(
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "#if 0\n    AdoptDeviceBytesAsHost(d.b, w.packed);\n#endif\n",
        )
        self.assertTrue(any("`packed` skips" in p for p in body_violations(body_of(text))))

    def test_an_if_false_adoption_is_not_a_pass(self) -> None:
        text = mutate(
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "    if (false) { AdoptDeviceBytesAsHost(d.b, w.packed); }\n",
        )
        self.assertTrue(any("`packed` skips" in p for p in body_violations(body_of(text))))

    def test_a_commented_upload_counter_is_not_a_pass(self) -> None:
        text = mutate(
            "    vllm::load_stats::AddDeviceUpload(sb);\n",
            "    // vllm::load_stats::AddDeviceUpload(sb);\n",
        )
        problems = body_violations(body_of(text))
        self.assertTrue(any("`scale`'s host->device upload is NOT counted" in p for p in problems), problems)
        self.assertFalse(any("`packed`'s host->device upload is NOT counted" in p for p in problems), problems)

    def test_a_commented_publication_is_not_a_pass(self) -> None:
        text = mutate(PACKED_PUBLISH, "    // " + PACKED_PUBLISH.strip() + "\n")
        self.assertTrue(any("`packed` does not PUBLISH" in p for p in body_violations(body_of(text))))

    def test_an_if_0_publication_is_not_a_pass(self) -> None:
        text = mutate(SCALE_PUBLISH, "#if 0\n" + SCALE_PUBLISH + "#endif\n")
        self.assertTrue(any("`scale` does not PUBLISH" in p for p in body_violations(body_of(text))))

    def test_a_commented_upload_copy_is_not_a_pass(self) -> None:
        text = mutate(
            "    d.b.Copy(d.q, p, w.scale.bytes.data(), sb);\n",
            "    // d.b.Copy(d.q, p, w.scale.bytes.data(), sb);\n",
        )
        self.assertTrue(any("does not upload `w.scale.bytes.data()`" in p for p in body_violations(body_of(text))))

    # --- the other direction ----------------------------------------------------
    def test_ordinary_comments_beside_live_statements_still_pass(self) -> None:
        # The real function carries a nine-line ENG-LOAD-DIRECT-UPLOAD comment above
        # its first statement, and it NAMES every statement below it. Blanking
        # comments must not turn that into a false red.
        text = mutate(
            "    vllm::load_stats::AddDeviceUpload(pb);\n",
            "    // ENG-LOAD-DIRECT-UPLOAD (issue #150): account the move, publish\n"
            "    // the allocation on the OwnedTensor (w.packed.d_dev = shared_ptr),\n"
            "    /* then AdoptDeviceBytesAsHost(d.b, w.packed) releases the pages. */\n"
            "    vllm::load_stats::AddDeviceUpload(pb);\n",
        )
        self.assertEqual(body_violations(body_of(text)), [])

    def test_an_ifdef_build_configuration_is_left_alone(self) -> None:
        # THE DECLARED LIMITATION, pinned in the direction it was declared. `#if 0` is
        # a disguised deletion; `#ifdef VT_CUTLASS_NVFP4` is a real build
        # configuration, and deciding it needs the build's macro state. This gate does
        # not model the preprocessor and must not pretend to: the case stays GREEN.
        text = mutate(
            "    AdoptDeviceBytesAsHost(d.b, w.packed);\n",
            "#ifdef VT_CUTLASS_NVFP4\n    AdoptDeviceBytesAsHost(d.b, w.packed);\n#endif\n",
        )
        self.assertEqual(body_violations(body_of(text)), [])

    def test_normalization_does_not_move_the_reported_line(self) -> None:
        # `file_violations` prefixes every problem with `label:line`, so the
        # normalization has to be position-preserving. A four-line block comment above
        # the definition must push the reported line by exactly four.
        text = "/* one\n   two\n   three\n   four */\n" + GOOD
        bodies = checked_bodies(text)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0][0], 6)  # GOOD's definition is on its own line 2

    def test_a_whole_definition_inside_if_0_is_not_COUNTED_as_checked(self) -> None:
        # `main` reports "N ResidentNvfp4 definition(s) ... " from the same
        # `checked_bodies`, so a copy commented out wholesale can never be counted as
        # checked while being reported as absent — the OK banner would be a lie.
        self.assertEqual(checked_bodies("#if 0\n" + GOOD + "#endif\n"), [])
        self.assertEqual(len(checked_bodies(GOOD)), 1)
        problems = file_violations("#if 0\n" + GOOD + "#endif\n", "some/file.cpp")
        self.assertEqual(len(problems), 1)
        self.assertIn("no `ResidentNvfp4", problems[0])


class ReleaseTests(unittest.TestCase):
    """Clause (g): the one release, and the ordering that makes it a release.

    Pre-fix, the repack builders reset ONLY the type-specific handle, so the
    surviving `d_dev` alias kept a full packed+scale device copy for the process
    lifetime. These cases pin the clause that sees that shape and each half of its
    repair, on a miniature and on the REAL header both upload copies release
    through."""

    def test_the_real_release_shape_passes(self) -> None:
        self.assertEqual(mod.release_violations(GOOD_RELEASE, "mini"), [])

    def test_dropping_the_packed_reset_fails(self) -> None:
        text = mutate("  packed.d_dev.reset();\n", "", GOOD_RELEASE)
        problems = mod.release_violations(text, "mini")
        self.assertTrue(any("`packed.d_dev` is never reset" in p for p in problems), problems)
        self.assertFalse(any("`scale.d_dev` is never reset" in p for p in problems), problems)

    def test_dropping_the_scale_reset_fails(self) -> None:
        text = mutate("  scale.d_dev.reset();\n", "", GOOD_RELEASE)
        problems = mod.release_violations(text, "mini")
        self.assertTrue(any("`scale.d_dev` is never reset" in p for p in problems), problems)

    def test_dropping_the_packed_twin_drop_fails(self) -> None:
        text = mutate(
            "  if (packed.HostViewIsDeviceTwin()) packed.ReleaseHost();\n", "", GOOD_RELEASE
        )
        problems = mod.release_violations(text, "mini")
        self.assertTrue(any("`packed`'s adopted host twin is never dropped" in p for p in problems), problems)

    def test_dropping_the_scale_twin_drop_fails(self) -> None:
        text = mutate(
            "  if (scale.HostViewIsDeviceTwin()) scale.ReleaseHost();\n", "", GOOD_RELEASE
        )
        problems = mod.release_violations(text, "mini")
        self.assertTrue(any("`scale`'s adopted host twin is never dropped" in p for p in problems), problems)

    def test_dropping_the_twin_after_the_reset_fails(self) -> None:
        # The ordering the release depends on: the adopted view holds the block, so
        # a `d_dev` reset first frees nothing and the drop below touches a view
        # whose owner is gone.
        text = mutate(
            "  if (scale.HostViewIsDeviceTwin()) scale.ReleaseHost();\n", "", GOOD_RELEASE
        )
        text = mutate(
            "  scale.d_dev.reset();\n",
            "  scale.d_dev.reset();\n  if (scale.HostViewIsDeviceTwin()) scale.ReleaseHost();\n",
            text,
        )
        problems = mod.release_violations(text, "mini")
        self.assertTrue(any("dropped AFTER its `d_dev` reset" in p for p in problems), problems)

    def test_a_missing_release_definition_is_a_violation_not_a_pass(self) -> None:
        problems = mod.release_violations("int main() { return 0; }", "some/file.h")
        self.assertEqual(len(problems), 1)
        self.assertIn("no `void ReleaseResident", problems[0])

    def test_the_LIVE_release_resets_both_dev_slots(self) -> None:
        text = (ROOT / mod.RELEASE_SOURCE).read_text(encoding="utf-8", errors="ignore")
        self.assertIsNotNone(mod.release_body(text), "ReleaseResident() is unreachable")
        self.assertEqual(mod.release_violations(text, str(mod.RELEASE_SOURCE)), [])

    def test_LIVE_scale_reset_dropped_goes_red(self) -> None:
        # THE LEAK, against the real header: the pre-fix builders dropped only the
        # type-specific handle, so this is the shape a regression takes.
        rel = mod.RELEASE_SOURCE
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        anchor = "    scale.d_dev.reset();\n"
        self.assertEqual(text.count(anchor), 1, anchor)
        problems = mod.release_violations(text.replace(anchor, "", 1), str(rel))
        self.assertTrue(any("`scale.d_dev` is never reset" in p for p in problems), problems)

    def test_LIVE_packed_reset_dropped_goes_red(self) -> None:
        rel = mod.RELEASE_SOURCE
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        anchor = "    packed.d_dev.reset();\n"
        self.assertEqual(text.count(anchor), 1, anchor)
        problems = mod.release_violations(text.replace(anchor, "", 1), str(rel))
        self.assertTrue(any("`packed.d_dev` is never reset" in p for p in problems), problems)

    def test_LIVE_scale_twin_dropped_goes_red(self) -> None:
        rel = mod.RELEASE_SOURCE
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        anchor = "    if (scale.HostViewIsDeviceTwin()) scale.ReleaseHost();\n"
        self.assertEqual(text.count(anchor), 1, anchor)
        problems = mod.release_violations(text.replace(anchor, "", 1), str(rel))
        self.assertTrue(any("`scale`'s adopted host twin is never dropped" in p for p in problems), problems)


class LiveTreeTests(unittest.TestCase):
    def test_the_checker_passes_on_the_current_tree(self) -> None:
        r = subprocess.run(
            [sys.executable, str(CHECKER)], capture_output=True, text=True, cwd=ROOT
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_both_copies_and_the_release_are_actually_reached(self) -> None:
        # The gate is worthless if SOURCES drifts off the real files: it would print
        # OK over nothing. Assert every named file really contains a checked body,
        # with both per-buffer upload blocks inside it, and that the release source
        # really defines the one release.
        for rel in mod.SOURCES:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
            bodies = checked_bodies(text)
            self.assertEqual(len(bodies), 1, str(rel))
            _, recv, body = bodies[0]
            for buffer in mod.BUFFERS:
                self.assertIsNotNone(buffer_block(body, recv, buffer), f"{rel}:{buffer}")
        release_text = (ROOT / mod.RELEASE_SOURCE).read_text(encoding="utf-8", errors="ignore")
        self.assertIsNotNone(mod.release_body(release_text), str(mod.RELEASE_SOURCE))

    def test_disguised_deletions_of_the_LIVE_duplicate_go_red(self) -> None:
        # The findings against the REAL qwen3_5.cpp text, not the miniature. The
        # round-2 checker returned exit 0 for every replacement below while the
        # compiler saw the statement gone.
        rel = Path("src/vllm/model_executor/models/qwen3_5.cpp")
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        adopt = "    AdoptDeviceBytesAsHost(d.b, w.packed);\n"
        publish = PACKED_PUBLISH
        for anchor, replacement in (
            (adopt, "    // AdoptDeviceBytesAsHost(d.b, w.packed);\n"),
            (adopt, "    /* AdoptDeviceBytesAsHost(d.b, w.packed); */\n"),
            (adopt, "#if 0\n" + adopt + "#endif\n"),
            (adopt, "    if (false) { AdoptDeviceBytesAsHost(d.b, w.packed); }\n"),
            (publish, "    // w.packed.d_dev = w.d_packed;\n"),
            (publish, "#if 0\n" + publish + "#endif\n"),
            (
                "    vllm::load_stats::AddDeviceUpload(pb);\n",
                "    // vllm::load_stats::AddDeviceUpload(pb);\n",
            ),
        ):
            self.assertEqual(text.count(anchor), 1, anchor)
            mutant = text.replace(anchor, replacement, 1)
            self.assertNotEqual(file_violations(mutant, str(rel)), [], replacement)

    def test_sinking_the_LIVE_upload_copy_below_its_adoption_goes_red(self) -> None:
        # NAME THE MOTION. This SINKS the `Copy` to below the adoption, leaving the
        # `d_dev` publication where it is, so clause (e) is still satisfied and only
        # (f) can fire — that is the shape the pre-(f) checker passed. HOISTING the
        # adoption above the `Copy` instead would lift it above the publication too
        # and trip (e), which the old checker already caught; it proves nothing
        # about (f).
        rel = Path("src/vllm/model_executor/models/qwen3_5.cpp")
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        anchor = (
            "    d.b.Copy(d.q, p, w.packed.bytes.data(), pb);\n"
            "    Backend* bk = &d.b;\n"
            + PACKED_PUBLISH
            + "    AdoptDeviceBytesAsHost(d.b, w.packed);\n"
        )
        self.assertEqual(text.count(anchor), 1)
        lines = anchor.splitlines(keepends=True)
        moved = "".join(lines[1:] + lines[:1])  # the Copy after the adopt
        problems = file_violations(text.replace(anchor, moved, 1), str(rel))
        self.assertTrue(any("BEFORE the upload copy" in p for p in problems), problems)

    def test_dropping_ONE_live_upload_counter_goes_red(self) -> None:
        # The finding, against the REAL qwen3_5.cpp text rather than the miniature:
        # the previous body-wide checker returned 0 here.
        rel = Path("src/vllm/model_executor/models/qwen3_5.cpp")
        text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        for anchor in (
            "    vllm::load_stats::AddDeviceUpload(pb);\n",
            "    vllm::load_stats::AddDeviceUpload(sb);\n",
        ):
            self.assertEqual(text.count(anchor), 1, anchor)
            self.assertNotEqual(
                file_violations(text.replace(anchor, "", 1), str(rel)), [], anchor
            )


if __name__ == "__main__":
    unittest.main()
