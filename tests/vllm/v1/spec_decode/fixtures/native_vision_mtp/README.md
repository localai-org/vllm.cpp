# Executed image MTP1 transition

This compact integer fixture comes from one actual eager C1 image request in
the pinned custom EXL3 reference image. Its XPUModelRunnerV2 uses MTPSpeculator;
both target and draft declare multimodal input support. One encode supplies
four actual proposals during a four-token completion. Capture overhead excludes
these observations from performance claims.

The first target prompt contains 234 tokens and 192 image rows at [4,196).
The draft shifts IDs left by one, splices sampled token 760 at the last row and
merges the same image bytes at [3,195). Target feedback is byte-identical to the
target forward's final hidden output. The target receives [3,234] M-RoPE
positions; this executed V2 draft receives ordinary positions 0..233, even
though the target tail positions are 52..57. Preserve cache positions and
rotary coordinates as separate facts rather than applying the target delta to
the draft by assumption.

Source SHA-256 anchors and tensor identity digests are retained in the fixture.
Raw image embeddings/hidden states and observer captures remain local evidence.
The focused C++ input-preparation test checks the existing shift/splice contract
against this executed fixture. Draft merged-embedding consumption, runner
lookahead gathering, MTP3 rollback and graphs remain implementation work.
