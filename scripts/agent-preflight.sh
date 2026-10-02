#!/usr/bin/env bash
# One command to run at session start and again before committing.
#
# The record gates are cheap, deterministic and CPU-only, but they were spread
# across a dozen invocations that an agent had to remember individually. A
# forgotten gate becomes a red build after the push, which is the expensive
# moment to find out. This runs all of them and prints the resume digest.
#
#   scripts/agent-preflight.sh              # gates + role + print .agents/NOW.md
#   scripts/agent-preflight.sh --staged     # also check the staged change
#   scripts/agent-preflight.sh --quiet      # gates only, no digest
#   scripts/agent-preflight.sh --no-require-role  # tolerate an undeclared role
#   scripts/agent-preflight.sh --role-only  # ONLY the role gate; NOT a preflight
#   scripts/agent-preflight.sh --fail-on-skip  # exit 1 when any gate was SKIPPED
#
# A session declares a role, and an UNDECLARED one is a failing gate by default:
# the obligation used to live in prose and in an opt-in flag, and neither fired.
# read-only passes a plain run and fails --staged, because staging is writing.
#
# --role-only exists so the mutation suite can EXECUTE that gate instead of
# grepping it: text assertions catch a REWRITE of the default and miss an
# OVERRIDE on a later line. A nested FULL run is impossible -- this script runs
# the very suite that would call it, so it would recurse without bound -- hence
# a mode that runs the role block and stops. It is not an opt-out: it checks the
# ROLE strictly, which --no-require-role does not, and it never prints the "All
# gates green." banner, because it skips every record gate and mutation suite
# --no-require-role runs and so has not earned it. Neither mode is a superset of
# the other; --role-only is narrower and stricter, and says so on stdout.
#
# Every gate reports one of THREE states. `ok` ran and passed, `FAIL` ran and
# failed, `SKIP` did not run and says why. "All gates green." needs an empty
# FAIL list AND an empty SKIP list, because a banner over a block that never
# executed is a false report rather than a weaker gate (#998).
#
# The exit status carries only TWO of those states, so a SKIP still exits 0 and
# a reader who wants the third has to read the report. That is right for a human
# on a branch behind main and wrong for a program: `scripts/agent-ready.py` read
# preflight by exit code alone and called a run with two unexecuted trailer gates
# "green". --fail-on-skip is the opt-in for a machine consumer -- it exits 1 when
# anything was skipped, and it does not change what any gate demands or what a
# plain run reports. Ask for it when a skip must not read as success.
#
# It writes NOTHING into the source tree. The compile gate below configures a
# CMake build directory under the system temporary directory and removes it
# again, because reading a build's real flags is the only way to compile with
# them; nothing else here writes at all. It is still always safe to run.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGED=0
QUIET=0
ROLE_ONLY=0
# OFF by default, and that default is the decision argued in the spec's §3.4: a
# branch behind origin/main is ordinary work, and a gate that fires on ordinary
# work is the defect. The flag exists because the exit status cannot carry the
# third state, so every consumer that reads only the return code reads a SKIP as
# success. There was exactly one such consumer (#998).
FAIL_ON_SKIP=0

# Checkers the range and trailer blocks below invoke WITH arguments. The
# discovered sweep skips these names so a gate that already ran properly is not
# re-run bare and reported as a usage skip.
#
# check-tree-compiles.py is here for a HARDER reason than tidiness, and it is
# here because the omission cost a run. The sweep runs every `scripts/check-*.py`
# bare, so a name missing from this list executes TWICE. For the four names above
# the second run is a cheap usage error. For this one it is a second full
# `-fsyntax-only` pass at -j8: measured 65 s, 39 units and 531 MB RSS on a branch
# behind main, started while the first pass had just finished, on a box already
# at load 157 from other agents' builds. The run it was added to died there
# without writing an exit line. Parallel builds have OOM-killed this box before,
# and a gate that spawns a second copy of its own compiler is that hazard wearing
# a checker's name.
NAMED_CHECKERS="check-agent-record.py check-commit-style.py check-commit-trailers.py check-now-current.py check-tree-compiles.py"
# ON by default: an undeclared session is a FAILING gate. The mutation suite
# anchors on THIS line (`^REQUIRE_ROLE=1$`) and refuses any line-anchored
# assignment of zero, quoted or not, so a silent revert of the default goes red
# however it is spelled. The opt-out in the arg loop is indented and therefore
# not line-anchored, which is what keeps the two distinguishable.
REQUIRE_ROLE=1
for arg in "$@"; do
  case "$arg" in
    --staged) STAGED=1 ;;
    --quiet) QUIET=1 ;;
    --require-role) REQUIRE_ROLE=1 ;;
    --no-require-role) REQUIRE_ROLE=0 ;;
    --role-only) ROLE_ONLY=1 ;;
    --fail-on-skip) FAIL_ON_SKIP=1 ;;
    -h|--help) sed -n '2,43p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

CHECKERS=(
  # FIRST deliberately. Every other gate reads a file for its own reason and
  # measures its own budget, so a table row that is half one branch and half
  # another satisfies all of them (#1417). This one asks whether a merge tool
  # wrote into a tracked file at all, and a reader who sees it fail knows to
  # stop reading the verdicts below.
  check-conflict-markers
  check-prompt-contract
  check-agent-record
  check-release-binary-contract
  check-release-workflow
  check-windows-release-state
  check-container-matrix
  check-container-workflow
  check-build-runtime-deps
  # A lease script that keeps its ccache cache on the CIFS share reads as
  # compliance and caches nothing: symlink(2) is EOPNOTSUPP there, so every
  # lock fails and `ccache -s` shows zero of everything (#2473).
  check-lease-ccache
  claim-view
  check-readme-structure
  check-quickstart-recipes
  check-benchmark-index
  check-model-checklist
  check-supported-models
  check-env-doc
  check-fusion-consistency
  check-attention-rung-consistency
  check-fp4-resident-consistency
  check-cuda-op-arch-gate
  check-rocm-dp4a-intrinsic
  check-runner-routing-consistency
  check-surface-coverage
  check-test-registration
  check-snapshot-pins
  check-oracle-pins
  check-comparator-pins
  check-oracle-denominator-flags
  check-now-current
  check-gate-commands
  check-symbol-anchors
)

SUITES=(
  test_check_prompt_contract
  test_agent_gates
  test_agent_record
  test_check_release_binary_contract
  test_release_manifest
  test_release_archive
  test_release_pipeline
  test_check_windows_release_state
  test_release_postpublish_audit
  test_check_container_matrix
  test_check_container_workflow
  test_check_build_runtime_deps
  test_check_lease_ccache
  test_validate_container_image
  test_release_index
  test_release_metadata
  test_release_accelerator_metadata
  test_release_macos_metadata
  test_release_windows_metadata
  test_cpu_release_gates
  test_agent_role
  test_agent_onboard
  test_agent_start
  test_gate_bringup
  test_env_agnostic_tooling
  test_claim_view
  test_upstream_inventory
  test_check_readme_structure
  test_check_quickstart_recipes
  test_check_benchmark_index
  test_check_model_checklist
  test_check_supported_models
  test_check_env_doc
  test_check_deepseek_v4_vision_manifests
  test_checker_text
  test_check_fusion_consistency
  test_check_attention_rung_consistency
  test_check_fp4_resident_consistency
  test_check_cuda_op_arch_gate
  test_check_rocm_dp4a_intrinsic
  test_rocm_rank_kernels
  test_check_runner_routing_consistency
  test_check_surface_coverage
  test_check_test_registration
  test_check_snapshot_pins
  test_check_oracle_pins
  test_check_comparator_pins
  test_dsv4v_w6_compare
  test_cpu_x86_llamacpp_floor
  test_audit_live_rows
  test_check_gate_commands
  test_gpu_lock_one_truth
  test_main_baseline
  test_agent_preflight_skip_report
  test_agent_pr_body
  test_agent_issue_index
  test_check_symbol_anchors
  test_check_oracle_denominator_flags
  test_rocm_quant_gather_compare
  test_rocm_strix_ourarm_staged
  test_rocm_cooperative_toggles
  # The TOKENGATE job's build stage (#2895). `--no-build-isolation` makes pip
  # skip the pin's [build-system] requires, so the harness has to install them
  # itself, and the hand-kept list it used had drifted three names behind the
  # pin. That cost a dgx lease and a queue position for a 3-second traceback.
  # Bash and the standard library over one committed script: no GPU, no lease,
  # no toolchain, no network, and nothing in it can skip.
  test_tokengate_buildreq
  test_check_conflict_markers
  test_check_tree_compiles
  test_prepush_checker_names
  test_ab_arms_differ
  # The variadic-load harness's own controls (#2970). It measures two
  # engines on a leased GPU, but every property it CLAIMS -- linear
  # percentiles, a closed loop that really overlaps, a warmup the
  # measured window cannot see, a token count that comes from `usage`
  # and never from a chunk count -- is a property of the client and the
  # report. All of them are checkable against a loopback mock with no
  # lease, no toolchain and no weights, which is where they belong.
  test_variadic_harness
  test_ltx25_pixel_ab_harness
  test_ltx2_dit_attn_knob_arms
  test_ltx25_ab_memwatch
  test_ltx2_oracle_goldens
  # The latent-dump driver's own controls (#2514). Nothing executed that file,
  # so an incomplete rename (`raw_bf16` for `raw_native`) killed every control
  # in it with an uncaught KeyError, and its frame check passed on ZERO decoded
  # frames. Both were catchable with no GPU, no torch and no weights; neither
  # was caught, and both reached a queued dgx lease.
  test_ltx2_latent_dump
  test_tower_skip_rss_report
  # The other half of that harness: `run_arm`, its readiness poll and its
  # teardown, against a fake server on a scratch port (#1844). The reporter
  # suite beside it reads finished files and cannot see how they were made.
  test_tower_skip_rss_arm
  test_ci_walk_base
  test_rc_stage_checkpoint
  test_sglang_lease_identity
  test_rocprof_attach_preflight
  # #2877. Registered in TWO places -- here and in `.github/workflows/ci.yml` --
  # and pinned in `REQUIRED_SUITE_REGISTRATIONS`, so neither registration is
  # deletable at rc=0. It landed on NO lane at all: absent from this array and
  # from CI, while the tool it gates had never been run on a real fingerprint.
  # Standard library only, ~1.2 s, no artifact and no GPU.
  test_q4exp_layerfp_diff
)

failed=()
# A gate that did NOT run is a THIRD state (#998). It is not `ok`, because
# nothing was verified, and it is not `FAIL`, because nothing was found wrong.
# Reporting it as neither is what let this script print "All gates green." over
# a block that never executed: three times in one session, twice because
# origin/main moved MID-RUN. See BASE_SHA below and the two range blocks.
skipped=()

run() {
  local label="$1"
  shift
  if output=$("$@" 2>&1); then
    printf '  \033[32mok\033[0m   %s\n' "$label"
  else
    printf '  \033[31mFAIL\033[0m %s\n' "$label"
    printf '%s\n' "$output" | sed 's/^/         /' | head -12
    failed+=("$label")
  fi
}

# The same shape as run(), for a gate this run cannot execute. The REASON prints
# with it and is not optional: a reader who learns only that something did not
# happen cannot tell a stale checkout from a broken one.
skip() {
  local label="$1"
  shift
  printf '  \033[33mSKIP\033[0m %s\n' "$label"
  printf '%s\n' "$*" | sed 's/^/         /'
  skipped+=("$label")
}

# origin/main resolved ONCE, before any gate runs. Every range block below
# compares against this SHA and never against the ref.
#
# The ref is remote-tracking and therefore SHARED by every linked worktree of
# this checkout. A fetch in any other worktree moves it while this run is in
# flight, so `--is-ancestor origin/main HEAD` answered true at the top of the
# run and false by the time the trailer block asked. The trailer gates then
# vanished from a report that still said green, and the `ok` count fell from 76
# to 74 with no other change in the output. Pinning removes that race. Printing
# the SHA tells the reader which revision the verdict is about.
#
# Resolving to a commit (`^{commit}`) rather than to whatever the ref names
# keeps a tag or an annotated object from reaching a gate as a base.
BASE_REF="origin/main"
BASE_SHA="$(git rev-parse --verify -q "${BASE_REF}^{commit}" 2>/dev/null || true)"
BASE_UNRESOLVED="${BASE_REF} does not resolve here, so this run cannot tell which
commits are new. Fetch the remote, or name the remote that carries the base
branch, then rerun. Unknown is not absence."

# The other two questions about the pinned base, asked ONCE and beside the pin,
# with the git exit status kept rather than collapsed.
#
# `--is-ancestor` answers 1 for "no" and 128 for "that question cannot be asked
# here" -- an unborn HEAD (`git checkout --orphan`) is one way to reach it. Both
# used to take the same arm, so the report named a cause ("is not an ancestor of
# HEAD") that was not the cause.
#
# `rev-list --count` used to be spelled `|| echo 0`, which filed a FAILED count
# under the deliberate empty-range exemption of §3.6. An empty range withholds
# nothing and keeps the banner. A count that could not be taken withholds
# everything and must not. Mapping the second onto the first is the same
# unknown-as-success conflation this row exists to remove.
#
# The count keeps stderr OUT of its value, and the value is then validated. The
# repair for `|| echo 0` was first written as `2>&1`, which reintroduced the very
# defect through a narrower door: git can write to stderr AND exit 0, so the
# status catches nothing while the value carries the error text ahead of the
# number. `[ "$RANGE_COUNT" -gt 0 ]` does not evaluate false there, it ERRORS
# with status 2, which reads as false, and both range blocks fell through to
# their empty-range arm while "All gates green." printed over five dropped
# gates. A `.git/objects/info/alternates` naming a path that does not exist is
# one way to reach it: `error: unable to normalize alternate object path: ...`
# on stderr, the count on stdout, exit 0.
#
# Discarding stderr is what makes the VALUE right, and it is also what keeps a
# git that merely warns from costing five gates a SKIP it does not deserve.
# Validating the value is what makes an arm exist at all for a count that is not
# a count: without it, any non-numeric value reaches `-gt` and its status-2 error
# is indistinguishable from "zero commits". The two halves cover different
# failures, so both are here.
#
# Discarding stderr also had a COST, and the cost is paid separately rather than
# argued away. `2>/dev/null` threw the message away along with its influence on
# the value, so an unborn HEAD reported "exited 128 and printed [] on stdout"
# while git had already named the cause in one line. That report is honest and
# not actionable, and this row owes both. The rule is message-NOT-a-value, not
# no-message: the text goes into its own variable, which nothing compares and no
# arm reads, exactly as ANCESTRY_ERROR already did one line above.
ANCESTRY_ERROR=""
ANCESTRY_STATUS=0
RANGE_COUNT=""
RANGE_ERROR=""
RANGE_STATUS=0
if [ -n "$BASE_SHA" ]; then
  # stderr stays merged HERE deliberately. ANCESTRY_ERROR is a MESSAGE and never
  # a value: nothing compares it, and only ANCESTRY_STATUS selects an arm below.
  # Checked rather than assumed -- under the broken-alternates case above this
  # call also writes to stderr and exits 0, and every ancestry arm is unaffected.
  ANCESTRY_ERROR="$(git merge-base --is-ancestor "$BASE_SHA" HEAD 2>&1)"
  ANCESTRY_STATUS=$?
  # The count is asked TWICE, for two different things, and the order matters
  # twice over. `2>&1 >/dev/null` duplicates stderr onto the capture BEFORE
  # stdout leaves for /dev/null, so this call yields the message alone and never
  # the count. The value call runs second so RANGE_STATUS still describes the
  # command the value came from. Nothing selects an arm from RANGE_ERROR.
  RANGE_ERROR="$(git rev-list --count "${BASE_SHA}..HEAD" 2>&1 >/dev/null)"
  RANGE_COUNT="$(git rev-list --count "${BASE_SHA}..HEAD" 2>/dev/null)"
  RANGE_STATUS=$?
fi
# One predicate for "this run holds a usable commit count", so the two range
# blocks below cannot drift apart on what counts as unknown.
RANGE_NUMERIC=1
case "$RANGE_COUNT" in
  '' | *[!0-9]*) RANGE_NUMERIC=0 ;;
esac
ANCESTRY_UNKNOWN="git merge-base --is-ancestor ${BASE_SHA} HEAD exited
${ANCESTRY_STATUS}, so this run could not ask whether the base is behind HEAD:
${ANCESTRY_ERROR}
An unborn HEAD is one way to reach this. Unknown is not a verdict on ancestry."
# The closing sentence stays on ONE line deliberately. The suite asserts it
# verbatim, and a rewrap that splits it across a newline turns a correct report
# into a red gate rather than into a weaker one.
RANGE_UNKNOWN="git rev-list --count ${BASE_SHA}..HEAD exited ${RANGE_STATUS} and
printed [${RANGE_COUNT}] on stdout, which is not a commit count, so this run
could not count the commits under judgement. An unborn HEAD is one way to reach
this, and so is a git that writes an error to stderr and still exits 0. git
wrote this to stderr:
${RANGE_ERROR}
Unknown is not an empty range."

echo "Session role:"
if role_line=$(python3 scripts/agent-role.py show 2>&1); then
  printf '  \033[32mok\033[0m   %s\n' "$role_line"
else
  printf '  \033[33m--\033[0m   %s\n' "$(printf '%s' "$role_line" | head -1)"
  printf '       This session has not declared a role. Run scripts/agent-start.py\n'
  printf '       for the canonical role interview and exact next actions.\n'
  if [ "$REQUIRE_ROLE" -eq 1 ]; then
    failed+=("role-undeclared")
  fi
fi

# read-only PASSES a plain preflight -- that is the point of the third answer.
# It fails --staged, because staging is writing.
if [ "$STAGED" -eq 1 ] && printf '%s' "$role_line" | grep -q 'role=read-only'; then
  printf '  \033[31mFAIL\033[0m read-only sessions do not write. Claim operator or helper first.\n'
  failed+=("read-only-cannot-stage")
fi

if [ "$ROLE_ONLY" -eq 1 ]; then
  echo "ROLE CHECK ONLY -- this is NOT a full preflight; no record gate ran."
  if [ "${#failed[@]}" -ne 0 ]; then
    echo "${#failed[@]} gate(s) failed: ${failed[*]}"
    exit 1
  fi
  exit 0
fi

echo "Record gates:"
for checker in "${CHECKERS[@]}"; do
  case "$checker" in
    # Both default to a REPORT that exits 0 whatever the record says; the gate
    # is the flag. Wiring either without --check installs a gate that cannot
    # fail, which for check-gate-commands is the very defect it classifies.
    claim-view|check-gate-commands) run "$checker" python3 "scripts/$checker.py" --check ;;
    # check-agent-record also carries the record-anchor ratchet
    # (ENG-RECORD-ANCHOR-RATCHET, #632): a citation that names a line no longer
    # holding the symbol beside it fails HERE, on the plain call, and the error
    # names the bucket that moved. Deliberately NOT wired as --report: `run`
    # shows only the first 12 lines of a failure, and the report's offender list
    # would push the error message out of that window. The full list is one
    # command away (`scripts/check-agent-record.py --report`) and CI prints it
    # unconditionally.
    *) run "$checker" python3 "scripts/$checker.py" ;;
  esac
done

run "ready-for-helper" python3 scripts/ready-for-helper.py --check
run "upstream-inventory" python3 scripts/upstream-inventory.py --check
# Every ACTIVE row still has real Git evidence behind it. Wired only AFTER the
# record was repaired (P0), so it never had to be relaxed to pass: a red here
# means the record drifted, never that the gate is too strict.
#
# It is UNCONDITIONAL, and that is the deliberate call. The audit aborts when
# origin/main does not resolve (a clone whose remote is not named `origin`, a
# shallow/detached checkout), so preflight goes RED there rather than skipping.
# A skip would have to survive the "All gates green." banner below, and a green
# preflight that never verified the record is the one unacceptable outcome --
# the same reason audit-live-rows.py refuses to treat absence of information as
# absence of work. The repair is one command and the abort message names it, so
# the red is actionable. Preflight itself must NOT fetch: it is documented above
# as never writing anything, and a gate that mutates refs to make itself pass is
# exactly the shape this protocol forbids.
run "audit-live-rows" python3 scripts/audit-live-rows.py --check

echo "Mutation suites:"
for suite in "${SUITES[@]}"; do
  run "$suite" python3 "tests/scripts/$suite.py"
done
# QWEN3-CAPTURE-TOOLS: begin
if python3 -c 'import numpy' >/dev/null 2>&1; then
  run "test_qwen3_capture_tools" python3 tests/scripts/test_qwen3_capture_tools.py
  run "test_qwen3_capture_outputs" python3 tests/scripts/test_qwen3_capture_outputs.py
else
  for suite in test_qwen3_capture_tools test_qwen3_capture_outputs; do
    skip "$suite" "PENDING: numpy is not importable. CI installs python3-numpy for these capture suites."
  done
fi
# QWEN3-CAPTURE-TOOLS: end

# THE ONE SUITE HERE WITH A THIRD-PARTY DEPENDENCY (#1612). It exercises
# `scripts/ltx25-render-compare.py`, whose only import beyond the standard
# library is numpy -- the tool reads PPM and WAV by hand precisely so that a
# leased worker needs nothing else. It ran on NO lane at all until now: absent
# from this array, from the enumerated python block in `.github/workflows/ci.yml`
# and from `tests/CMakeLists.txt`, while section 8 of its spec registered it as a
# gate. A gate no lane runs is "nothing lands dead" applied to the instrument.
#
# A missing numpy is a SKIP and never an `ok`: nothing was verified. CI installs
# it, so the lane that must not be silent is not the one that can be.
if python3 -c 'import numpy' >/dev/null 2>&1; then
  run "test_ltx25_render_compare" python3 tests/scripts/test_ltx25_render_compare.py
else
  skip "test_ltx25_render_compare" \
    "numpy is not importable here, and the tool this suite exercises needs it." \
    "CI installs python3-numpy and runs the same suite."
fi
# THE ABSOLUTE-REFERENCE SUITE (#1854), registered in the SAME change that adds
# it, because the paragraph above is the record of what happens otherwise: that
# suite "ran on NO lane at all until now", while its spec registered it as a
# gate. This one exercises the half of the same tool that CHECKS rather than
# reports, so a lane that ran only the older suite would leave the new bound
# unexecuted while the tool it lives in looked covered.
#
# Same numpy condition and the same SKIP-never-ok discipline. One case inside it
# also needs ffmpeg, to decode the committed reference mp4; that case skips
# itself with its own reason rather than the whole suite, because the other
# eighteen need neither.
if python3 -c 'import numpy' >/dev/null 2>&1; then
  run "test_ltx25_absolute_reference" python3 tests/scripts/test_ltx25_absolute_reference.py
else
  skip "test_ltx25_absolute_reference" \
    "numpy is not importable here, and the tool this suite exercises needs it." \
    "CI installs python3-numpy and runs the same suite."
fi
# THE PROMPT-ADHERENCE SUITE (#2295, owning #1854's first sub-question),
# registered in the SAME change that adds it, for the reason the paragraph above
# records. It exercises the third half of this tool: not "are these two renders
# the same" and not "does this render have artefacts in it", but "does it depict
# what was asked for".
#
# Same numpy condition and the same SKIP-never-ok discipline. FIVE of its 47
# cases need the pinned CLIP checkpoint and skip themselves, individually, with
# their own reason when `VT_LTX25_ADHERENCE_MODEL` is unset -- the other 42
# need numpy and nothing else, no checkpoint, no network and no GPU, because
# every bound they check is arithmetic over a score matrix the case builds.
if python3 -c 'import numpy' >/dev/null 2>&1; then
  run "test_ltx25_prompt_adherence" python3 tests/scripts/test_ltx25_prompt_adherence.py
else
  skip "test_ltx25_prompt_adherence" \
    "numpy is not importable here, and the tool this suite exercises needs it." \
    "CI installs python3-numpy and runs the same suite."
fi
# THE DIAGNOSIS THAT DECIDED WHETHER THE SMOOTHNESS CAUSES THE ADHERENCE GAP
# (`LTX25-ADHERENCE-DETAIL-LOSS`, #2513). The row published a NEGATIVE result --
# the spectrum says our render is not smoother, the within-render correlation
# points the wrong way, and blurring the reference does not reproduce the gap --
# and a negative result is the shape a BROKEN instrument produces for free. So
# every case feeds a part of that instrument a signal whose answer is known in
# closed form, and several assert it moves in the direction the falsified
# hypothesis would have needed. Two of them were red on arrival and found real
# defects: an empty radial bin read as "below unity" and pushed the crossover
# upward, and a colour fixture measured luma's channel mixing rather than the
# gamma fit. numpy only -- no checkpoint, no frames, no network, no GPU.
if python3 -c 'import numpy' >/dev/null 2>&1; then
  run "test_ltx25_adherence_detail_loss" python3 tests/scripts/test_ltx25_adherence_detail_loss.py
else
  skip "test_ltx25_adherence_detail_loss" \
    "numpy is not importable here, and every case computes over arrays." \
    "CI installs python3-numpy and runs the same suite."
fi
# THE GLM-5.3-Flash GGUF CONVERTER (#2011). Same shape and the same one
# dependency: `scripts/convert-glm5-next-gguf.py` deliberately does not use
# gguf-py -- upstream has no `glm5_next` and, decisively, `gguf.quants.Q2_K`
# implements `dequantize_blocks` and NO `quantize_blocks` -- so numpy is all it
# needs and all this suite needs. The suite builds a SYNTHETIC tiny-shape
# checkpoint: it needs no weights, no GPU and no C++ build, which is the whole
# reason the converter can be gated at all while the real 305.78 GiB artifact
# stays out of reach.
#
# A missing numpy is a SKIP and never an `ok`, for the reason above.
if python3 -c 'import numpy' >/dev/null 2>&1; then
  run "test_convert_glm5_next_gguf" python3 tests/scripts/test_convert_glm5_next_gguf.py
else
  skip "test_convert_glm5_next_gguf" \
    "numpy is not importable here, and the converter this suite exercises needs it." \
    "CI installs python3-numpy and runs the same suite."
fi
# THE WINDOWS SOURCE CONTRACT, which ran on NO lane until #1829 (#646, #680).
# `main` failed to COMPILE under MSVC while every POSIX lane was green, because
# `[[noreturn]]` on a non-void return type is C4646 -> C2220 there and a silent
# accept everywhere else, and `windows-msvc-*` are pull-request-only jobs that
# give `main` no verdict at all. The suite is now registered here and in the
# `agent-record` job of `.github/workflows/ci.yml`, like the two above.
#
# `shipped_server_sources` derives its set from the CMake file API and forces
# `-G Ninja`, so a box without both tools cannot run this suite. A missing tool
# is a SKIP and never an `ok`: nothing was verified. CI installs `ninja-build`
# and runs the same suite, so the lane that must not be silent is not the one
# that can be.
if command -v cmake >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1; then
  run "test_check_windows_portability" python3 -m unittest \
    tests.scripts.test_check_windows_portability
else
  skip "test_check_windows_portability" \
    "cmake and ninja are both needed: the checker derives the shipped-server" \
    "source set from the CMake file API with the Ninja generator." \
    "CI installs ninja-build in agent-record and runs the same suite."
fi
run "trailer suites" python3 -m unittest \
  tests.scripts.test_check_commit_trailers
run "commit style suites" python3 -m unittest \
  tests.scripts.test_check_commit_style
# THE BENCHMARK-TOOL SUITES, which PREFLIGHT never ran until #1646 (#1648).
# `tests/tools/` holds the oracle pin, the clock-state assertions, the
# online-gate client and summary, and the serve-low request-set completeness.
#
# #1646 said no lane ran them at all. That was WRONG and #1648 corrects it:
# `tests/CMakeLists.txt` has registered them as the CTest target
# `test_serve_low_tools` since `e58858a91`, and CI's `build-test-cpu` runs
# `ctest --test-dir build` on every pull request. The claim came from grepping
# for `tests.tools`, the dotted module path, while CMake spells it
# `tests/tools` -- a null grep proving the terms wrong rather than the thing
# absent. The parity ledger's "all tools" citations were therefore citing a
# LIVE suite, not a dead one.
#
# The narrow gap was real and this line closes it: preflight ran none of them,
# so a local pre-edit check missed a red these suites would have caught, and
# only a full C++ configure-and-build surfaced it.
#
# DISCOVERED rather than enumerated. An enumeration is a shared list every new
# suite must edit, which is the record-lock shape AGENTS.md §Records forbids;
# discovery makes the file's existence the registration. Standard library only,
# no GPU and no wheel.
run "tools suites" python3 -m unittest discover -s tests/tools -t . -p "test_*.py"

# EVERY checker in scripts/, DISCOVERED the same way and for the same reason.
#
# This block used to name five: check-agent-record, check-commit-style,
# check-commit-trailers, check-issue-index-append-only and check-now-current.
# scripts/ holds many more than five. "All gates green." therefore described a
# small minority of the gates, and every checker the list omitted reached the
# working tree only where its own test suite happened to call it against the real
# ROOT -- a property of the test, not of the gate (#467).
#
# How many is deliberately not written here. A count of one file stored inside
# another goes stale on an edit whose author never reads this line, and it
# couples every unrelated change to a line it does not own, which AGENTS.md
# §Records forbids. This comment carried `There are 42.` and it was wrong twice
# over: `7dc2ef1ea` took the directory to 41, and later checkers have since put
# it back at 42, so the stale line happened to read true again for a reason
# nobody intended.
#
# `check-issue-index-append-only` above is history, not a file. `7dc2ef1ea`
# deleted it together with the append-only issue index it gated. The loop below
# discovers whatever scripts/check-*.py holds today, which is the point, and a
# named list is the same record-lock shape: every new checker would have to edit
# one shared line here.
#
# A checker that needs arguments this block cannot supply is a SKIP carrying the
# reason, never silence. The argparse USAGE TEXT is what distinguishes that case,
# not the exit code: rc 2 is argparse's usage error AND a legitimate verdict in
# at least one checker here, so keying on the code alone would read a real red as
# "needs arguments" and drop it from the report.
for checker in scripts/check-*.py; do
  name="$(basename "$checker")"
  case " $NAMED_CHECKERS " in *" $name "*) continue ;; esac
  if output=$(python3 "$checker" 2>&1); then
    printf '  \033[32mok\033[0m   %s\n' "$name"
  elif printf '%s' "$output" | grep -qE 'the following arguments are required|expected one argument|usage: '; then
    skip "$name" "needs arguments preflight does not supply: $(printf '%s' "$output" | head -1)"
  else
    printf '  \033[31mFAIL\033[0m %s\n' "$name"
    printf '%s\n' "$output" | sed 's/^/         /' | head -12
    failed+=("$name")
  fi
done

# The COMMITTED range, checked the way CI checks it. Deliberately OUTSIDE the
# --staged block: `--staged` inspects staged paths and is therefore VACUOUS after
# `git commit`, which is when preflight normally runs -- so the obligation went
# unchecked for a whole series and PR #80 landed eight commits that reddened
# documentation-checkpoint on main, where a diff-scoped range is never re-covered.
# Gating this on --staged would reproduce that hole exactly.
#
# An EMPTY range is not a skipped gate and does not print SKIP. A verdict over
# zero commits withholds nothing, and reporting it as a skip would fire on the
# ordinary session-start run of a freshly cut branch. An UNRESOLVABLE base is
# the other case entirely: there are commits to judge and this run cannot tell
# which, so it reports SKIP and forfeits the banner.
if [ -z "$BASE_SHA" ]; then
  echo "Committed range vs ${BASE_REF}:"
  skip "now-current range" "$BASE_UNRESOLVED"
  skip "doc-checkpoint range" "$BASE_UNRESOLVED"
elif [ "$RANGE_STATUS" -ne 0 ] || [ "$RANGE_NUMERIC" -eq 0 ]; then
  echo "Committed range vs ${BASE_REF} ${BASE_SHA}:"
  skip "now-current range" "$RANGE_UNKNOWN"
elif [ "$RANGE_COUNT" -gt 0 ]; then
  echo "Committed range vs ${BASE_REF} ${BASE_SHA}:"
  run "now-current range" python3 scripts/check-now-current.py \
    --base "$BASE_SHA" --head HEAD
else
  echo "Committed range vs ${BASE_REF} ${BASE_SHA}: empty, HEAD adds no commits."
fi

# DOES THE TREE COMPILE? Nothing above this line asks (#2401).
#
# Every gate above validates a record, a document, an anchor or a trailer
# against a tree, and every one of them passes on a tree that does not build.
# `main` was pushed twice on 2026-08-31 in exactly that state -- `5263ac31f`, a
# shell line continuation inside a `//` comment that `-Wcomment` under this
# tree's `-Werror` rejects, and `08fa2f5aa` (#2395), `MlaSharedSelection*` bound
# to a `vt::Tensor*` parameter -- with this script green both times. CI compiles
# four ways and would have caught both, up to two hours after the push, and both
# landed by a direct push. So the missing verdict is not "does anything build",
# it is "does anything build BEFORE the push".
#
# It is diff-scoped, and the scope is the only reason this is affordable: over
# the 60 commits ending at 9fa3be388, 30 touch no C++ at all and cost nothing
# here, 20 reach 1-4 translation units, and the worst reaches 783. A full build
# is ~12 minutes and 9.4 GiB, which is a gate that fires on ordinary work.
#
# THREE states, and the third is not a pass. The checker exits 2 when it could
# not take the measurement -- no cmake, a failed configure, a base that does not
# resolve -- and that maps onto SKIP, which forfeits the banner below and exits
# 1 under --fail-on-skip. It is NOT mapped onto FAIL, because a box without a
# compiler is not a defective change; and it is NOT mapped onto ok, because
# nothing was verified.
#
# BASE_SHA is passed when it resolved, and the REF when it did not, so the
# checker reports the unresolvable base in its own words rather than being
# handed an empty string that would read as an empty diff.
#
# The notice prints BEFORE the run, and it is not decoration. Every other gate
# here finishes in under a few seconds, so `run()` captures output and shows
# only a label. This one can take minutes on a wide header change, and a silent
# multi-minute gate in a list of one-second gates reads as a hang.
echo "Tree compiles:"
printf '       compiling what this change reaches. A records-only change returns\n'
printf '       at once; a wide header change can take minutes.\n'
compile_output="$(python3 scripts/check-tree-compiles.py --base "${BASE_SHA:-$BASE_REF}" 2>&1)"
# Read from the ASSIGNMENT, never after a pipe: `$?` after `cmd | head` is
# head's status, and a failing gate then reports rc=0.
compile_status=$?
case "$compile_status" in
  0)
    printf '  \033[32mok\033[0m   tree-compiles\n'
    printf '%s\n' "$compile_output" | tail -1 | sed 's/^/         /'
    ;;
  2)
    skip "tree-compiles" "$compile_output"
    ;;
  *)
    printf '  \033[31mFAIL\033[0m tree-compiles\n'
    printf '%s\n' "$compile_output" | sed 's/^/         /' | head -40
    failed+=("tree-compiles")
    ;;
esac

# Trailer enforcement reads only committed Git objects.
#
# The ancestry arm is the one #998 was filed for. It USED to be spelled as a
# silent `&&` in the condition, so a base that was not an ancestor deleted both
# gates from the report and the run still printed "All gates green.". The SKIP
# arm that replaced it is gone now (#2366): check-commit-style.py walks from
# the merge base, and `rev-list ${BASE_SHA}..HEAD` selects the branch's own
# commits under any ancestry, so a branch behind the base runs both gates
# instead of being told to merge first.
if [ -z "$BASE_SHA" ]; then
  echo "Commit trailers vs ${BASE_REF}:"
  skip "commit-trailers" "$BASE_UNRESOLVED"
  skip "commit-style" "$BASE_UNRESOLVED"
elif [ "$ANCESTRY_STATUS" -gt 1 ]; then
  echo "Commit trailers vs ${BASE_REF} ${BASE_SHA}:"
  skip "commit-trailers" "$ANCESTRY_UNKNOWN"
  skip "commit-style" "$ANCESTRY_UNKNOWN"
elif [ "$RANGE_STATUS" -ne 0 ] || [ "$RANGE_NUMERIC" -eq 0 ]; then
  echo "Commit trailers vs ${BASE_REF} ${BASE_SHA}:"
  skip "commit-trailers" "$RANGE_UNKNOWN"
  skip "commit-style" "$RANGE_UNKNOWN"
elif [ "$RANGE_COUNT" -gt 0 ]; then
  echo "Commit trailers vs ${BASE_REF} ${BASE_SHA}:"
  run "commit-trailers" python3 scripts/check-commit-trailers.py \
    --range "${BASE_SHA}..HEAD"
  run "commit-style" python3 scripts/check-commit-style.py \
    --range "${BASE_SHA}..HEAD"
else
  echo "Commit trailers vs ${BASE_REF} ${BASE_SHA}: empty, HEAD adds no commits."
fi

if [ "$STAGED" -eq 1 ]; then
  echo "Staged change:"
  run "now-current --staged" python3 scripts/check-now-current.py --staged
fi

# Printed BEFORE the failure summary and outside it, so a run that both failed
# and skipped reports both facts. They are different facts.
if [ "${#skipped[@]}" -ne 0 ]; then
  echo
  echo "${#skipped[@]} gate(s) SKIPPED: ${skipped[*]}"
  echo "NOT a green preflight: a skipped gate reported nothing about this tree."
  echo "Each reason is printed beside its SKIP above."
fi

if [ "${#failed[@]}" -ne 0 ]; then
  echo
  echo "${#failed[@]} gate(s) failed: ${failed[*]}"
  echo "Repair the record. Never weaken a checker to make a transition pass."
  exit 1
fi

# The opt-in, and the only thing that changes the exit status for a skip. The
# report above already says everything this line says. A caller reaches here
# because it reads the STATUS and not the report, which is what agent-ready.py
# did while printing the word "green" over two gates that never ran.
if [ "${#skipped[@]}" -ne 0 ] && [ "$FAIL_ON_SKIP" -eq 1 ]; then
  echo
  echo "--fail-on-skip: ${#skipped[@]} gate(s) did not run, so this run cannot"
  echo "answer the question that was asked of it. Exit 1 for the caller that"
  echo "reads only the status. Nothing here failed, and nothing here passed."
  exit 1
fi

# Reachable ONLY when both arrays are empty. A skip used to survive this line,
# which made the banner a claim the run had not earned (#998). The DEFAULT exit
# status stays 0 for a skip: a checkout whose base does not resolve is ordinary
# work, and
# exit 1 would merge "a gate did not run" into the signal that means "a gate ran
# and failed". A caller that cannot read the report asks for --fail-on-skip.
if [ "${#skipped[@]}" -eq 0 ]; then
  echo
  echo "All gates green."
fi

if [ "$QUIET" -eq 0 ]; then
  echo
  echo "================ .agents/NOW.md ================"
  cat .agents/NOW.md
fi
