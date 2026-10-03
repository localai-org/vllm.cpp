// PORT of SGLang's breakable-CUDA-graph unit suite,
// `test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py` @
// `f63458b5be` (305 lines, four classes). AGENTS.md requires the upstream tests
// in the same change that ports the behavior; `.agents/specs/eng-cudagraph-break.md`
// `## Tests to port` maps every upstream case to the local case that owes it,
// and those numbers (T1..T11) are repeated in each case name below.
//
// Row ENG-CUDAGRAPH-BREAK W1, issue #1192, parent #1163.
//
// THE HARNESS ADAPTATION is stated in `tests/vt/recording_capture_backend.h` and
// nowhere else: upstream skips without CUDA, we record the call sequence and
// simulate the graph, so a captured operation is filed at capture and RUN at
// replay. Upstream's arithmetic chains and its post-replay assertions therefore
// port literally — `x.fill_(5); graph.replay(); y == 6` is the same assertion
// here — as do the break counts, the segment counts, the in-place-versus-assign
// split and the non-copyable fallback.
//
// TWO deliberate deviations, named rather than claimed away:
//   * Upstream's `eager_on_graph` always returns a fresh tensor, so every
//     upstream break is the DESTINATION form. The in-place form has no upstream
//     case; it is exercised alongside T2 and by the bare marker.
//   * `test_gsm8k_accuracy` (`:288`) is EXCLUDED with the reason recorded in the
//     spec: it is a distributional accuracy floor (`mgsm_en >= 0.80`) on a
//     PREFILL capture path, and both halves are wrong for us. Our gate polarity
//     is bit-exactness against the model's own eager forward (spec `## Gates`
//     G1), which is strictly stronger.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __linux__
#include <unistd.h>  // readlink("/proc/self/exe"), getpid, for T4's child arm
#endif

#include "vt/breakable_graph.h"
#include "vt/recording_capture_backend.h"

namespace {

using vt::BreakableGraph;
using vt::BreakSlot;
using vt::GraphBreak;
using vt::GraphCaptureScope;
using vt_test::RecordingCaptureBackend;

// THE SUITE'S OWN PRECONDITION. Every capturing case below asserts the CAPTURING
// lane, so exporting `VLLM_CPP_CUDAGRAPH=0` around a run must not read as a code
// failure: an environment red and a code red are different verdicts and a suite
// that cannot tell them apart reports neither. The kill-switch case is the one
// that deliberately runs in the other lane, and it does so in a CHILD process.
void RequireCaptureLane() {
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this suite gates the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");
}

// A device-resident int32 vector on the recording backend, so a case can talk
// about a tensor's ADDRESS as well as its contents. The address is the whole
// point of the writeback contract (spec D9): on replay N the eager operation
// returns a FRESH allocation whose address is not the one the next segment
// baked at capture.
struct IntBuf {
  vt::Tensor t;
  int32_t* host = nullptr;
};

IntBuf MakeBuf(RecordingCaptureBackend& b, const std::vector<int32_t>& v) {
  IntBuf out;
  auto* p = static_cast<int32_t*>(b.Alloc(v.size() * sizeof(int32_t)));
  std::memcpy(p, v.data(), v.size() * sizeof(int32_t));
  out.host = p;
  out.t.data = p;
  out.t.dtype = vt::DType::kI32;
  out.t.rank = 1;
  out.t.shape[0] = static_cast<int64_t>(v.size());
  out.t.stride[0] = 1;
  return out;
}

// Always read THROUGH the tensor view, never through a cached pointer. The
// capture-time semantic of the destination form is that the slot BECOMES the
// first eager result (`:222-227`, upstream's
// `captured_output = _weak_ref_if_tensor(output)`), so a pointer cached before
// the break names a buffer nothing writes to afterwards.
std::vector<int32_t> ReadT(const vt::Tensor& t) {
  std::vector<int32_t> v(static_cast<size_t>(t.shape[0]));
  std::memcpy(v.data(), t.data, v.size() * sizeof(int32_t));
  return v;
}
std::vector<int32_t> Read(const IntBuf& b) { return ReadT(b.t); }

int32_t* Data(IntBuf& b) { return static_cast<int32_t*>(b.t.data); }

void Fill(IntBuf& b, int32_t v) {
  for (int64_t i = 0; i < b.t.shape[0]; ++i) Data(b)[i] = v;
}

// `dst = src + k`, the shape of every captured segment in the upstream cases.
void AddInto(IntBuf& dst, const vt::Tensor& src, int32_t k) {
  const std::vector<int32_t> s = ReadT(src);
  for (size_t i = 0; i < s.size(); ++i) Data(dst)[i] = s[i] + k;
}

// An eager break function: reads its input, scales it, and returns a FRESH
// allocation — which is exactly what makes the writeback contract load-bearing.
vt::Tensor FreshScaled(RecordingCaptureBackend& b, const vt::Tensor& src, int32_t mul,
                       int32_t add) {
  std::vector<int32_t> v = ReadT(src);
  for (int32_t& e : v) e = e * mul + add;
  return MakeBuf(b, v).t;
}

}  // namespace

// ---------------------------------------------------------------------------
// TestBreakableCUDAGraphBasic (`:30`) — the capture and replay mechanism.
// ---------------------------------------------------------------------------

// T1 <- test_no_break_capture_replay (`:49-63`). Zero breaks captures and
// replays exactly like a plain graph: `y = x + 1`, refill x to 5, replay, 6.
TEST_CASE("T1 breakable graph: no break yields ONE segment and replays like a plain graph") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;

  IntBuf x = MakeBuf(b, {0, 0, 0, 0});
  IntBuf y = MakeBuf(b, {0, 0, 0, 0});
  {
    GraphCaptureScope scope(b, q, g);
    REQUIRE(scope.active());
    b.Record([&] { AddInto(y, x.t, 1); });  // captured, NOT run
  }
  CHECK(g.segment_count() == 1);
  CHECK(g.break_count() == 0);
  CHECK(g.captured());
  // Exactly one Begin/EndCaptureGraph pair, in that order.
  CHECK(b.Trace() == "Begin EndCaptureGraph");
  CHECK(Read(y) == std::vector<int32_t>{0, 0, 0, 0});  // capture executed nothing

  Fill(x, 5);  // upstream's x.fill_(5.0)
  b.ClearLog();
  g.Replay(q);
  CHECK(b.Trace() == "ReplayGraph");
  CHECK(Read(y) == std::vector<int32_t>{6, 6, 6, 6});  // upstream's y == 6.0
  CHECK(g.replay_count() == 1);
}

// T2 <- test_single_break (`:65-87`). One break splits capture into two segments
// and the chain composes: x=10 -> +1=11 -> eager *2=22 -> +3=25.
TEST_CASE("T2 breakable graph: ONE break yields two segments and the chain composes") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;

  IntBuf x = MakeBuf(b, {0, 0, 0, 0});
  IntBuf mid = MakeBuf(b, {0, 0, 0, 0});
  IntBuf y = MakeBuf(b, {0, 0, 0, 0});
  BreakSlot<vt::Tensor> broken;

  {
    GraphCaptureScope scope(b, q, g);
    b.Record([&] { AddInto(mid, x.t, 1); });                      // segment 0: mid = x + 1
    GraphBreak([&] { return FreshScaled(b, mid.t, 2, 0); }, broken);  // the break: eager *2
    b.Record([&] { AddInto(y, *broken, 3); });                    // segment 1: y = broken + 3
  }

  CHECK(g.segment_count() == 2);
  CHECK(g.break_count() == 1);
  CHECK(b.Trace() == "Begin EndCaptureGraph Begin EndCaptureGraph");

  Fill(x, 10);  // upstream's x.fill_(10.0)
  g.Replay(q);
  // x=10 -> +1=11 -> eager *2=22 -> +3=25, the upstream chain and values.
  CHECK(ReadT(*broken) == std::vector<int32_t>{22, 22, 22, 22});
  CHECK(Read(y) == std::vector<int32_t>{25, 25, 25, 25});
}

// T3 <- test_multiple_breaks (`:89-115`). Two breaks, three segments, chained:
// x=5 -> +1=6 -> add_one=7 -> +1=8 -> double=16.
TEST_CASE("T3 breakable graph: N breaks yield N+1 segments and the chain composes (N=2)") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;

  IntBuf x = MakeBuf(b, {0, 0, 0, 0});
  IntBuf t1 = MakeBuf(b, {0, 0, 0, 0});
  IntBuf t3 = MakeBuf(b, {0, 0, 0, 0});
  IntBuf y = MakeBuf(b, {0, 0, 0, 0});
  BreakSlot<vt::Tensor> t2, t4;

  {
    GraphCaptureScope scope(b, q, g);
    b.Record([&] { AddInto(t1, x.t, 1); });                    // segment 0: t1 = x + 1
    GraphBreak([&] { return FreshScaled(b, t1.t, 1, 1); }, t2);  // break 1: add_one
    b.Record([&] { AddInto(t3, *t2, 1); });                    // segment 1: t3 = t2 + 1
    GraphBreak([&] { return FreshScaled(b, t3.t, 2, 0); }, t4);  // break 2: double
    b.Record([&] { AddInto(y, *t4, 0); });                     // segment 2: y = t4
  }
  CHECK(g.segment_count() == 3);
  CHECK(g.break_count() == 2);
  CHECK(b.Count("Begin") == 3);
  CHECK(b.Count("EndCaptureGraph") == 3);

  Fill(x, 5);  // upstream's x.fill_(5.0)
  g.Replay(q);
  // x=5 -> +1=6 -> add_one=7 -> +1=8 -> double=16, the upstream chain and values.
  CHECK(Read(y) == std::vector<int32_t>{16, 16, 16, 16});
}

// T4 <- test_eager_on_graph_disabled (`:117-129`). With the wrapper DISABLED the
// function is returned unchanged and runs normally. Our disable switch is item 1
// of the spec's `## Our baseline`: VLLM_CPP_CUDAGRAPH=0.
//
// IT RUNS IN A CHILD PROCESS, and that is the point. The switch is read once per
// process into a function-local static, so no case can toggle it. Asserting the
// OTHER inert arm instead — a backend that cannot capture — substitutes a
// different conjunct of `active_ = GraphCaptureEnabled() && SupportsGraphCapture()`,
// and deleting `GraphCaptureEnabled()` from that expression left the whole suite
// green. `docs/USAGE.md` sells this switch to users; a switch nothing executes is
// a claim, not a feature.
TEST_CASE("T4 breakable graph: the VLLM_CPP_CUDAGRAPH kill switch makes a capture-capable "
          "backend inert") {
  const char* sentinel_path = std::getenv("VLLM_CPP_BREAK_KILLSWITCH_SENTINEL");
  if (sentinel_path != nullptr) {
    // ---- CHILD ARM: this process was started with VLLM_CPP_CUDAGRAPH=0. ----
    REQUIRE_FALSE(vt::GraphCaptureEnabled());
    RecordingCaptureBackend b(/*supports_capture=*/true);  // the backend CAN capture
    vt::Queue q = b.CreateQueue();
    BreakableGraph g;
    int ran = 0;
    {
      GraphCaptureScope scope(b, q, g);
      REQUIRE_FALSE(scope.active());  // inert anyway: the switch is the only reason
      GraphBreak([&] { ++ran; });
    }
    REQUIRE(ran == 1);                 // the break function still runs
    REQUIRE(g.segment_count() == 0);
    REQUIRE(g.break_count() == 0);
    REQUIRE(b.Trace().empty());        // zero backend calls
    std::ofstream out(sentinel_path);
    out << "INERT-OK";
    return;
  }

  // ---- PARENT ARM. ----
#ifdef __linux__
  char exe[4096] = {0};
  const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
  REQUIRE_MESSAGE(n > 0, "cannot resolve this test binary; the child arm cannot be run");
  const std::filesystem::path sentinel =
      std::filesystem::temp_directory_path() /
      ("vllm_cpp_break_killswitch_" + std::to_string(::getpid()) + ".txt");
  std::filesystem::remove(sentinel);
  const std::string cmd = "VLLM_CPP_CUDAGRAPH=0 VLLM_CPP_BREAK_KILLSWITCH_SENTINEL='" +
                          sentinel.string() + "' '" + std::string(exe) +
                          "' -tc='*kill switch*' >/dev/null 2>&1";
  const int rc = std::system(cmd.c_str());
  CHECK(rc == 0);
  // The sentinel is what defeats "0 cases ran, SUCCESS!": a filter that matched
  // nothing exits 0 and writes no file.
  REQUIRE(std::filesystem::exists(sentinel));
  std::ifstream in(sentinel);
  std::string got;
  in >> got;
  CHECK(got == "INERT-OK");
  std::filesystem::remove(sentinel);
#else
  MESSAGE("kill-switch child arm needs /proc/self/exe; not run on this platform");
#endif
}

// T5 <- test_eager_on_graph_outside_capture (`:131-142`). Outside any capture
// the wrapper is a pass-through.
TEST_CASE("T5 breakable graph: OUTSIDE a scope GraphBreak runs fn and calls no backend") {
  RecordingCaptureBackend b;
  CHECK(GraphCaptureScope::Current() == nullptr);

  int ran = 0;
  GraphBreak([&] { ++ran; });
  CHECK(ran == 1);
  GraphBreak();  // the bare marker is also a pass-through
  CHECK(b.Trace().empty());

  // The destination form is a pass-through too, and it does NOT fabricate a
  // copy: the slot takes the value the break function produced, and its storage
  // is never pinned, so a non-capturing forward allocates nothing extra.
  BreakSlot<int> out;
  GraphBreak([] { return 7; }, out, vt::NoWriteback{});
  CHECK(*out == 7);
  CHECK_FALSE(out.pinned());
  CHECK(b.copies() == 0);
}

// T6 <- test_replay_updates_output (`:144-169`). TWO replays with different
// inputs give different outputs: 3.0, then after x.fill_(10), 33.0. It is the
// upstream anchor for G1's "more than one replay" requirement and the one case
// that can see a break function writing to a stale address — and it asserts the
// break's POSITION in the same log as the segments, not in a second sequence.
TEST_CASE("T6 breakable graph: two replays with different inputs give different outputs") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;

  IntBuf x = MakeBuf(b, {0, 0, 0, 0});
  IntBuf t = MakeBuf(b, {0, 0, 0, 0});
  IntBuf y = MakeBuf(b, {0, 0, 0, 0});
  BreakSlot<vt::Tensor> t2;
  {
    GraphCaptureScope scope(b, q, g);
    b.Record([&] { AddInto(t, x.t, 1); });                        // t = x + 1
    GraphBreak(
        [&] {
          b.Note("scale");
          return FreshScaled(b, t.t, 3, 0);
        },
        t2);                                                      // eager: t * 3
    b.Record([&] { AddInto(y, *t2, 0); });                        // y = t2
  }
  REQUIRE(g.segment_count() == 2);
  REQUIRE(g.break_count() == 1);

  // First replay: x=0 -> 0+1=1 -> 1*3=3.
  b.ClearLog();
  g.Replay(q);
  CHECK(b.Trace() == "ReplayGraph scale ReplayGraph");
  CHECK(Read(y) == std::vector<int32_t>{3, 3, 3, 3});

  // Second replay: x=10 -> 10+1=11 -> 11*3=33.
  Fill(x, 10);
  b.ClearLog();
  g.Replay(q);
  CHECK(b.Trace() == "ReplayGraph scale ReplayGraph");
  CHECK(Read(y) == std::vector<int32_t>{33, 33, 33, 33});
  CHECK(g.replay_count() == 2);
}

// ---------------------------------------------------------------------------
// TestCopyOutput (`:172`) — the output-writeback contract, spec `## Port map` §3
// and D9. Upstream tests `_copy_output` (`:172-201`) SEPARATELY from the capture
// machinery because it is a separate guarantee, and so does this.
// ---------------------------------------------------------------------------

// T7 <- test_tensor_copy (`:187-192`). A tensor destination is written IN PLACE
// and the SAME object is returned (`assertIs(result, dst)`).
TEST_CASE("T7 CopyOutput: a tensor destination is written IN PLACE, address preserved") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  IntBuf dst = MakeBuf(b, {0, 0, 0, 0});
  IntBuf src = MakeBuf(b, {5, 5, 5, 5});
  void* const dst_addr = dst.t.data;
  REQUIRE(dst.t.data != src.t.data);

  vt::CopyOutput(b, q, dst.t, src.t);

  CHECK(dst.t.data == dst_addr);  // assertIs(result, dst): identity preserved
  CHECK(Read(dst) == std::vector<int32_t>{5, 5, 5, 5});
  CHECK(b.copies() == 1);  // a real device copy, not a pointer assignment
}

// The two sides have to AGREE first. Sizing the copy from the destination alone
// makes a break function that returned the wrong shape or dtype an out-of-bounds
// READ of the source, where upstream's `dst.copy_(src)` raises. This is the
// refusal that keeps a mis-registered break point loud (spec D7).
TEST_CASE("T7c CopyOutput: a destination the break output does not match is REFUSED") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  IntBuf dst = MakeBuf(b, {0, 0, 0, 0});
  IntBuf small = MakeBuf(b, {1, 1});
  CHECK_THROWS_AS(vt::CopyOutput(b, q, dst.t, small.t), std::runtime_error);
  CHECK(b.copies() == 0);  // nothing moved

  IntBuf other = MakeBuf(b, {2, 2, 2, 2});
  other.t.dtype = vt::DType::kF32;  // same element count, different dtype
  CHECK_THROWS_AS(vt::CopyOutput(b, q, dst.t, other.t), std::runtime_error);
  CHECK(b.copies() == 0);
}

// T8 <- test_dict_copy (`:194-208`). A keyed set of destinations is copied
// field by field. This is the qwen3_5.cpp:9570-9577 shape: logits plus the
// auxiliary hidden taps.
TEST_CASE("T8 CopyOutput: a KEYED set of destinations is written field by field") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  IntBuf da = MakeBuf(b, {0, 0, 0, 0});
  IntBuf db = MakeBuf(b, {0, 0, 0, 0});
  IntBuf sa = MakeBuf(b, {1, 1, 1, 1});
  IntBuf sb = MakeBuf(b, {2, 2, 2, 2});
  void* const da_addr = da.t.data;
  void* const db_addr = db.t.data;

  std::map<std::string, vt::Tensor> dst{{"a", da.t}, {"b", db.t}};
  const std::map<std::string, vt::Tensor> src{{"a", sa.t}, {"b", sb.t}};
  vt::CopyOutput(b, q, dst, src);

  CHECK(dst.at("a").data == da_addr);
  CHECK(dst.at("b").data == db_addr);
  CHECK(Read(da) == std::vector<int32_t>{1, 1, 1, 1});
  CHECK(Read(db) == std::vector<int32_t>{2, 2, 2, 2});
  CHECK(b.copies() == 2);
}

// T9 <- test_object_copy (`:210-223`). A struct destination copies its tensor
// fields IN PLACE and ASSIGNS its non-tensor fields (`dst.label == "new"`).
//
// Harness adaptation: upstream reaches the fields through `__dict__`
// (`:182-192`). C++ has no reflection, so a struct participates by providing its
// own CopyOutput overload, found by argument-dependent lookup. The SPLIT the
// case pins — in place for the device field, assign for the rest — is preserved
// exactly.
namespace {
struct BreakStructOut {
  vt::Tensor tensor;
  std::string label;
};
void CopyOutput(vt::Backend& b, vt::Queue& q, BreakStructOut& dst,
                const BreakStructOut& src) {
  vt::CopyOutput(b, q, dst.tensor, src.tensor);  // in place
  dst.label = src.label;                         // assigned
}
}  // namespace

TEST_CASE("T9 CopyOutput: a struct copies tensor fields in place and ASSIGNS the rest") {
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  IntBuf dt = MakeBuf(b, {0, 0, 0, 0});
  IntBuf st = MakeBuf(b, {3, 3, 3, 3});
  void* const dt_addr = dt.t.data;

  BreakStructOut dst{dt.t, "old"};
  const BreakStructOut src{st.t, "new"};
  CopyOutput(b, q, dst, src);

  CHECK(dst.tensor.data == dt_addr);
  CHECK(Read(dt) == std::vector<int32_t>{3, 3, 3, 3});
  CHECK(dst.label == "new");
}

// T10 <- test_non_tensor_fallback (`:225-227`). With nothing copyable,
// `_copy_output` returns `src` — the documented fallback. Here it must be ASKED
// FOR: `vt::NoWriteback{}` is the opt-in, and without it the destination form
// refuses this type at compile time rather than degrading into the branch whose
// own comment says such a break must use the in-place form.
TEST_CASE("T10 CopyOutput: the non-copyable fallback is legal and must be asked for") {
  RequireCaptureLane();
  CHECK_FALSE(vt::detail::HasCopyOutput<int>::value);
  CHECK(vt::detail::HasCopyOutput<vt::Tensor>::value);

  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  BreakSlot<int> out;
  {
    GraphCaptureScope scope(b, q, g);
    GraphBreak([] { return 42; }, out, vt::NoWriteback{});
  }
  CHECK(*out == 42);
  CHECK(g.segment_count() == 2);  // it still splits
  CHECK(g.break_count() == 1);
  CHECK(b.copies() == 0);  // and it fabricates no copy

  // On replay the break function runs, and — this is the fallback — the seam
  // writes NOTHING back, because it has no way to copy into this destination.
  // That is upstream's `return src` (`:201`) and it is why such a break must use
  // the in-place form instead. Asserting the ABSENCE of the writeback is the
  // point: it is the warning a future author needs.
  b.ClearLog();
  *out = 0;
  g.Replay(q);
  CHECK(*out == 0);
  CHECK(b.copies() == 0);
}

// The destination form's REPLAY behavior, which is what D9 is about: on replay
// the eager operation produces a FRESH result and the seam writes it back into
// the capture-time destination, so the following segment's baked address holds
// the new data. A container that replayed the caller's raw `fn` and dropped its
// return value would leave the destination holding capture-time data forever —
// wrong numerics, not a fault.
TEST_CASE("T7b GraphBreak(fn slot): every REPLAY writes back into the capture-time address") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;

  BreakSlot<vt::Tensor> dst;
  int32_t feed = 11;
  // The break returns a FRESH buffer every call, exactly as an eager operation
  // allocating its own output does.
  auto fresh = [&] { return MakeBuf(b, {feed, feed, feed, feed}).t; };

  void* baked = nullptr;
  {
    GraphCaptureScope scope(b, q, g);
    GraphBreak(fresh, dst);
    // The destination is now the CAPTURE-TIME result, and this address is what
    // the following segment bakes. Upstream does the same: `output = inner(...)`
    // then `captured_output = _weak_ref_if_tensor(output)` (`:222-227`).
    baked = dst->data;
  }
  CHECK(baked != nullptr);
  CHECK(dst.pinned());  // the seam owns the destination, not this frame
  CHECK(ReadT(*dst) == std::vector<int32_t>{11, 11, 11, 11});

  for (int32_t v : {22, 33, 44}) {
    feed = v;
    g.Replay(q);
    CHECK(dst->data == baked);  // the address never moved
    CHECK(ReadT(*dst) == std::vector<int32_t>{v, v, v, v});
  }
}

// The destination OUTLIVES the frame that declared it. This is the rule the
// header used to state in a comment and the first production site broke: a
// `std::optional<DBuf>` local of the function containing the break dies on that
// function's `return`, while the following segment goes on reading its address
// on every replay. With `BreakSlot` the seam owns the storage, so the slot's
// frame can be gone and the writeback still lands.
TEST_CASE("Lifetime rule 1: the destination survives the frame that declared it") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  int32_t feed = 7;
  void* baked = nullptr;
  std::vector<int32_t> seen;

  {
    GraphCaptureScope scope(b, q, g);
    // The slot is a local of THIS block, exactly as the qwen3 site's was a local
    // of RunLayer, and it is destroyed before the replay below.
    {
      BreakSlot<vt::Tensor> inner;
      GraphBreak([&] { return MakeBuf(b, {feed, feed}).t; }, inner);
      baked = inner->data;
      // The following segment BAKES THE ADDRESS, which is what a real capture
      // records: a pointer, not a reference to whatever frame declared it.
      const vt::Tensor baked_view = *inner;
      b.Record([&, baked_view] { seen = ReadT(baked_view); });  // NOT run at capture
    }
  }
  REQUIRE(baked != nullptr);

  feed = 99;
  g.Replay(q);
  // The break wrote into the cell the closure owns, and the following segment
  // read the same address. A destination that died with its frame would have
  // been read after its storage was recycled.
  CHECK(seen == std::vector<int32_t>{99, 99});
}

// The ALIASING half of the SAME rule, which `BreakSlot` did NOT close. The type
// closed the LIFETIME half: the seam owns the storage, so no caller can hand the
// break a destination that dies first. It left the other half expressible. ONE
// slot reused for TWO break points inside one capture compiled, and
// `PinForCapture` returned the cell it had already made, so BOTH replay closures
// wrote through the SAME address: break 0's writeback was overwritten by break
// 1's on every replay, and any segment that baked break 0's destination read
// break 1's data. Measured on the head before this repair, `&*slot` was byte
// identical after break 0 and after break 1, and after `Replay` the slot held
// only break 1's value. That is the shape of the defect the review already
// found once at the production site, with the lifetime half fixed and this half
// open — and W2 through W5 add nine more callers.
//
// The rule is one destination per break point, and registration now REFUSES the
// second. Refusing at `AppendBreak` rather than at the slot is deliberate: every
// form of `GraphBreak` must state its destination to register at all, so a form
// added later cannot opt out by forgetting.
TEST_CASE("Aliasing: one BreakSlot reused for TWO break points in one capture is REFUSED") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  int32_t feed = 3;

  auto two_breaks_into_one_slot = [&] {
    GraphCaptureScope scope(b, q, g);
    BreakSlot<vt::Tensor> shared;
    GraphBreak([&] { return MakeBuf(b, {feed, feed}).t; }, shared);
    GraphBreak([&] { return MakeBuf(b, {feed * 10, feed * 10}).t; }, shared);
  };
  CHECK_THROWS_AS(two_breaks_into_one_slot(), std::runtime_error);
  // The refusal leaves NO capture behind. The exception propagates out of the
  // scope, so the drain sees it and destroys the half-built container rather
  // than handing back a forward that replays two segments of three.
  CHECK_FALSE(g.captured());
  CHECK(g.break_count() == 0);
  CHECK(GraphCaptureScope::Current() == nullptr);

  // CONTROL 1: two DISTINCT slots in the same capture are accepted, and each
  // keeps its own destination. A refusal that fired here would be a mute switch
  // on the whole destination form.
  BreakableGraph g2;
  BreakSlot<vt::Tensor> first, second;
  {
    GraphCaptureScope scope(b, q, g2);
    GraphBreak([&] { return MakeBuf(b, {feed, feed}).t; }, first);
    GraphBreak([&] { return MakeBuf(b, {feed * 10, feed * 10}).t; }, second);
  }
  CHECK(g2.segment_count() == 3);
  CHECK(g2.break_count() == 2);
  CHECK(first->data != second->data);
  feed = 5;
  g2.Replay(q);
  CHECK(ReadT(*first) == std::vector<int32_t>{5, 5});
  CHECK(ReadT(*second) == std::vector<int32_t>{50, 50});

  // CONTROL 2: the constraint is per CAPTURE, not per slot for life. Reusing one
  // slot in a LATER capture is legal, because the later capture bakes its own
  // addresses and there is only one closure writing through the cell.
  BreakableGraph g3;
  {
    GraphCaptureScope scope(b, q, g3);
    GraphBreak([&] { return MakeBuf(b, {feed + 1, feed + 1}).t; }, first);
  }
  CHECK(g3.segment_count() == 2);
  CHECK(g3.break_count() == 1);
  CHECK(ReadT(*first) == std::vector<int32_t>{6, 6});
}

// ---------------------------------------------------------------------------
// TestBreakGraphHelper (`:230`).
// ---------------------------------------------------------------------------

// T11 <- test_break_graph_inserts_segment (`:249-265`). The BARE marker splits
// the segment even though its body does nothing: x=10 -> +1=11 -> break -> +2=13.
TEST_CASE("T11 GraphBreak(): the BARE marker splits the segment and runs nothing") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;

  IntBuf x = MakeBuf(b, {0, 0, 0, 0});
  IntBuf t = MakeBuf(b, {0, 0, 0, 0});
  IntBuf y = MakeBuf(b, {0, 0, 0, 0});
  {
    GraphCaptureScope scope(b, q, g);
    b.Record([&] { AddInto(t, x.t, 1); });  // segment 0: t = x + 1
    GraphBreak();                           // the bare marker
    b.Record([&] { AddInto(y, t.t, 2); });  // segment 1: y = t + 2
  }
  CHECK(g.segment_count() == 2);
  CHECK(g.break_count() == 1);
  CHECK(b.Trace() == "Begin EndCaptureGraph Begin EndCaptureGraph");

  Fill(x, 10);  // upstream's x.fill_(10.0)
  b.ClearLog();
  g.Replay(q);
  // The bare marker's break function is empty, so replay is segments only.
  CHECK(b.Trace() == "ReplayGraph ReplayGraph");
  CHECK(Read(y) == std::vector<int32_t>{13, 13, 13, 13});  // x=10 -> 11 -> 13
}

// ---------------------------------------------------------------------------
// Tests this row owes with no upstream counterpart, numbered on from the ported
// set in the spec's `## Tests to port`.
// ---------------------------------------------------------------------------

// Test 12, REPLAY ORDER rather than replay arithmetic. Upstream asserts the
// composed VALUE, which a wrong order can still satisfy for a commutative chain.
// The break markers go into the BACKEND's own log, so segments and breaks form
// ONE sequence: a container that replayed every segment and then ran every break
// would satisfy two independent assertions and fails this one.
TEST_CASE("Test 12: replay emits segment break segment break segment IN THAT ORDER every time") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  {
    GraphCaptureScope scope(b, q, g);
    GraphBreak([&] { b.Note("break0"); });
    GraphBreak([&] { b.Note("break1"); });
  }
  REQUIRE(g.segment_count() == 3);
  REQUIRE(g.break_count() == 2);
  // The two break functions ran ONCE eagerly during capture, so the outputs hold
  // real data (breakable_cuda_graph.py:222-223).
  CHECK(b.Trace() ==
        "Begin EndCaptureGraph break0 Begin EndCaptureGraph break1 Begin EndCaptureGraph");

  for (int rep = 0; rep < 3; ++rep) {
    b.ClearLog();
    g.Replay(q);
    CHECK(b.Trace() == "ReplayGraph break0 ReplayGraph break1 ReplayGraph");
    CHECK(g.replay_count() == rep + 1);
  }
  CHECK(b.replayed().size() == 3);
}

// Test 13, the CAPTURE-FAILURE DRAIN. The spec recorded this as "the destructor
// already catches and resets; what is owed is the test", and that was not true:
// the catch only ever guarded a throwing `EndCaptureGraph`. An exception from
// ordinary user code mid-capture left a partial 2-segment graph reporting
// `captured() == true`, which is replayable as HALF A FORWARD. All three arms
// below are the behaviour, not only its test.
TEST_CASE("Test 13a: a break function that THROWS during capture leaves nothing captured") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  CHECK_THROWS_AS(
      [&] {
        GraphCaptureScope scope(b, q, g);
        GraphBreak([] { throw std::runtime_error("break blew up"); });
      }(),
      std::runtime_error);

  CHECK_FALSE(g.captured());
  CHECK(g.segment_count() == 0);
  CHECK(g.break_count() == 0);
  CHECK(b.live_graphs() == 0);       // every instantiated segment released
  CHECK(b.Count("EndCaptureGraph") == 1);  // and the stream taken out of capture
  CHECK(b.Count("DestroyGraph") == 1);
  CHECK(GraphCaptureScope::Current() == nullptr);
}

TEST_CASE("Test 13b: ORDINARY code throwing mid-segment leaves no partial capture") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  CHECK_THROWS_AS(
      [&] {
        GraphCaptureScope scope(b, q, g);
        GraphBreak();  // one clean break: two segments are now in flight
        throw std::runtime_error("model code blew up");
      }(),
      std::runtime_error);

  CHECK_FALSE(g.captured());
  CHECK(g.segment_count() == 0);
  CHECK(b.live_graphs() == 0);
  CHECK(b.Count("EndCaptureGraph") == 2);
  CHECK(GraphCaptureScope::Current() == nullptr);
}

TEST_CASE("Test 13c: a throwing EndCaptureGraph never propagates out of the destructor") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  {
    GraphCaptureScope scope(b, q, g);
    GraphBreak();
    b.FailNextEndCapture();  // the FINAL close refuses
  }  // a destructor that propagated would terminate
  CHECK_FALSE(g.captured());
  CHECK(b.live_graphs() == 0);
  CHECK(GraphCaptureScope::Current() == nullptr);
}

// The invariant `segment_count() == break_count() + 1`, asserted rather than
// documented. Re-entering a scope on an already-captured container appended to
// it and produced `seg == brk`, whose last break `Replay` silently drops.
TEST_CASE("Invariant: a scope REFUSES a BreakableGraph that already holds a capture") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  {
    GraphCaptureScope scope(b, q, g);
    GraphBreak();
  }
  REQUIRE(g.segment_count() == 2);
  REQUIRE(g.break_count() == 1);

  CHECK_THROWS_AS(GraphCaptureScope(b, q, g), std::runtime_error);
  CHECK(g.segment_count() == 2);  // untouched
  CHECK(g.break_count() == 1);
  CHECK(GraphCaptureScope::Current() == nullptr);

  g.Replay(q);
  REQUIRE(g.replay_count() == 1);

  g.Reset();  // the documented way to capture into it again
  CHECK_FALSE(g.captured());
  // A released graph's replay count goes with it. A count left behind describes
  // a graph that no longer exists, and G3's whole job is to be the number nobody
  // has to trust twice.
  CHECK(g.replay_count() == 0);
  {
    GraphCaptureScope scope(b, q, g);
    GraphBreak();
  }
  CHECK(g.segment_count() == 2);
  CHECK(g.break_count() == 1);
}

// Nesting is REFUSED. `BeginCapture` on a stream that is already capturing is
// CUDA_ERROR_ILLEGAL_STATE on a real backend, and the trace a nested pair emits
// is not a legal capture sequence on any backend.
TEST_CASE("Nesting: a second scope on the same thread is REFUSED before any backend call") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph outer_g, inner_g;
  {
    GraphCaptureScope outer(b, q, outer_g);
    REQUIRE(outer.active());
    CHECK_THROWS_AS(GraphCaptureScope(b, q, inner_g), std::runtime_error);
    // The refusal made no backend call: still exactly one Begin.
    CHECK(b.Count("Begin") == 1);
    CHECK(b.Count("EndCaptureGraph") == 0);
  }
  CHECK(outer_g.segment_count() == 1);  // the outer scope closed normally
  CHECK(inner_g.segment_count() == 0);
  CHECK(GraphCaptureScope::Current() == nullptr);
}

// ---------------------------------------------------------------------------
// Ownership: the container routes every acquisition and release through
// Backend::EndCaptureGraph and Backend::DestroyGraph, treating a segment handle
// as OPAQUE, so ENG-CUDAGRAPH-DEDUP (#1162) can interpose at the backend
// without editing the container (spec `## Risks/decisions` D4).
// ---------------------------------------------------------------------------
TEST_CASE("BreakableGraph: destruction releases EVERY segment through DestroyGraph") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  {
    BreakableGraph g;
    {
      GraphCaptureScope scope(b, q, g);
      GraphBreak();
      GraphBreak();
    }
    REQUIRE(g.segment_count() == 3);
    CHECK(b.live_graphs() == 3);
  }
  CHECK(b.destroyed().size() == 3);
  CHECK(b.live_graphs() == 0);
}

// Test 14, the NON-CAPTURING BACKEND (spec `## Tests to port`). Vulkan
// (`vulkan_backend.cpp:16`) and Metal (`metal_backend.mm:13`) are the live
// cases: the scope is inert, the forward runs eager, and GraphBreak makes ZERO
// backend calls. This is the OTHER conjunct of `active_`; T4 covers the switch.
TEST_CASE("Test 14: on a backend that cannot capture the scope is inert and the forward is eager") {
  RecordingCaptureBackend no_capture(/*supports_capture=*/false);
  vt::Queue q = no_capture.CreateQueue();
  BreakableGraph g;
  int ran = 0;
  {
    GraphCaptureScope scope(no_capture, q, g);
    CHECK_FALSE(scope.active());
    GraphBreak([&] { ++ran; });
  }
  CHECK(ran == 1);  // the break function still runs
  CHECK(g.segment_count() == 0);
  CHECK(g.break_count() == 0);
  CHECK(no_capture.Trace().empty());  // zero backend calls
}

// The G3 observability counters (spec `## Gates` G3): without them there is no
// way to tell a two-segment capture from a fully eager step, and "the graph ran"
// is exactly the claim a broken instrument fabricates.
TEST_CASE("G3: the seam reports segments captured, breaks registered and replays run") {
  RequireCaptureLane();
  vt::ResetGraphBreakStats();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  {
    GraphCaptureScope scope(b, q, g);
    GraphBreak();
  }
  g.Replay(q);
  g.Replay(q);
  const vt::GraphBreakStats s = vt::GetGraphBreakStats();
  CHECK(s.break_points_reached == 1);
  CHECK(s.segments_captured == 2);
  CHECK(s.breaks_registered == 1);
  CHECK(s.replays == 2);
}

// ---------------------------------------------------------------------------
// vt::GraphCaptureMode — the second half of vLLM's capture contract, added by
// W2 (#1261). Upstream has no counterpart: SGLang's BCG IS the piecewise mode
// and has no full-graph arm to select. The primary oracle does, and it is the
// one a decode driver reaches for.
// ---------------------------------------------------------------------------

// The DEFAULT is piecewise, stated as a case rather than left to the twenty-odd
// cases above that construct the three-argument scope. A default that silently
// changed would turn every one of them into a one-segment capture whose break
// assertions then fail for a reason none of them names.
TEST_CASE("Mode: the default scope is kPiecewise and it splits") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  {
    GraphCaptureScope scope(b, q, g);
    CHECK(scope.mode() == vt::GraphCaptureMode::kPiecewise);
    CHECK(scope.active());
    CHECK(scope.splits());
    vt::GraphBreak();
  }
  CHECK(g.segment_count() == 2);
  CHECK(g.break_count() == 1);
}

// kFULL <- vLLM `CUDAGraphMode.FULL` (`vllm/config/compilation.py:61`), the half
// of `FULL_AND_PIECEWISE` (`:63`) that `decode_mode()` (`:65-66`) selects for a
// uniform decode batch, documented at `:630-632`. A break point inside such a
// scope runs its function INSIDE the single segment and splits nothing, which is
// exactly what a captured decode step has always done.
//
// The break function issues its work through `Record`, so the assertion is not
// "the break was skipped" but the stronger "the break's work landed in the ONE
// segment and runs on every replay" — the difference between a full capture and
// a step that quietly lost an operation.
TEST_CASE("Mode: a kFull scope captures ONE segment and the break's work is INSIDE it") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;

  IntBuf x = MakeBuf(b, {0, 0, 0, 0});
  IntBuf mid = MakeBuf(b, {0, 0, 0, 0});
  IntBuf y = MakeBuf(b, {0, 0, 0, 0});
  IntBuf broken_dst = MakeBuf(b, {0, 0, 0, 0});
  BreakSlot<vt::Tensor> broken;

  const vt::GraphBreakStats before = vt::GetGraphBreakStats();
  {
    GraphCaptureScope scope(b, q, g, vt::GraphCaptureMode::kFull);
    REQUIRE(scope.active());
    CHECK(scope.mode() == vt::GraphCaptureMode::kFull);
    CHECK_FALSE(scope.splits());
    b.Record([&] { AddInto(mid, x.t, 1); });  // mid = x + 1
    GraphBreak(
        [&] {
          // Work a real break function would issue on the queue. In kFull the
          // stream is still capturing, so it belongs to the one segment.
          b.Record([&] { AddInto(broken_dst, mid.t, 0); });
          return broken_dst.t;
        },
        broken);
    b.Record([&] { AddInto(y, *broken, 3); });  // y = broken + 3
  }
  const vt::GraphBreakStats after = vt::GetGraphBreakStats();

  // ONE segment, NO break function, and ONE Begin/EndCaptureGraph pair: the
  // capture never re-began.
  CHECK(g.segment_count() == 1);
  CHECK(g.break_count() == 0);
  CHECK(b.Trace() == "Begin EndCaptureGraph");
  CHECK(after.segments_captured - before.segments_captured == 1);
  CHECK(after.breaks_registered - before.breaks_registered == 0);
  // The site was still REACHED, which is the counter that separates a kFull
  // capture from one whose break-point registration was deleted.
  CHECK(after.break_points_reached - before.break_points_reached == 1);
  // The slot took the pass-through path, so the seam owns no cell for it.
  CHECK_FALSE(broken.pinned());

  Fill(x, 10);
  b.ClearLog();
  g.Replay(q);
  CHECK(b.Trace() == "ReplayGraph");
  // x=10 -> +1=11 -> break copies 11 -> +3=14. The break's operation ran at
  // replay, from inside the segment.
  CHECK(Read(mid) == std::vector<int32_t>{11, 11, 11, 11});
  CHECK(Read(broken_dst) == std::vector<int32_t>{11, 11, 11, 11});
  CHECK(Read(y) == std::vector<int32_t>{14, 14, 14, 14});
}

// The refusal at the ONE registration point. Every `GraphBreak` form already
// takes the pass-through arm in kFull, so this is what stops a form added later
// from registering a break into a capture that has a single segment — where
// `break_count() == segment_count()` and `Replay` drops the last break.
TEST_CASE("Mode: registering a break into a kFull scope is REFUSED") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();
  BreakableGraph g;
  {
    GraphCaptureScope scope(b, q, g, vt::GraphCaptureMode::kFull);
    REQUIRE(scope.active());
    CHECK_THROWS(scope.AppendBreak([] {}, nullptr));
    // The CONTROL: the identical call on a piecewise scope is accepted, so the
    // refusal is about the mode and not about the arguments.
  }
  BreakableGraph g2;
  {
    GraphCaptureScope scope(b, q, g2, vt::GraphCaptureMode::kPiecewise);
    CHECK_NOTHROW(scope.AppendBreak([] {}, nullptr));
    scope.EndSegment();
    scope.BeginSegment();
  }
  CHECK(g2.break_count() == 1);
  CHECK(g2.segment_count() == 2);
}

// THE MODE COUNTERS, added by W3 (#1291) because the mode was UNOBSERVABLE from
// outside a driver and a mute guard was measured hiding that.
//
// WHAT WENT WRONG WITHOUT THEM. W2's driver gate could pin its scope's mode only
// because `qwen3.cpp` registers a production `vt::GraphBreak`, so
// `breaks_registered` moved in one mode and not the other. The three drivers W3
// migrated register none, so `breaks_registered == 0` holds in BOTH modes there.
// Measured, not reasoned: flipping `kFull` to `kPiecewise` in `qwen3_moe.cpp`
// compiled clean and left that driver's whole gate green at 226/226. The mode is
// the difference between one graph and one eager attention call per layer, and a
// token gate cannot see a segment count, so it needed an observable of its own.
//
// THE INERT ARM IS THE CONTROL that stops the counters from being "scopes
// constructed". A scope that cannot capture makes no backend call in either
// mode, so counting it would report a mode that never reached a backend.
TEST_CASE("Mode: the scope counters report which mode an ACTIVE capture opened in") {
  RequireCaptureLane();
  RecordingCaptureBackend b;
  vt::Queue q = b.CreateQueue();

  const vt::GraphBreakStats before = vt::GetGraphBreakStats();
  {
    BreakableGraph g;
    GraphCaptureScope scope(b, q, g, vt::GraphCaptureMode::kFull);
    REQUIRE(scope.active());
  }
  const vt::GraphBreakStats after_full = vt::GetGraphBreakStats();
  CHECK(after_full.full_scopes - before.full_scopes == 1);
  CHECK(after_full.piecewise_scopes - before.piecewise_scopes == 0);

  {
    BreakableGraph g;
    GraphCaptureScope scope(b, q, g, vt::GraphCaptureMode::kPiecewise);
    REQUIRE(scope.active());
  }
  const vt::GraphBreakStats after_piece = vt::GetGraphBreakStats();
  CHECK(after_piece.full_scopes - after_full.full_scopes == 0);
  CHECK(after_piece.piecewise_scopes - after_full.piecewise_scopes == 1);

  // THE CONTROL: an INERT scope counts as neither. This backend answers
  // `SupportsGraphCapture()` false, which is Vulkan and Metal.
  RecordingCaptureBackend nb(/*supports_capture=*/false);
  vt::Queue nq = nb.CreateQueue();
  {
    BreakableGraph g;
    GraphCaptureScope scope(nb, nq, g, vt::GraphCaptureMode::kFull);
    REQUIRE_FALSE(scope.active());
  }
  {
    BreakableGraph g;
    GraphCaptureScope scope(nb, nq, g, vt::GraphCaptureMode::kPiecewise);
    REQUIRE_FALSE(scope.active());
  }
  const vt::GraphBreakStats after_inert = vt::GetGraphBreakStats();
  CHECK(after_inert.full_scopes - after_piece.full_scopes == 0);
  CHECK(after_inert.piecewise_scopes - after_piece.piecewise_scopes == 0);

  // And ResetGraphBreakStats() clears them, like every other G3 counter: a
  // number that survives its own reset describes a run nobody can bound.
  vt::ResetGraphBreakStats();
  const vt::GraphBreakStats cleared = vt::GetGraphBreakStats();
  CHECK(cleared.full_scopes == 0);
  CHECK(cleared.piecewise_scopes == 0);
}

// ---------------------------------------------------------------------------
// Test 13d, the DISTINCTION the drain owed a driver. W2's fresh review found the
// HIGH this closes.
// ---------------------------------------------------------------------------
//
// The drain (13a/13b/13c) is correct and stays: a destructor that propagated
// would terminate. But it leaves the container in the SAME observable state as a
// scope that was never active — `captured() == false` — and those two states have
// OPPOSITE meanings for the caller. An INERT scope ran the forward EAGERLY, so
// the values the caller holds are real. A FAILED capture ran NOTHING: under
// stream capture every operation between `BeginCapture` and the failure was
// RECORDED, not executed, so the caller's buffers hold whatever the pool last
// left there. A driver that cannot tell the two apart returns uncomputed memory
// as its step's result, silently, and no token gate can see it
// (`src/vllm/model_executor/models/qwen3.cpp`, the HIGH this case gates).
//
// `capture_failed()` is that distinction, and `capture_error()` carries the
// ORIGINAL exception where the seam holds it, so a driver can propagate the
// runtime's own diagnosis instead of inventing one.
TEST_CASE("Test 13d: a DRAINED capture records that it drained; an INERT scope does not") {
  RequireCaptureLane();

  // ARM 1, the FAILED capture: the close refuses, the destructor swallows (it
  // must), and the container says so.
  {
    RecordingCaptureBackend b;
    vt::Queue q = b.CreateQueue();
    BreakableGraph g;
    {
      GraphCaptureScope scope(b, q, g);
      REQUIRE(scope.active());
      b.FailNextEndCapture();
    }
    CHECK_FALSE(g.captured());
    CHECK(g.capture_failed());
    REQUIRE(g.capture_error() != nullptr);
    // The ORIGINAL error, not a substitute: a driver rethrows this so the
    // runtime's own message reaches the operator.
    CHECK_THROWS_AS(std::rethrow_exception(g.capture_error()), std::runtime_error);
  }

  // ARM 2, an exception PROPAGATING through the scope. The drain fires on the
  // uncaught-depth comparison rather than on a throwing close, and the container
  // records the FAILURE but not a cause — pinned here because it is the one
  // place the two accessors disagree, and the disagreement is a language rule,
  // not a bug. No handler has been entered for an exception that is still
  // unwinding, so `std::current_exception()` in `~GraphCaptureScope` is null.
  // Nothing is lost: on this arm the real exception reaches the caller under its
  // own power, which is what `CHECK_THROWS_AS` below observes.
  {
    RecordingCaptureBackend b;
    vt::Queue q = b.CreateQueue();
    BreakableGraph g;
    CHECK_THROWS_AS(
        [&] {
          GraphCaptureScope scope(b, q, g);
          throw std::runtime_error("model code blew up");
        }(),
        std::runtime_error);
    CHECK_FALSE(g.captured());
    CHECK(g.capture_failed());
    CHECK(g.capture_error() == nullptr);
  }

  // ARM 3, the INERT scope — the CONTROL, and the arm that makes this a
  // distinction rather than a flag that is always set. The backend cannot
  // capture, the forward ran eager, and there is no error to report. Without
  // this arm a driver that treated `!captured()` as failure would look gated.
  {
    RecordingCaptureBackend b(/*supports_capture=*/false);
    vt::Queue q = b.CreateQueue();
    BreakableGraph g;
    {
      GraphCaptureScope scope(b, q, g);
      CHECK_FALSE(scope.active());
    }
    CHECK_FALSE(g.captured());
    CHECK_FALSE(g.capture_failed());
    CHECK(g.capture_error() == nullptr);
  }

  // ARM 4, a CLEAN capture. Nothing failed, so nothing is recorded — and
  // `Reset()` returns the container to its as-constructed state, error included,
  // so a stale failure cannot outlive the capture that produced it.
  {
    RecordingCaptureBackend b;
    vt::Queue q = b.CreateQueue();
    BreakableGraph g;
    {
      GraphCaptureScope scope(b, q, g);
      REQUIRE(scope.active());
    }
    REQUIRE(g.captured());
    CHECK_FALSE(g.capture_failed());
    CHECK(g.capture_error() == nullptr);

    g.Reset();
    CHECK_FALSE(g.capture_failed());
    CHECK(g.capture_error() == nullptr);

    // Now FAIL a capture on the same container, and then SUCCEED on it without
    // an intervening `Reset()` — which is legal, because the drain already reset
    // it. The clear at scope ENTRY is what this pins: without it the container
    // would report a graph it holds AND a failure it recovered from, and a
    // driver reading the failure first would refuse a step that captured fine.
    {
      GraphCaptureScope scope(b, q, g);
      b.FailNextEndCapture();
    }
    REQUIRE(g.capture_failed());
    REQUIRE(g.capture_error() != nullptr);
    {
      GraphCaptureScope scope(b, q, g);
      REQUIRE(scope.active());
    }
    REQUIRE(g.captured());
    CHECK_FALSE(g.capture_failed());
    CHECK(g.capture_error() == nullptr);
  }
}

// ---------------------------------------------------------------------------
// TEST 15 of `## Tests to port` — THE AUXILIARY-STREAM AUTO-JOIN (D10).
// Port of `_end_current_segment` (`:353-361`) and of the `wait_stream` hook
// (`:101-153`) that populates the set it walks. ENG-CUDAGRAPH-BREAK W5 (#1335).
// ---------------------------------------------------------------------------
//
// WHY THIS COULD NOT BE WRITTEN BEFORE W5, and why the record says so rather
// than implying the rule was always covered. W1 registered its break point on a
// model that forks no auxiliary queue. W2, W3 and W4 all migrated drivers that
// open `kFull`, which has exactly ONE segment and therefore no segment CLOSE
// between two pieces of a forward for an outstanding fork to straddle. W5 is the
// first stage that owns a driver whose fork is inside the captured region by
// construction (`laguna.cpp:2572-2576` fork, `:2612` join).
//
// WHAT IT ASSERTS AND WHY IT IS AN ORDER. Closing a capture with an unjoined
// fork FAILS at `cudaStreamEndCapture`, so "the join happened" is not the claim
// — "the join happened BEFORE the close" is. Both ends are therefore asserted
// out of ONE backend trace, for the same reason W1 had to move break markers
// into the backend's own log: two independently asserted sequences are satisfied
// by an implementation that interleaves nothing.
TEST_CASE("BreakableGraph: an outstanding fork is joined BEFORE the segment closes") {
  using vt::BreakableGraph;
  using vt::GraphCaptureMode;
  using vt::GraphCaptureScope;

  // ARM 1, THE RULE. A fork registered inside a piecewise capture and never
  // joined by the model is joined by the SCOPE, at the break point that closes
  // the segment and again at the scope's own close.
  {
    RecordingCaptureBackend b;
    vt::Queue q = b.CreateQueue();
    vt::Queue aux = b.CreateQueue();
    vt::Event done = b.CreateEvent();
    vt::ResetGraphBreakStats();
    BreakableGraph g;
    {
      GraphCaptureScope scope(b, q, g, GraphCaptureMode::kPiecewise);
      REQUIRE(scope.active());
      // The model forks and tells the scope, exactly as `laguna.cpp` does.
      vt::GraphNoteFork(aux, done);
      CHECK(scope.outstanding_forks() == 1);
      // ... and then hits a break point WITHOUT joining. Upstream's hazard.
      vt::GraphBreak();
      // The scope joined it, so nothing is outstanding for the next segment.
      CHECK(scope.outstanding_forks() == 0);
    }
    REQUIRE(g.captured());
    CHECK(g.segment_count() == 2);
    CHECK(g.break_count() == 1);
    // THE ORDER, out of one trace. `QueueWaitEvent` sits between the fork's
    // `RecordEvent` and the `EndCaptureGraph` it protects, on the FIRST segment.
    CHECK(b.Trace() ==
          "Begin RecordEvent QueueWaitEvent EndCaptureGraph Begin EndCaptureGraph");
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.forks_tracked == 1);
    CHECK(s.forks_auto_joined == 1);
  }

  // ARM 2, THE CONTROL that makes arm 1 a distinction rather than a flag that is
  // always set. The model joins its OWN fork before the break, which is what
  // every shipped fork site actually does. The scope must then do NOTHING: no
  // second wait, no extra event record, and `forks_auto_joined` stays 0. Without
  // this arm an implementation that joined unconditionally on every segment
  // close would pass arm 1 while issuing a redundant wait on every real step.
  {
    RecordingCaptureBackend b;
    vt::Queue q = b.CreateQueue();
    vt::Queue aux = b.CreateQueue();
    vt::Event done = b.CreateEvent();
    vt::ResetGraphBreakStats();
    BreakableGraph g;
    {
      GraphCaptureScope scope(b, q, g, GraphCaptureMode::kPiecewise);
      REQUIRE(scope.active());
      vt::GraphNoteFork(aux, done);
      CHECK(scope.outstanding_forks() == 1);
      b.RecordEvent(done, aux);      // the model's own join, first half
      b.QueueWaitEvent(q, done);     // ... and its second half
      vt::GraphNoteJoin(aux);        // the registration is retired
      CHECK(scope.outstanding_forks() == 0);
      vt::GraphBreak();
      CHECK(scope.outstanding_forks() == 0);
    }
    REQUIRE(g.captured());
    // ONE RecordEvent and ONE QueueWaitEvent in the whole trace — the model's.
    CHECK(b.Count("RecordEvent") == 1);
    CHECK(b.Count("QueueWaitEvent") == 1);
    CHECK(b.Trace() ==
          "Begin RecordEvent QueueWaitEvent EndCaptureGraph Begin EndCaptureGraph");
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.forks_tracked == 1);
    CHECK(s.forks_auto_joined == 0);
  }

  // ARM 3, `kFull`. The mode every migrated driver opens. A fork registered here
  // is still tracked and is still joined before the ONE segment closes — which
  // is the arm `laguna.cpp` actually takes, because its fork and join both sit
  // inside `RunChain`. It is asserted rather than assumed, because "the set is
  // empty so the rule is vacuous" is a claim about the MODEL, not about the seam,
  // and a driver that returns early between its fork and its join would make it
  // false without changing a line of this file.
  {
    RecordingCaptureBackend b;
    vt::Queue q = b.CreateQueue();
    vt::Queue aux = b.CreateQueue();
    vt::Event done = b.CreateEvent();
    vt::ResetGraphBreakStats();
    BreakableGraph g;
    {
      GraphCaptureScope scope(b, q, g, GraphCaptureMode::kFull);
      REQUIRE(scope.active());
      vt::GraphNoteFork(aux, done);
      CHECK(scope.outstanding_forks() == 1);
    }
    REQUIRE(g.captured());
    CHECK(g.segment_count() == 1);
    CHECK(b.Trace() == "Begin RecordEvent QueueWaitEvent EndCaptureGraph");
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.forks_auto_joined == 1);
  }

  // ARM 4, THE INERT SCOPE. Outside a capture, and inside a scope the backend
  // cannot honour, both calls make ZERO backend calls and move no counter — the
  // same pass-through guarantee `GraphBreak` gives, applied to the fork hooks.
  // A model calls them unconditionally, so a version that tracked in the inert
  // lane would issue joins on a forward that never captured.
  {
    RecordingCaptureBackend b(/*supports_capture=*/false);
    vt::Queue q = b.CreateQueue();
    vt::Queue aux = b.CreateQueue();
    vt::Event done = b.CreateEvent();
    vt::ResetGraphBreakStats();
    vt::GraphNoteFork(aux, done);  // no scope at all
    vt::GraphNoteJoin(aux);
    BreakableGraph g;
    {
      GraphCaptureScope scope(b, q, g);
      REQUIRE_FALSE(scope.active());
      vt::GraphNoteFork(aux, done);
      CHECK(scope.outstanding_forks() == 0);
    }
    CHECK(b.Trace().empty());
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.forks_tracked == 0);
    CHECK(s.forks_auto_joined == 0);
  }

  // ARM 5, RE-REGISTERING ONE QUEUE. A model that forks the same auxiliary queue
  // in two layers of one capture must leave ONE entry, not two: two entries make
  // the auto-join issue two waits for one fork, and — the half that actually
  // corrupts state — a single `GraphNoteJoin` would then retire only one of them
  // and leave a joined queue looking outstanding forever.
  {
    RecordingCaptureBackend b;
    vt::Queue q = b.CreateQueue();
    vt::Queue aux = b.CreateQueue();
    vt::Event done = b.CreateEvent();
    vt::ResetGraphBreakStats();
    BreakableGraph g;
    {
      GraphCaptureScope scope(b, q, g, GraphCaptureMode::kFull);
      REQUIRE(scope.active());
      vt::GraphNoteFork(aux, done);
      vt::GraphNoteFork(aux, done);
      CHECK(scope.outstanding_forks() == 1);
      vt::GraphNoteJoin(aux);
      CHECK(scope.outstanding_forks() == 0);
    }
    CHECK(b.Count("QueueWaitEvent") == 0);
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.forks_tracked == 1);
    CHECK(s.forks_auto_joined == 0);
  }
}
