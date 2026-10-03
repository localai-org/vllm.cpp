// Implementation of the break-point capture seam. See
// `include/vt/breakable_graph.h` for the port map and the lifetime rules, and
// `.agents/specs/eng-cudagraph-break.md` for the design. Row
// ENG-CUDAGRAPH-BREAK W1, issue #1192, parent #1163.
//
// Upstream: SGLang at pin `f63458b5be`,
// `python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`;
// every bare `:N` is a line in that file.
#include "vt/breakable_graph.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace vt {
namespace {

// The capture scope of the CURRENT THREAD. Replaces upstream's `ContextVar`
// (`:63`); see the header for why this is thread-local and not global.
thread_local GraphCaptureScope* t_current_scope = nullptr;

std::atomic<int64_t> g_break_points{0};
std::atomic<int64_t> g_segments{0};
std::atomic<int64_t> g_breaks{0};
std::atomic<int64_t> g_replays{0};
std::atomic<int64_t> g_full_scopes{0};
std::atomic<int64_t> g_piecewise_scopes{0};
std::atomic<int64_t> g_forks_tracked{0};
std::atomic<int64_t> g_forks_auto_joined{0};

}  // namespace

namespace detail {
void CountBreakPoint() { g_break_points.fetch_add(1, std::memory_order_relaxed); }
}  // namespace detail

GraphBreakStats GetGraphBreakStats() {
  GraphBreakStats s;
  s.break_points_reached = g_break_points.load(std::memory_order_relaxed);
  s.segments_captured = g_segments.load(std::memory_order_relaxed);
  s.breaks_registered = g_breaks.load(std::memory_order_relaxed);
  s.replays = g_replays.load(std::memory_order_relaxed);
  s.full_scopes = g_full_scopes.load(std::memory_order_relaxed);
  s.piecewise_scopes = g_piecewise_scopes.load(std::memory_order_relaxed);
  s.forks_tracked = g_forks_tracked.load(std::memory_order_relaxed);
  s.forks_auto_joined = g_forks_auto_joined.load(std::memory_order_relaxed);
  return s;
}

void ResetGraphBreakStats() {
  g_break_points.store(0, std::memory_order_relaxed);
  g_segments.store(0, std::memory_order_relaxed);
  g_breaks.store(0, std::memory_order_relaxed);
  g_replays.store(0, std::memory_order_relaxed);
  g_full_scopes.store(0, std::memory_order_relaxed);
  g_piecewise_scopes.store(0, std::memory_order_relaxed);
  g_forks_tracked.store(0, std::memory_order_relaxed);
  g_forks_auto_joined.store(0, std::memory_order_relaxed);
}

// Read ONCE into a function-local static, so a process is in exactly one lane
// for its whole life and nothing can toggle it mid-run.
bool GraphCaptureEnabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("VLLM_CPP_CUDAGRAPH");
    return e == nullptr || std::string(e) != "0";
  }();
  return enabled;
}

// ---------------------------------------------------------------------------
// CopyOutput — port of `_copy_output` (`:172-201`).
// ---------------------------------------------------------------------------

static size_t Numel(const Tensor& t) {
  size_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= static_cast<size_t>(t.shape[i]);
  return n;
}

void CopyOutput(Backend& b, Queue& q, Tensor& dst, const Tensor& src) {
  if (dst.data == nullptr || src.data == nullptr) return;
  if (dst.data == src.data) return;  // already the same storage: nothing to do
  // The two sides must AGREE before a byte moves. Sizing the copy from `dst`
  // alone turns a break function that returned the wrong shape or dtype into an
  // out-of-bounds read of `src` — a fault at best and someone else's data at
  // worst — where upstream's `dst.copy_(src)` raises. This is the refusal that
  // makes a mis-registered break point loud instead of silently wrong.
  VT_CHECK(dst.dtype == src.dtype,
           "CopyOutput: break output dtype does not match its destination");
  VT_CHECK(Numel(dst) == Numel(src),
           "CopyOutput: break output element count does not match its destination");
  const size_t bytes = Numel(dst) * SizeOf(dst.dtype);
  // In place, into the address the FOLLOWING SEGMENT baked. `dst` is not
  // rebound, which is upstream's `assertIs(result, dst)` (`:190`).
  b.Copy(q, dst.data, src.data, bytes);
}

void CopyOutput(Backend& b, Queue& q, std::vector<Tensor>& dst,
                const std::vector<Tensor>& src) {
  const size_t n = dst.size() < src.size() ? dst.size() : src.size();
  for (size_t i = 0; i < n; ++i) CopyOutput(b, q, dst[i], src[i]);
}

void CopyOutput(Backend& b, Queue& q, std::map<std::string, Tensor>& dst,
                const std::map<std::string, Tensor>& src) {
  for (const auto& kv : src) {
    auto it = dst.find(kv.first);
    // A key the destination holds is copied IN PLACE; a key it does not hold is
    // ASSIGNED, which is upstream's `dst[key] = src_val` branch (`:200`).
    if (it != dst.end())
      CopyOutput(b, q, it->second, kv.second);
    else
      dst[kv.first] = kv.second;
  }
}

// ---------------------------------------------------------------------------
// BreakableGraph — port of `BreakableCUDAGraph` (`:246-274`).
// ---------------------------------------------------------------------------

BreakableGraph::~BreakableGraph() { Reset(); }

void BreakableGraph::Reset() {
  if (backend_ != nullptr) {
    for (void* g : segments_) backend_->DestroyGraph(g);
  }
  segments_.clear();
  break_fns_.clear();
  // The replay count describes the graph that was just released. Leaving it
  // behind makes the next capture report replays it never ran, and G3's whole
  // job is to be the number nobody has to trust twice.
  replays_ = 0;
  // Same argument for the recorded failure: it describes ONE capture attempt.
  // A driver that rethrows it clears it by resetting, and a stale one would
  // make the NEXT step propagate an error its own capture never hit.
  capture_failed_ = false;
  capture_error_ = nullptr;
}

void BreakableGraph::Replay(Queue& q) {
  if (backend_ == nullptr || segments_.empty()) return;
  // The invariant, asserted where violating it would be SILENT. A container with
  // `break_count() == segment_count()` replays with its last break dropped, and
  // the only symptom is a forward that skips one host-dependent operation.
  VT_CHECK(break_fns_.size() + 1 == segments_.size(),
           "BreakableGraph::Replay: segment_count() must equal break_count() + 1");
  // Interleaved, mirroring `:255-263`: segment i, then break i.
  for (size_t i = 0; i < segments_.size(); ++i) {
    backend_->ReplayGraph(q, segments_[i]);
    if (i + 1 < segments_.size()) break_fns_[i]();
  }
  ++replays_;
  g_replays.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// GraphCaptureScope — port of `BreakableCUDAGraphCapture` (`:277-367`).
// ---------------------------------------------------------------------------

GraphCaptureScope* GraphCaptureScope::Current() { return t_current_scope; }

GraphCaptureScope::GraphCaptureScope(Backend& b, Queue& q, BreakableGraph& out,
                                     GraphCaptureMode mode)
    : b_(&b),
      q_(&q),
      g_(&out),
      mode_(mode),
      uncaught_on_entry_(std::uncaught_exceptions()) {
  // Refused before ANY backend call, and refused in both lanes so the switch
  // cannot change which programs are legal.
  //
  // NESTING. `prev_`-style save and restore made a nested pair look contemplated;
  // it is not. Two scopes on one queue trace `Begin Begin EndCaptureGraph Begin
  // EndCaptureGraph EndCaptureGraph ...`, whose second `BeginCapture` lands on an
  // already-capturing stream — `CUDA_ERROR_ILLEGAL_STATE` on a real backend.
  VT_CHECK(t_current_scope == nullptr,
           "GraphCaptureScope: a capture scope is already open on this thread");
  // RE-ENTRY. Appending to a container that already holds a capture yields
  // `break_count() == segment_count()`, whose last break Replay drops. Reset it.
  VT_CHECK(!out.captured() && out.break_count() == 0,
           "GraphCaptureScope: this BreakableGraph already holds a capture; Reset() it first");
  active_ = GraphCaptureEnabled() && b.SupportsGraphCapture();
  if (!active_) return;  // inert: the forward runs eager, exactly as today
  // A fresh capture attempt starts with no failure recorded. Clearing here and
  // not only in `Reset()` is what stops a caller that swallowed one attempt's
  // error from seeing it again after the next attempt SUCCEEDED.
  g_->capture_failed_ = false;
  g_->capture_error_ = nullptr;
  g_->backend_ = b_;
  // The MODE this capture opened in, counted where it is decided. Nothing
  // outside a driver could read it before, and a decode driver that opened
  // `kPiecewise` would split a shipped one-graph step into one eager attention
  // call per layer with no gate in this tree able to see it; a token gate cannot
  // see a segment count either. Counted only on the ACTIVE path, because an
  // inert scope captures nothing in either mode.
  if (mode_ == GraphCaptureMode::kPiecewise)
    g_piecewise_scopes.fetch_add(1, std::memory_order_relaxed);
  else
    g_full_scopes.fetch_add(1, std::memory_order_relaxed);
  t_current_scope = this;
  BeginSegment();
}

GraphCaptureScope::~GraphCaptureScope() {
  if (!active_) return;
  // `__exit__` closes the final segment inside a `finally` (`:322-333`), so the
  // scope pointer is restored even when the close throws. A destructor that
  // propagates would terminate, and a skipped close poisons the stream
  // permanently — which is why three drivers hand-rolled this same drain
  // (`qwen3_5.cpp:9913`, `qwen3_5.cpp:10335`, `qwen3_dflash.cpp:1106`).
  //
  // THE DRAIN IS ABOUT THE EXCEPTION, NOT ONLY ABOUT THE CLOSE. Catching a
  // throwing `EndCaptureGraph` covers one failure. The failure that actually
  // happens is a break function, or ordinary model code between two break
  // points, throwing MID-CAPTURE: the close then succeeds, the stream is clean,
  // and the container is left holding a PARTIAL forward that reports
  // `captured() == true` and replays as half a step. Comparing the uncaught
  // depth against the one recorded at entry is what tells the two apart, and the
  // partial container is destroyed rather than handed back.
  //
  // AND THE DRAIN RECORDS THAT IT DRAINED. Swallowing is mandatory here; being
  // SILENT about it is not. `Reset()` leaves the container reporting
  // `captured() == false`, which is also what an INERT scope reports — and the
  // caller's correct response to those two states is opposite (see
  // `BreakableGraph::capture_failed()`). The flag and the exception are stashed
  // AFTER the reset, because `Reset()` clears them.
  const bool unwinding = std::uncaught_exceptions() > uncaught_on_entry_;
  try {
    EndSegment();
  } catch (...) {
    g_->Reset();
    g_->capture_failed_ = true;
    // Inside a handler, so this IS the close's own exception. This is the arm a
    // caller can observe, because nothing else is propagating.
    g_->capture_error_ = std::current_exception();
    t_current_scope = nullptr;
    return;
  }
  if (unwinding) {
    g_->Reset();
    g_->capture_failed_ = true;
    // No `capture_error_` here, and not by omission: no handler has been entered
    // for the exception that is unwinding through this scope, so
    // `std::current_exception()` returns null. The real exception reaches the
    // caller on its own; see `BreakableGraph::capture_error()`.
  }
  t_current_scope = nullptr;
}

void GraphCaptureScope::BeginSegment() {
  if (!active_ || segment_open_) return;
  b_->BeginCapture(*q_);
  segment_open_ = true;
}

void GraphCaptureScope::EndSegment() {
  if (!active_ || !segment_open_) return;
  // D10, and it runs BEFORE the close rather than after it, because after is too
  // late: `cudaStreamEndCapture` on a capture graph that still has an unjoined
  // fork FAILS. Upstream does exactly this inside `_end_current_segment`
  // (`:353-361`), immediately before `capture_end()`.
  JoinOutstandingForks();
  segment_open_ = false;  // cleared FIRST: a throwing end must not be retried
  void* seg = b_->EndCaptureGraph(*q_);
  g_->segments_.push_back(seg);
  g_segments.fetch_add(1, std::memory_order_relaxed);
}

void GraphCaptureScope::AppendBreak(std::function<void()> fn, const void* destination) {
  if (!active_) return;
  // A `kFull` scope has ONE segment by definition, so a registered break would
  // leave `break_count() == segment_count()` and `Replay` would drop it. Every
  // `GraphBreak` form already takes the pass-through arm here (`splits()` is
  // false), and this refusal is what stops a form added later from opting out of
  // that — the same reason the aliasing check lives at this one registration
  // point rather than in each form.
  VT_CHECK(mode_ == GraphCaptureMode::kPiecewise,
           "GraphBreak: this capture scope is kFull (vLLM CUDAGraphMode.FULL), which "
           "captures the forward as ONE segment; a break point cannot be registered "
           "in it. Open the scope in kPiecewise to split");
  // The ALIASING half of lifetime rule 1, refused rather than documented. Two
  // break points writing through one CELL bind both replay closures to the same
  // address: the earlier writeback is overwritten on every replay, and any
  // segment that baked the earlier destination reads the later break's data.
  // Nothing faults and the token gate cannot see it, which is why this is a
  // refusal and not a comment. The check is here, at the ONE registration
  // point, so a `GraphBreak` form added later cannot opt out of it.
  if (destination != nullptr) {
    for (const void* d : destinations_) {
      VT_CHECK(d != destination,
               "GraphBreak: this destination is already the target of another break point "
               "in the open capture; both replays would write through the SAME address. "
               "Give each break point its own BreakSlot");
    }
    destinations_.push_back(destination);
  }
  g_->break_fns_.push_back(std::move(fn));
  g_breaks.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// D10 — the outstanding-fork set. Port of the `wait_stream` hook (`:101-153`)
// and of the auto-join in `_end_current_segment` (`:353-361`).
// ---------------------------------------------------------------------------

void GraphCaptureScope::NoteFork(Queue& aux, Event& join_event) {
  if (!active_) return;
  // Registering the SAME queue twice would make the auto-join issue two waits
  // for one fork, and — worse — would leave one entry behind after `NoteJoin`
  // retired the other, so a joined queue would still look outstanding. The
  // identity is the QUEUE, because that is what a join targets.
  for (ForkedQueue& f : forks_) {
    if (f.q == &aux) {
      f.e = &join_event;  // a re-fork of the same queue: keep ONE entry, newest event
      return;
    }
  }
  forks_.push_back(ForkedQueue{&aux, &join_event});
  g_forks_tracked.fetch_add(1, std::memory_order_relaxed);
}

void GraphCaptureScope::NoteJoin(Queue& aux) {
  if (!active_) return;
  for (size_t i = 0; i < forks_.size(); ++i) {
    if (forks_[i].q == &aux) {
      forks_.erase(forks_.begin() + static_cast<long>(i));
      return;
    }
  }
}

void GraphCaptureScope::JoinOutstandingForks() {
  if (!active_ || forks_.empty()) return;
  for (ForkedQueue& f : forks_) {
    // The join upstream spells `self.stream.wait_stream(s)` (`:359`): make the
    // CAPTURE queue wait for everything submitted to the side queue. Ours is the
    // explicit two-call form — record on the side queue, wait on the capture
    // queue — which is the same pair the model itself issues when it joins.
    b_->RecordEvent(*f.e, *f.q);
    b_->QueueWaitEvent(*q_, *f.e);
    g_forks_auto_joined.fetch_add(1, std::memory_order_relaxed);
  }
  forks_.clear();
}

void GraphNoteFork(Queue& aux, Event& join_event) {
  GraphCaptureScope* s = GraphCaptureScope::Current();
  if (s == nullptr) return;
  s->NoteFork(aux, join_event);
}

void GraphNoteJoin(Queue& aux) {
  GraphCaptureScope* s = GraphCaptureScope::Current();
  if (s == nullptr) return;
  s->NoteJoin(aux);
}

// The bare marker (`:370-374`).
void GraphBreak() {
  GraphCaptureScope* s = GraphCaptureScope::Current();
  if (s == nullptr || !s->splits()) {
    detail::CountBreakPoint();
    return;
  }
  s->EndSegment();
  s->AppendBreak([] {}, nullptr);
  s->BeginSegment();
  detail::CountBreakPoint();
}

}  // namespace vt
