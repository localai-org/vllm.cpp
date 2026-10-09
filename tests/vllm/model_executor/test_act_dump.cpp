// The activation-dump writer must REFUSE rather than write nothing (#2590).
//
// WHAT IT GUARDS. Three activation dumps in `qwen3_5.cpp` shared one shape:
//
//     std::FILE* f = std::fopen(path.c_str(), "wb");
//     if (f != nullptr) { std::fwrite(...); std::fclose(f); }
//
// A dump into a directory that does not exist therefore produced no file, no
// message and no non-zero exit. That is byte-for-byte the same experience as a
// dump that was never switched on, and `.agents/verification.md` names the
// class: an instrument whose failure looks like a result. The sibling
// logit-dump row (#2534) spent a GPU lease on exactly this, and its own log
// recorded the outcome as `ALIGNMENT=BROKEN checked=0 bad=6` — six missing
// sidecars from a writer that had silently done nothing.
//
// So the cases below are about the REFUSALS, not about the happy path. Each one
// deletes a specific way of being silent:
//
//   * an unwritable directory must throw, naming the knob and the path;
//   * a manifest that cannot be appended must throw, because a blob nobody can
//     describe is not evidence;
//   * a step that wrote fewer blobs than were due must throw, because "zero
//     blobs" is the signature the instrument exists to make impossible;
//   * `LayerScope` must restore the previous key, because a nested dump that
//     leaked its key would shift every later file by one layer and the join
//     would still look clean.
//
// WHAT IT CANNOT GUARD. Nothing here reads a device. The wrappers that download
// a tensor (`ActDumpTensor` / `ActDumpStream`) live in `qwen3_5.cpp` beside the
// forward, and only a real model run exercises those. This suite pins the
// writer's contract — the manifest columns the comparator parses, and the
// refusals — which is the part that was wrong.
#include <doctest/doctest.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "vllm/model_executor/models/act_dump.h"

namespace {

std::string MakeTempDir() {
  char tmpl[] = "/tmp/vllm_actdump_XXXXXX";
  const char* d = ::mkdtemp(tmpl);
  REQUIRE(d != nullptr);
  return std::string(d);
}

std::vector<std::string> ReadLines(const std::string& path) {
  std::ifstream in(path);
  std::vector<std::string> out;
  std::string line;
  while (std::getline(in, line)) out.push_back(line);
  return out;
}

std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string part;
  while (std::getline(ss, part, sep)) out.push_back(part);
  return out;
}

}  // namespace

TEST_CASE("actdump: selected ordinal preserves refusal and restores nested selection") {
  using namespace vllm::actdump;
  CHECK(StepSelected(0));
  CHECK_FALSE(StepSelected(-1));
  {
    const StepSelectionScope selected(29);
    CHECK_FALSE(StepSelected(28));
    CHECK(StepSelected(29));
    CHECK_FALSE(StepSelected(30));
    g_blobs_step.store(0);
    g_stream_blobs_step.store(0);
    CHECK_NOTHROW(EndStep(28, 130, 1));
    CHECK_THROWS_AS(EndStep(29, 130, 1), std::runtime_error);
    {
      const StepSelectionScope nested(11);
      CHECK(StepSelected(11));
      CHECK_FALSE(StepSelected(29));
    }
    CHECK(StepSelected(29));
    CHECK_THROWS_AS(StepSelectionScope(-2), std::runtime_error);
    CHECK(SelectedStep() == 29);
  }
  CHECK(SelectedStep() == -1);
  CHECK(StepSelected(30));
}

TEST_CASE("actdump: substage layer selection restores and does not filter stream ordinals") {
  using namespace vllm::actdump;
  CHECK(StageLayerSelected(0));
  {
    const StageLayerSelectionScope selected(0);
    CHECK(StageLayerSelected(0));
    CHECK_FALSE(StageLayerSelected(1));
    CHECK_FALSE(StageLayerSelected(-1));
    CHECK(StepSelected(1));
    {
      const StageLayerSelectionScope nested(1);
      CHECK(StageLayerSelected(1));
      CHECK_FALSE(StageLayerSelected(0));
    }
    CHECK(StageLayerSelected(0));
    CHECK_THROWS_AS(StageLayerSelectionScope(-2), std::runtime_error);
    CHECK(SelectedStageLayer() == 0);
  }
  CHECK(SelectedStageLayer() == -1);
  CHECK(StageLayerSelected(1));
}

TEST_CASE("actdump: a blob and its manifest row describe the same thing") {
  const std::string dir = MakeTempDir();
  const std::vector<uint16_t> payload = {0x3F80, 0x4000, 0xC000, 0x0000,
                                         0x1234, 0x5678};
  vllm::actdump::WriteBlob("VT_DUMP_ACT", dir, /*step=*/7, /*layer=*/-1,
                           "hidden", vt::DType::kBF16, /*rows=*/2, /*cols=*/3,
                           payload.data(), payload.size() * sizeof(uint16_t));

  // The blob is named by (step, layer, stage) and holds exactly what was passed.
  const std::string blob = dir + "/s7_l-1_hidden.bin";
  std::ifstream in(blob, std::ios::binary);
  REQUIRE(in.good());
  std::vector<char> got((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  CHECK(got.size() == payload.size() * sizeof(uint16_t));
  CHECK(std::memcmp(got.data(), payload.data(), got.size()) == 0);

  // The manifest carries the eight columns the comparator joins and checks on.
  // They are asserted individually rather than as one string, so a column that
  // moves names itself in the failure.
  const std::vector<std::string> rows = ReadLines(dir + "/manifest.tsv");
  REQUIRE(rows.size() == 1);
  const std::vector<std::string> col = Split(rows[0], '\t');
  REQUIRE(col.size() == 8);
  CHECK(col[0] == "7");                 // step
  CHECK(col[1] == "-1");                // layer: the post-embedding snapshot
  CHECK(col[2] == "hidden");            // stage
  CHECK(col[3] == std::string(vt::Name(vt::DType::kBF16)));
  CHECK(col[4] == "2");                 // rows
  CHECK(col[5] == "3");                 // cols
  CHECK(col[6] == "12");                // bytes
  CHECK(col[7] == "s7_l-1_hidden.bin");  // file

  std::remove(blob.c_str());
  std::remove((dir + "/manifest.tsv").c_str());
  ::rmdir(dir.c_str());
}

TEST_CASE("actdump: an unwritable directory REFUSES instead of writing nothing") {
  const std::string missing = "/tmp/vllm_actdump_does_not_exist_2590/sub";
  const uint16_t payload[2] = {1, 2};

  // The predecessor writers returned silently here. Anything that is not a
  // throw is the defect this case exists for.
  CHECK_THROWS_AS(vllm::actdump::WriteBlob("VT_DUMP_ACT", missing, 0, 0,
                                           "hidden", vt::DType::kBF16, 1, 2,
                                           payload, sizeof(payload)),
                  std::runtime_error);

  // The message must name the knob and the path, because the operator who set
  // the variable is the only person who can fix a bad directory.
  bool named_the_knob = false;
  bool named_the_path = false;
  try {
    vllm::actdump::WriteBlob("VT_DUMP_ACT", missing, 0, 0, "hidden",
                             vt::DType::kBF16, 1, 2, payload, sizeof(payload));
  } catch (const std::runtime_error& e) {
    const std::string what = e.what();
    named_the_knob = what.find("VT_DUMP_ACT") != std::string::npos;
    named_the_path = what.find(missing) != std::string::npos;
  }
  CHECK(named_the_knob);
  CHECK(named_the_path);
}

TEST_CASE("actdump: a manifest that cannot be appended REFUSES") {
  const std::string missing = "/tmp/vllm_actdump_no_manifest_dir_2590";
  CHECK_THROWS_AS(vllm::actdump::FileManifestRow("VT_DUMP_ACT_SUB", missing,
                                                 "s0_l0_block_out.bin", 0, 0,
                                                 "block_out", vt::DType::kBF16,
                                                 1, 4, 8),
                  std::runtime_error);
}

TEST_CASE("actdump: a step that wrote fewer STREAM blobs than were due REFUSES") {
  const std::string dir = MakeTempDir();
  const uint16_t payload[2] = {1, 2};

  // Two stream blobs written, two due: accepted.
  vllm::actdump::g_blobs_step.store(0);
  vllm::actdump::g_stream_blobs_step.store(0);
  vllm::actdump::WriteBlob("VT_DUMP_ACT", dir, 0, 0, "hidden",
                           vt::DType::kBF16, 1, 2, payload, sizeof(payload));
  vllm::actdump::WriteBlob("VT_DUMP_ACT", dir, 0, 0, "res", vt::DType::kBF16, 1,
                           2, payload, sizeof(payload));
  vllm::actdump::g_stream_blobs_step.fetch_add(2);
  CHECK_NOTHROW(vllm::actdump::EndStep(0, 2, 1));

  // Two written, three due: refused. This is the guard that turns "the
  // instrument produced nothing" from a clean-looking profile into a failure.
  CHECK_THROWS_AS(vllm::actdump::EndStep(0, 3, 1), std::runtime_error);

  // THE CASE A TOTAL CANNOT SEE, and the reason the counter is split. Plenty of
  // blobs were written this step -- the GDN stage probes fire off the same
  // environment variable -- but NOT ONE of them is a residual-stream blob. A
  // floor keyed on the total passes here; the reachability mutation that
  // deleted the stream call sites produced exactly this state and a
  // total-keyed gate reported success over it.
  vllm::actdump::g_blobs_step.store(50);
  vllm::actdump::g_stream_blobs_step.store(0);
  CHECK_THROWS_AS(vllm::actdump::EndStep(0, 2, 1), std::runtime_error);

  // Nothing written at all is the case that actually happened on #2534.
  vllm::actdump::g_blobs_step.store(0);
  vllm::actdump::g_stream_blobs_step.store(0);
  CHECK_THROWS_AS(vllm::actdump::EndStep(0, 1, 1), std::runtime_error);

  // The sub-stage knob alone on a path that HAS sub-stage probes: no stream blob
  // is due, but a step that wrote nothing at all is still a refusal.
  CHECK_THROWS_AS(vllm::actdump::EndStep(0, 0, 1), std::runtime_error);
  vllm::actdump::g_blobs_step.store(7);
  CHECK_NOTHROW(vllm::actdump::EndStep(0, 0, 1));

  // ... and on a path that carries none, zero blobs is CORRECT and must not
  // refuse. The MoE loop is that path.
  vllm::actdump::g_blobs_step.store(0);
  CHECK_NOTHROW(vllm::actdump::EndStep(0, 0, 0));

  // A step of -1 is the instrument switched OFF, and must stay silent.
  CHECK_NOTHROW(vllm::actdump::EndStep(-1, 1000, 1000));

  std::remove((dir + "/s0_l0_hidden.bin").c_str());
  std::remove((dir + "/s0_l0_res.bin").c_str());
  std::remove((dir + "/manifest.tsv").c_str());
  ::rmdir(dir.c_str());
}

TEST_CASE("actdump: LayerScope restores the key it replaced") {
  vllm::actdump::Current() = vllm::actdump::Where{3, 11};
  {
    const vllm::actdump::LayerScope inner(3, 12);
    CHECK(vllm::actdump::Current().step == 3);
    CHECK(vllm::actdump::Current().layer == 12);
    {
      const vllm::actdump::LayerScope nested(3, 13);
      CHECK(vllm::actdump::Current().layer == 13);
    }
    // A nested dump that leaked its key would shift every later blob by a
    // layer, and the comparator's join would still look complete.
    CHECK(vllm::actdump::Current().layer == 12);
  }
  CHECK(vllm::actdump::Current().step == 3);
  CHECK(vllm::actdump::Current().layer == 11);
}
