ID: ISSUE-LOCAL-01M4GPEZ987KVAXCYBFB2EVNT7
Title: Port the TT backend to the updated tt-metal stack (tt_umd 0.9.12/KMD 2.11.1/FW 19.15.0)
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: -

## Problem

Host Tenstorrent stack updated (tt-smi 6.7.0, tt-umd 0.9.12, KMD 2.11.1-pre, board firmware 19.7.1 -> 19.15.0). The frozen tt-metal pin (~/Sources/tt/tt-metal-pin, tt_umd v0.9.11-427) can no longer run on the P150: its tt_umd reports num_hw_cqs=0 against the new KMD/firmware, so every vllm TT binary fails at first op with TT_FATAL mesh_device.cpp:874 cq_id 0 out of range. A/B-proven: the same board through tt_umd 0.9.12 (probe /tmp/umd-probe, trial build /tmp/tt-metal-umdtrial + /tmp/umdtrial-install) reports num_hw_cqs=1 and CQ0 creates fine. Root cause and trial recipe: memory note p150-ttumd-rootcause-confirmed.md (2026-10-09). Developer decision: option 3, full pin forward. Work owed: port src/vt/tenstorrent/ + include/vllm/ to the new tt-metal headers/APIs (first break: tt-metalium/tensor/spec/memory_config/memory_config.hpp moved to include/tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp), link+load against /tmp/umdtrial-install, then rerun device gates: test_tenstorrent_backend embedding exactness, test_kolibri1 27/234, test_kolibri1_tt_b2i staging, decode bench anchor 109726, and the B2b-ii full-model token gate (141/145 argmax positions, 4 near-tie flips inside 2.5-nat band, 0 hard flips). Stop conditions: if the new tt_umd rejects FW 19.15.0, or token-gate numerics move beyond the band, STOP and report.

## Resolution

-
