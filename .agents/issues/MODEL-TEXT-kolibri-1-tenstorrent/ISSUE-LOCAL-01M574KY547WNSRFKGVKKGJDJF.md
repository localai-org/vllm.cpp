ID: ISSUE-LOCAL-01M574KY547WNSRFKGVKKGJDJF
Title: kolibri1 TT routed tier: same-step slot-miss staging is unreached after the LRU pre-touch fix
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-18
Updated: 2026-10-18
Closed: 2026-10-18

## Problem

multi-token forwards select more experts than the slot pool holds; a same-step eviction could leave a staged expert unreadable

## Resolution

2026-10-18: the routed-tier compute loop (kolibri1_tt_forward.cpp) repairs a same-step SlotFor miss on demand through the same stage_expert body instead of refusing; committed on row/tt-kolibri-p150-three (PR #3475). The fallback is defensive on the current gate: after the LRU pre-touch fix the b2ii device token gate result is identical with the fallback deleted, so no measured walk reaches the overflow; the condition is deterministic and pinned red-first by the device-free test 'kolibri1 TT slot policy: a same-step distinct-set overflow leaves a same-step staged expert evicted' in tests/vllm/models/test_kolibri1_tt.cpp (mutation-checked: flipping LRU to MRU eviction turns it red at CHECK(policy.SlotFor(0, 7) < 0)).
