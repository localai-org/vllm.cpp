ID: ISSUE-LOCAL-01M7JYHQJEANK9Z6S0SJWKXWBC
Title: TT batched-decode RAC device lane: garbage at the second slot's second KV head under rich suite state (deterministic per binary layout)
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-16
Updated: 2026-11-16
Closed: -

## Problem

The W22-consume full-suite run at the merged head (e41f9fbbb: TT-PA-SMALL-WINDOW's fix merged into row/QWEN38-DFLASH2-STAGE-C) exposed a latent RAC device-lane defect: 'kTENSTORRENT batched decode RAC is capture-safe (num_slots=2)' fails its EAGER leg deterministically in the suite context (2/2 suite runs) — 'EAGER K bad u=1 h=1 e=8 got=4.45658e+28 want=0.00848389', 12/128 elements garbage, ALL of them at the second slot's second KV head — while the same case passes in isolation at both heads and passes in the suite at the pre-merge head 67d041d76 (2/2 suite runs). The garbage bits are recycled memory (the TT-PA-SMALL-WINDOW characterization's signature class), so this is a read-before-land/race or a misaddress in the RAC device lane under the host-free env (VT_TT_HOST_FREE_DECODE=1), whose trigger state shifted when the PA commit path's typecast program changed the allocator/JIT layout. NOT a regression of the PA fix (the fix cleared the W22 harness pa_bad 5/40 -> 0/40, the W22 discriminator, and the pure-decode PA case); it is a latent RAC defect newly exposed. Owed: characterize the trigger, root-cause the RAC device lane's second-slot second-head write, fix.

## Resolution

-
