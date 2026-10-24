ID: ISSUE-LOCAL-01M5Q1PS772CPTMYJ02AMVZ3Y0
Title: FlintModel::Forward overflows attn_out when hq*head_dim > hidden_size
Row: MODEL-TEXT-FLINT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-24
Updated: 2026-10-24
Closed: -

## Problem

The W1 CPU reference forward (commit ea5c3dade) sizes the attention output buffer attn_out at t*hidden_size but writes it with the head layout [(i*hq+hh)*head_dim + d], which spans t*num_attention_heads*head_dim. On the published RecursiveSI container hq*head_dim = 32*128 = 4096 > hidden_size 2048, so every forward writes two columns past the buffer — the heap corruption the GGUF token gate tripped as malloc(): invalid size (unsorted). The W1 synthetic gate passed because its tiny config has hq*head_dim (8) < hidden_size, masking the defect. Found at the flint GGUF wave-2 token gate via an AddressSanitizer heap-buffer-overflow trace (flint.cpp:427 writing past the buffer allocated at flint.cpp:401). Fix: size attn_out at t*hq*head_dim, which is what o_proj ([q, h]) consumes.

## Resolution

-
