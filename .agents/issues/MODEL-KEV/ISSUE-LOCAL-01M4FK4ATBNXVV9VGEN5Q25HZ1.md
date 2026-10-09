ID: ISSUE-LOCAL-01M4FK4ATBNXVV9VGEN5Q25HZ1
Title: ASan heap-buffer-overflow in kev PointerHeadForward: no shape validation, wrong-scale perturbation test reads past the weight vectors
Row: MODEL-KEV
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: -

## Problem

The sanitize-cpu lane is red on test_kev (main 41705a7a2). CI ASAN+UBSAN leg: AddressSanitizer heap-buffer-overflow READ at src/vllm/model_executor/models/kev_head.cpp:43 in PointerHeadForward (test case kev.PointerHead.perturbation.wrong_scale_detected, tests/vllm/models/test_kev.cpp:218). Local sanitizer build (assertions-enabled libstdc++): the same defect aborts earlier as a std::vector<float>::operator[] assertion '__n < this->size()' (SIGABRT, rc=134); log /tmp/sanitize-runs/test_kev.log. Root cause: the perturbation test simulates a wrong-scale implementation with `wrong.head_dim = params.hidden_size` (test_kev.cpp:235), i.e. dp=1024 while the golden HeadWeights carry only dp=256 rows (q_weight/k_weight are [256,1024], q_bias/k_bias are [256]). PointerHeadForward trusts params.head_dim for the loop bounds and indexes weights.q_weight[j*d] / weights.q_bias[j] for j up to dp-1 with NO shape validation, so j>=256 reads past the end of the weight vectors: the unchecked `w[i]` at kev_head.cpp:43 is the CI heap-buffer-overflow and the checked vector operator[] at line 39/46 is the local SIGABRT. This is a real product memory bug, not only a test artifact: the loader builds HeadParams from config (kev_registry.cpp:148-153, kev_head_dim) and HeadWeights from head.safetensors independently, so a config/checkpoint mismatch reaches the same out-of-bounds read in production. Fix (two parts): (1) PointerHeadForward validates that q_weight/k_weight are [head_dim, hidden_size], q_bias/k_bias are [head_dim], h_decide has >= hidden_size and h_opts has >= n_options*hidden_size elements, and throws std::invalid_argument naming the mismatch instead of reading out of bounds; (2) the wrong-scale perturbation keeps its documented meaning ("a mutation that still runs but produces wrong logits") by expressing the wrong scale without changing the projection shape: HeadParams gains a default-inert optional explicit scale (scale() = 1/sqrt(head_dim) when unset) and the test sets it to 1/sqrt(hidden_size), so the forward runs in-bounds and produces logits off by exactly the constant factor sqrt(dp/d). A new targeted case pins the validation (mismatched weights throw).

## Resolution

-
