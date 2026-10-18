// Compat shim for the tt-metal pin forward (ISSUE-LOCAL-01M4GPEZ987KVAXCYBFB2EVNT7).
//
// The updated tt-metal stack moved the tensor spec headers under the
// `experimental` include subtree:
//   tt-metalium/tensor/spec/... -> tt-metalium/experimental/tensor/spec/...
// Everything else this backend includes stayed put. Include this header
// instead of the four spec paths directly so the next move is a one-file fix.
#pragma once

#include <tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/page_config.hpp>
