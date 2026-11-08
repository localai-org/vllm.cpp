// Compat shim for the tt-metal pin forward (tt-metal origin/main 2500c1d8f63 + SFPI 7.89.0).
//
// The updated tt-metal stack moved the tensor spec headers under the
// `experimental` include subtree:
//   tt-metalium/tensor/spec/... -> tt-metalium/tensor/spec/...
// Everything else this backend includes stayed put. Include this header
// instead of the four spec paths directly so the next move is a one-file fix.
#pragma once

#include <tt-metalium/tensor/spec/memory_config/memory_config.hpp>
#include <tt-metalium/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/tensor/spec/layout/tensor_layout.hpp>
#include <tt-metalium/tensor/spec/layout/page_config.hpp>
