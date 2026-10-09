/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::fmha::kernel {

struct XeFHMAIndividualTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFHMAIndividualTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape) {
    using namespace cute;

    dim3 grid(
        size(ceil_div(shape.head_size_vo, get<1>(tile_shape))),  // V
        size(ceil_div(shape.seq_len_qo, get<0>(tile_shape))),    // Q
        size(shape.batch * shape.num_heads_q));  // (h,b) -- split later
    return Params{grid, {shape.num_heads_q}};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() { return valid_; }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    int idx_b = BlockIdxZ();
    int head;
    params.divmod_num_heads(idx_b, head, idx_b);
    return make_coord(BlockIdxY(), BlockIdxX(), head, idx_b);
  }

  CUTLASS_DEVICE
  XeFHMAIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

// Work item for compact grid dispatch: each WG reads one entry
struct DecodeWorkItem {
  int seq_idx;        // which sequence (batch index)
  int kv_tile_start;  // starting KV tile index (absolute, not per-split)
  int kv_tile_count;  // number of KV tiles this WG processes
  int split_idx;      // which split within this seq (for Oaccum indexing)
};

struct DecodeTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
    FastDivmod divmod_batch;
    int num_kv_splits_ = -1;
    // Compact grid mode: work_list[total_wgs] lookup table
    const DecodeWorkItem* work_list = nullptr;
    int total_wgs = 0;  // = sum(splits_per_seq) per head
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  DecodeTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape,
      const int& num_kv_splits = -1,
      const DecodeWorkItem* work_list = nullptr,
      int total_wgs = 0) {
    using namespace cute;

    // Decode packs the GQA head-group (head_group_q = num_heads_q /
    // num_heads_kv) into the Q/row dimension of the MMA tile (seq_len_qo is
    // always 1 for decode). When head_group_q exceeds the policy's packed-Q
    // tile size (get<0>(tile_shape), e.g. 8 or 16), we tile the head-group
    // across the grid's Q dimension so an arbitrarily large GQA ratio (e.g.
    // falcon-7b: 71 query heads / 1 KV head) is processed by ceil(ratio /
    // q_packed) work-groups instead of being capped at q_packed.
    int head_group_q = shape.num_heads_q / shape.num_heads_kv;
    dim3 grid(
        size(ceil_div(shape.head_size_vo, get<1>(tile_shape))),  // V
        size(ceil_div(head_group_q, get<0>(tile_shape))),        // Q
        size(shape.batch * shape.num_heads_q));  // (h,b) -- split later
    int num_head = shape.num_heads_q;
    if (num_kv_splits >= 1) {
      num_head = shape.num_heads_kv;
      if (work_list != nullptr && total_wgs > 0) {
        // Compact grid: exactly total_wgs × num_heads_kv WGs
        grid.z = total_wgs * shape.num_heads_kv;
      } else {
        // Original: batch × heads_kv × num_kv_splits
        grid.z = size(shape.batch * shape.num_heads_kv);
        grid.z *= num_kv_splits;
      }
    }
    return Params{
        grid,
        {num_head},
        {shape.batch * num_head},
        num_kv_splits,
        work_list,
        total_wgs};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() { return valid_; }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;
    int flat_idx = BlockIdxZ();
    int head, idx_b;
    int idx_kv_split;

    if (params.work_list != nullptr && params.total_wgs > 0) {
      // Compact grid: direct lookup, O(1)
      // flat_idx = head_kv * total_wgs + work_idx
      int work_idx = flat_idx % params.total_wgs;
      head = flat_idx / params.total_wgs;
      // Read pre-computed work assignment — O(1) direct lookup
      DecodeWorkItem wi = params.work_list[work_idx];
      idx_b = wi.seq_idx;
      idx_kv_split = wi.split_idx;
      return make_coord(
          BlockIdxY(),
          BlockIdxX(),
          head,
          idx_b,
          idx_kv_split,
          wi.kv_tile_start,
          wi.kv_tile_count);
    }

    if (params.num_kv_splits_ >= 1) {
      idx_kv_split = flat_idx;
      params.divmod_batch(idx_kv_split, idx_b, idx_kv_split);
      params.divmod_num_heads(idx_b, head, idx_b);
      return make_coord(
          BlockIdxY(),
          BlockIdxX(),
          head,
          idx_b,
          idx_kv_split,
          (int)-1,
          (int)-1);
    }

    idx_b = flat_idx;
    params.divmod_num_heads(idx_b, head, idx_b);
    return make_coord(
        BlockIdxY(), BlockIdxX(), head, idx_b, (int)-1, (int)-1, (int)-1);
  }

  CUTLASS_DEVICE
  DecodeTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

struct XeReduceSplitKTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
    int num_kv_splits = -1;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeReduceSplitKTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape,
      const int& num_kv_splits = -1) {
    using namespace cute;

    dim3 grid(shape.seq_len_qo, shape.num_heads_q, shape.batch);
    return Params{grid, {shape.num_heads_q}, num_kv_splits};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() { return valid_; }

  CUTLASS_DEVICE
  auto get_block_coord() {
    using namespace cute;

    return make_coord(BlockIdxX(), BlockIdxY(), BlockIdxZ());
  }

  CUTLASS_DEVICE
  XeReduceSplitKTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

}  // namespace cutlass::fmha::kernel
