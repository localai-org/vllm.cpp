cmake_minimum_required(VERSION 3.24)
project(exl3_projection_replay LANGUAGES CXX)
# Deliberately build only the replay and its actual native dependencies. This
# does not rebuild the full engine or require Torch, oneDNN, or SYCL-TLA.
get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(exl3_projection_replay replay_projection.cpp
  ${REPO}/src/vt/backend.cpp ${REPO}/src/vt/dtype.cpp
  ${REPO}/src/vt/tensor.cpp ${REPO}/src/vt/op_provider.cpp ${REPO}/src/vt/copy.cpp
  ${REPO}/src/vt/xpu/xpu_backend.cpp ${REPO}/src/vt/xpu/xpu_elementwise.cpp
  ${REPO}/src/vt/xpu/xpu_exl3.cpp ${REPO}/src/vt/xpu/xpu_exl3_prefill.cpp
  ${REPO}/src/vllm/model_executor/model_loader/safetensors_reader.cpp
  ${REPO}/src/vllm/model_executor/model_loader/read_only_file_mapping.cpp)
target_compile_features(exl3_projection_replay PRIVATE cxx_std_20)
target_include_directories(exl3_projection_replay PRIVATE ${REPO}/include ${REPO}/src)
target_include_directories(exl3_projection_replay SYSTEM PRIVATE ${REPO}/third_party)
target_compile_options(exl3_projection_replay PRIVATE -O3 -Wall -Wextra -Werror
  -fno-fast-math -ffp-contract=off -fsycl -foffload-fp32-prec-div -foffload-fp32-prec-sqrt)
target_link_options(exl3_projection_replay PRIVATE -fsycl)

# Host-only check of the actual shared reader, retaining its production map
# ownership implementation. Dead-section elimination avoids linking unrelated
# model forwards from the weight-owner translation unit into this diagnostic.
add_executable(exl3_loader_check check_loader.cpp
  ${REPO}/src/vt/dtype.cpp
  ${REPO}/src/vllm/model_executor/models/qwen3_5_weights.cpp
  ${REPO}/src/vllm/model_executor/model_loader/safetensors_reader.cpp
  ${REPO}/src/vllm/model_executor/model_loader/read_only_file_mapping.cpp)
target_compile_features(exl3_loader_check PRIVATE cxx_std_20)
target_include_directories(exl3_loader_check PRIVATE ${REPO}/include ${REPO}/src)
target_include_directories(exl3_loader_check SYSTEM PRIVATE ${REPO}/third_party)
target_compile_options(exl3_loader_check PRIVATE -O2 -Wall -Wextra -Werror
  -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
target_link_options(exl3_loader_check PRIVATE -Wl,--gc-sections)

add_executable(exl3_mul1_marker_test
  ${REPO}/tests/doctest_main.cpp
  ${REPO}/tests/vllm/model_executor/layers/test_exl3_mul1_marker.cpp
  ${REPO}/src/vt/dtype.cpp
  ${REPO}/src/vllm/model_executor/models/qwen3_5_weights.cpp
  ${REPO}/src/vllm/model_executor/model_loader/safetensors_reader.cpp
  ${REPO}/src/vllm/model_executor/model_loader/read_only_file_mapping.cpp)
target_compile_features(exl3_mul1_marker_test PRIVATE cxx_std_20)
target_include_directories(exl3_mul1_marker_test PRIVATE ${REPO}/include ${REPO}/src)
target_include_directories(exl3_mul1_marker_test SYSTEM PRIVATE ${REPO}/third_party)
target_compile_options(exl3_mul1_marker_test PRIVATE -O2 -Wall -Wextra -Werror
  -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
target_link_options(exl3_mul1_marker_test PRIVATE -Wl,--gc-sections)

add_executable(dense_f16_loader_test
  ${REPO}/tests/doctest_main.cpp
  ${REPO}/tests/vllm/model_executor/layers/test_dense_f16_loader.cpp
  ${REPO}/src/vt/dtype.cpp
  ${REPO}/src/vllm/model_executor/models/qwen3_5_weights.cpp
  ${REPO}/src/vllm/model_executor/model_loader/safetensors_reader.cpp
  ${REPO}/src/vllm/model_executor/model_loader/read_only_file_mapping.cpp)
target_compile_features(dense_f16_loader_test PRIVATE cxx_std_20)
target_include_directories(dense_f16_loader_test PRIVATE ${REPO}/include ${REPO}/src)
target_include_directories(dense_f16_loader_test SYSTEM PRIVATE ${REPO}/third_party)
target_compile_options(dense_f16_loader_test PRIVATE -O2 -Wall -Wextra -Werror
  -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
target_link_options(dense_f16_loader_test PRIVATE -Wl,--gc-sections)

add_executable(qwen_exl3_precision_test
  ${REPO}/tests/doctest_main.cpp
  ${REPO}/tests/vllm/models/test_qwen_exl3_precision.cpp
  ${REPO}/src/vllm/model_executor/models/qwen3_5_dense_weights.cpp)
target_compile_features(qwen_exl3_precision_test PRIVATE cxx_std_20)
target_include_directories(qwen_exl3_precision_test PRIVATE ${REPO}/include ${REPO}/src)
target_include_directories(qwen_exl3_precision_test SYSTEM PRIVATE ${REPO}/third_party)
target_compile_options(qwen_exl3_precision_test PRIVATE -O2 -Wall -Wextra -Werror
  -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
target_link_options(qwen_exl3_precision_test PRIVATE -Wl,--gc-sections)

add_executable(qwen_exl3_parameter_check check_qwen_parameters.cpp
  ${REPO}/src/vt/dtype.cpp
  ${REPO}/src/vllm/model_executor/layers/quantization/fp8_block_quant.cpp
  ${REPO}/src/vllm/model_executor/models/qwen3_5_dense_weights.cpp
  ${REPO}/src/vllm/model_executor/models/qwen3_5_weights.cpp
  ${REPO}/src/vllm/model_executor/model_loader/nvfp4_dequant.cpp
  ${REPO}/src/vllm/model_executor/model_loader/safetensors_reader.cpp
  ${REPO}/src/vllm/model_executor/model_loader/read_only_file_mapping.cpp)
target_compile_features(qwen_exl3_parameter_check PRIVATE cxx_std_20)
target_include_directories(qwen_exl3_parameter_check PRIVATE ${REPO}/include ${REPO}/src)
target_include_directories(qwen_exl3_parameter_check SYSTEM PRIVATE ${REPO}/third_party)
target_compile_options(qwen_exl3_parameter_check PRIVATE -O2 -Wall -Wextra -Werror
  -fno-fast-math -ffp-contract=off -ffunction-sections -fdata-sections)
target_link_options(qwen_exl3_parameter_check PRIVATE -Wl,--gc-sections)
