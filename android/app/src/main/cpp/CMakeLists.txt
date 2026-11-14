cmake_minimum_required(VERSION 3.22)
project(superscale C)

set(ORT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so")

add_library(superscale SHARED
  engine.c
  err.c
  image.c
  cpu.c
  gpu/gpu.c
  gpu/onnx.c
  gpu/plan.c
)

target_include_directories(superscale PRIVATE
  onnxruntime/include
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/gpu"
)

target_link_libraries(superscale "${ORT_LIB}" vulkan)

target_compile_options(superscale PRIVATE -O3 -Wall -Wextra -Wno-missing-field-initializers)

set_target_properties(superscale PROPERTIES BUILD_RPATH "$ORIGIN")
