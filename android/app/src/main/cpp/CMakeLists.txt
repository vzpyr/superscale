cmake_minimum_required(VERSION 3.22)
project(superscale_ort C)

set(ORT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so")

add_library(superscale_ort SHARED superscale_ort.c)
target_include_directories(superscale_ort PRIVATE onnxruntime/include)
target_link_libraries(superscale_ort "${ORT_LIB}")
target_compile_options(superscale_ort PRIVATE -O2)
set_target_properties(superscale_ort PROPERTIES BUILD_RPATH "$ORIGIN")
