cmake_minimum_required(VERSION 3.22.1)
project(mlc_llm_jni)

# Path to pre-built MLC-LLM and TVM Android libraries
set(MLC_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

add_library(tvm_runtime SHARED IMPORTED)
set_target_properties(tvm_runtime PROPERTIES
    IMPORTED_LOCATION "${MLC_LIB_DIR}/libtvm_runtime.so"
)

add_library(mlc_llm SHARED IMPORTED)
set_target_properties(mlc_llm PROPERTIES
    IMPORTED_LOCATION "${MLC_LIB_DIR}/libmlc_llm.so"
)

add_library(mlc_jni SHARED mlc_jni.cpp)

target_include_directories(mlc_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mlc_jni
    mlc_llm
    tvm_runtime
    android
    log
)
