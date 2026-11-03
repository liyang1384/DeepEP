# Copyright (c) 2026, Lu Lu

cmake_minimum_required(VERSION 3.20)
project(deepep_runtime LANGUAGES CXX)
if(CMAKE_CONFIGURATION_TYPES)
    message(FATAL_ERROR "Use a single-configuration generator such as Ninja or Unix Makefiles")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()
file(WRITE "${CMAKE_BINARY_DIR}/compiler-info.json"
    "{\"id\":\"${CMAKE_CXX_COMPILER_ID}\",\"version\":\"${CMAKE_CXX_COMPILER_VERSION}\"}\n")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(DEEPEP_BUILD_EP_GMM_FUSED "Build Ascend950 communication/GMM experiment" OFF)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
include(cmake/Dependencies.cmake)
option(DEEPEP_LL_STAGE_TIMER "Compile optional LL device stage tracing" OFF)
option(DEEPEP_LL_REFERENCE_PROFILE "Compile LL dispatch reference cycle profiling" OFF)
option(DEEPEP_LL_COUNT_HISTOGRAM "Experimental remote count hardware histogram" OFF)
option(DEEPEP_LL_JIT "Compile the LL dispatch kernel at runtime through DeepJIT" OFF)
if(DEEPEP_LL_STAGE_TIMER)
    message(FATAL_ERROR "Legacy LL stage/expert tracing was removed; use DEEPEP_LL_REFERENCE_PROFILE=ON")
endif()
option(DEEPEP_BUILD_MEGAMOE "Build the optional MegaMoE Beta" OFF)
option(DEEPEP_MEGAMOE_TIMER "Compile MegaMoE device tracing" OFF)
include(cmake/Device.cmake)
pybind11_add_module(_C csrc/bindings/module.cpp csrc/runtime/session.cpp csrc/ops/low_latency.cpp
    csrc/ops/ll_capabilities.cpp)
find_path(DEEPEP_PLATFORM_INCLUDE tiling/platform/platform_ascendc.h
    HINTS "${ASCEND_HOME_PATH}/asc/include" "${ASCEND_HOME_PATH}/include/op_common"
          "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include" NO_DEFAULT_PATH REQUIRED)
target_link_directories(_C PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(_C PRIVATE platform tiling_api unified_dlog)
target_include_directories(_C PRIVATE "${DEEPEP_PLATFORM_INCLUDE}" csrc "${ASCEND_HOME_PATH}/include" "${SHMEM_ROOT}/include")
target_link_libraries(_C PRIVATE ascendcl shmem deepep_device ${CMAKE_DL_LIBS})
target_compile_definitions(_C PRIVATE DEEPEP_LL_RDMA=1)
if(DEEPEP_LL_COUNT_HISTOGRAM)
    target_compile_definitions(_C PRIVATE DEEPEP_LL_COUNT_HISTOGRAM=1)
endif()
if(DEEPEP_LL_REFERENCE_PROFILE)
    target_compile_definitions(_C PRIVATE DEEPEP_LL_REFERENCE_PROFILE=1)
endif()
target_compile_options(_C PRIVATE "$<$<NOT:$<CONFIG:Debug>>:-O2>" -Wall -Wextra -Werror -fstack-protector-strong)
target_compile_options(_C PRIVATE "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
    "-ffile-prefix-map=${SHMEM_ROOT}=SHMEM_SDK" "-ffile-prefix-map=${ASCEND_HOME_PATH}=CANN_SDK")
target_compile_definitions(_C PRIVATE "$<$<NOT:$<CONFIG:Debug>>:_FORTIFY_SOURCE=2>")
target_link_options(_C PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
# DeepJIT compiles the LL dispatch kernel at runtime. It needs C++20 and the
# torch_npu stream types its kernel handle uses; the AOT build stays untouched.
if(DEEPEP_LL_JIT)
    if(NOT TORCH_NPU_PATH)
        message(FATAL_ERROR "DEEPEP_LL_JIT requires -DTORCH_NPU_PATH (installed torch_npu directory)")
    endif()
    find_package(Torch REQUIRED)
    target_sources(_C PRIVATE csrc/runtime/ll_jit.cpp)
    set_target_properties(_C PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    # The JIT reproduces the device compile line from the AOT build, so the
    # selected SDK root and doorbell mode travel with the Host module.
    target_compile_definitions(_C PRIVATE DEEPEP_LL_JIT=1
        "DEEPEP_LL_JIT_SOURCE_ROOT=\"${CMAKE_SOURCE_DIR}/csrc\""
        "DEEPEP_LL_JIT_SHMEM_ROOT=\"${SHMEM_ROOT}\""
        DEEPEP_LL_JIT_PI_ON_CHIP=${DEEPEP_SHMEM_HNS_1825_PI_ON_CHIP})
    # Third-party headers are not ours to keep warning-clean under -Werror.
    target_include_directories(_C SYSTEM PRIVATE
        "${CMAKE_SOURCE_DIR}/third-party/deep_jit/include"
        "${TORCH_NPU_PATH}/include" "${TORCH_NPU_PATH}/include/third_party/acl/inc"
        ${TORCH_INCLUDE_DIRS})
    target_link_directories(_C PRIVATE "${TORCH_NPU_PATH}/lib" ${TORCH_LIBRARY_DIRS})
    target_link_libraries(_C PRIVATE torch_npu)
endif()
# SDKs are external dependencies. Never embed checkout/toolkit absolute RPATHs.
set_target_properties(_C PROPERTIES INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS _C LIBRARY DESTINATION .)
if(DEEPEP_BUILD_MEGAMOE)
    target_compile_definitions(_C PRIVATE DEEPEP_BUILD_MEGAMOE=1)
    add_subdirectory(experiments/megamoe)
elseif(DEEPEP_MEGAMOE_TIMER)
    message(FATAL_ERROR "DEEPEP_MEGAMOE_TIMER requires DEEPEP_BUILD_MEGAMOE")
endif()
if(DEEPEP_BUILD_EP_GMM_FUSED)
    target_compile_definitions(_C PRIVATE DEEPEP_BUILD_EP_GMM_FUSED=1)
    add_subdirectory(experiments/ep_gmm_fused)
endif()


# Capability declarations live alongside the targets actually built above.
# Packaging consumes this file instead of maintaining another operator list.
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/capabilities.json" CONTENT [=[
{
  "capabilities": ["runtime-mte", "runtime-mpi-bootstrap", "event", "elastic-low-latency-ndr-hns1825-v8", "ll-only"],
  "communication_operations": ["ElasticBuffer.dispatch", "ElasticBuffer.combine"]
}
]=])
