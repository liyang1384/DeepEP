# 独立构建 C++/MPI dispatch 示例，不并入主工程。
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   cmake -S tests/ll_cpp_dispatch -B build/ll_cpp_dispatch \
#         -DDEEPEP_PACKAGE_DIR=$(python -c "import deep_ep,os;print(os.path.dirname(deep_ep.__file__))")
cmake_minimum_required(VERSION 3.20)
project(ll_cpp_dispatch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()

find_package(MPI REQUIRED)

# SDK 根目录：优先用 -D 传入，其次取环境变量（与主工程一致）。
if(NOT IS_DIRECTORY "${ASCEND_HOME_PATH}")
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT IS_DIRECTORY "${SHMEM_ROOT}")
    set(SHMEM_ROOT "$ENV{SHMEM_ROOT}")
endif()
if(NOT IS_DIRECTORY "${ASCEND_HOME_PATH}")
    message(FATAL_ERROR "ASCEND_HOME_PATH is required; source the CANN set_env.sh first")
endif()
if(NOT IS_DIRECTORY "${SHMEM_ROOT}")
    message(FATAL_ERROR "SHMEM_ROOT is required; it must point at the native SHMEM SDK")
endif()

get_filename_component(DEEPEP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
# libdeepep_kernels.so 随 wheel 安装在 deep_ep 包目录内。
if(NOT DEFINED DEEPEP_PACKAGE_DIR)
    set(DEEPEP_PACKAGE_DIR "$ENV{DEEPEP_PACKAGE_DIR}")
endif()
find_library(DEEPEP_KERNEL_LIB NAMES deepep_kernels
    PATHS "${DEEPEP_PACKAGE_DIR}" "${DEEPEP_SOURCE_DIR}/build" "${DEEPEP_SOURCE_DIR}/build/device" NO_DEFAULT_PATH)
if(NOT DEEPEP_KERNEL_LIB)
    message(FATAL_ERROR "libdeepep_kernels.so not found; pass -DDEEPEP_PACKAGE_DIR=<deep_ep dir>")
endif()

find_path(DEEPEP_PLATFORM_INCLUDE tiling/platform/platform_ascendc.h
    HINTS "${ASCEND_HOME_PATH}/asc/include" "${ASCEND_HOME_PATH}/include/op_common"
          "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/asc/include" NO_DEFAULT_PATH REQUIRED)

add_executable(ll_cpp_dispatch
    main.cpp
    ll_dispatch_host.cpp
    "${DEEPEP_SOURCE_DIR}/csrc/runtime/session.cpp")
target_include_directories(ll_cpp_dispatch PRIVATE
    "${DEEPEP_SOURCE_DIR}/csrc"
    "${DEEPEP_PLATFORM_INCLUDE}"
    "${ASCEND_HOME_PATH}/include"
    "${SHMEM_ROOT}/include")
# Host 与 Device 的 RDMA 能力必须同时打开，Session 里用同一个宏切换实现。
target_compile_definitions(ll_cpp_dispatch PRIVATE DEEPEP_LL_RDMA=1)
target_compile_options(ll_cpp_dispatch PRIVATE -O2 -Wall -Wextra)
target_link_directories(ll_cpp_dispatch PRIVATE "${ASCEND_HOME_PATH}/lib64")
target_link_libraries(ll_cpp_dispatch PRIVATE
    MPI::MPI_CXX
    "${DEEPEP_KERNEL_LIB}"
    ascendcl runtime platform tiling_api unified_dlog shmem
    ${CMAKE_DL_LIBS} pthread)
set_target_properties(ll_cpp_dispatch PROPERTIES BUILD_RPATH "${DEEPEP_PACKAGE_DIR}")
