project(damascene)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")


find_package(CUDA 7.5)
find_path(CUDA_COMMON_INCLUDE_DIRS
  helper_cuda.h
  PATHS "${INCLUDE_DIR}/cuda"
     "${CUDA_SDK_SEARCH_PATH}"
     "${CUDA_TOOLKIT_ROOT_DIR}"	# @Kai
  PATH_SUFFIXES "samples/common/inc"
  DOC "Location of CUDA headers"
  NO_DEFAULT_PATH
)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/modules")
include(DetectCudaArch)
select_nvcc_arch_flags(NVCC_FLAGS_EXTRA)

if(ACML_INSTALL_DIR)
  set(ACML_DIR_SEARCH_PATH "${ACML_INSTALL_DIR}")
else()
  file(GLOB ACML_DIR_SEARCH_PATH "/opt/amd/acml*/*fort*64" "$ENV{HOME}/acml*/*fort*64")
endif()
find_path(ACML_HEADER_PATH
  acml.h
  PATHS "${ACML_DIR_SEARCH_PATH}"
  PATH_SUFFIXES "include"
  DOC "Location of ACML header"
  NO_DEFAULT_PATH
)
if(ACML_HEADER_PATH)
  message(STATUS "ACML header found at: ${ACML_HEADER_PATH}")
else()
  message(SEND_ERROR "ACML header not found")
endif()

include_directories(
  ${INCLUDE_DIR}
  ${ACML_HEADER_PATH}
  ${CUDA_COMMON_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg -O3)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA_readable}")
list(APPEND CUDA_NVCC_FLAGS "-Xcudafe --diag_suppress=177")   # variable was declared but never referenced
list(APPEND CUDA_NVCC_FLAGS "-Xcudafe --diag_suppress=549")   # variable is used before its value is set
list(APPEND CUDA_NVCC_FLAGS "-Xcudafe --diag_suppress=1650")  # result of call is not used
list(APPEND CUDA_NVCC_FLAGS "-Xcudafe --diag_suppress=2464")  # conversion from a string literal to "char *" is deprecated
set(CMAKE_CXX_FLAGS "-Wno-unused-result")                     # ignoring return value of function
if(PROFILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

file(GLOB_RECURSE damascene_SRC "${SOURCE_DIR}/*.cu" "${SOURCE_DIR}/*.cpp")

cuda_add_library(damascene ${damascene_SRC})
