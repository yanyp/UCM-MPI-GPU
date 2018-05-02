project(ucm-mpi)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# @Kai
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-9.1")

if(ACML_INSTALL_DIR)
  set(ACML_DIR_SEARCH_PATH "${ACML_INSTALL_DIR}")
else()
  file(GLOB ACML_DIR_SEARCH_PATH "/opt/amd/acml*/*fort*64" "$ENV{HOME}/acml*/*fort*64")
endif()
find_path(ACML_DIR
  libacml.so
  PATHS "${ACML_DIR_SEARCH_PATH}"
  PATH_SUFFIXES "lib"
  DOC "Location of ACML library"
  NO_DEFAULT_PATH
)
set(ACML_LIBRARY "${ACML_DIR}/libacml.so")
if(EXISTS ${ACML_LIBRARY})
  message(STATUS "ACML library found at: ${ACML_LIBRARY}")
else()
  message(SEND_ERROR "ACML library not found")
endif()

find_package(CUDA 7.5)

if(MPI_INSTALL_DIR)
  set(CMAKE_PREFIX_PATH "${MPI_INSTALL_DIR}")
  message(STATUS "Custom MPI installation directory specified by user: ${MPI_INSTALL_DIR}")
endif()
find_package(MPI REQUIRED)

if(OPENCV_INSTALL_DIR)
  set(CMAKE_PREFIX_PATH "${OPENCV_INSTALL_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${OPENCV_INSTALL_DIR}/share/OpenCV")
  message(STATUS "Custom OpenCV installation directory specified by user: ${OPENCV_INSTALL_DIR}")
endif()
find_package(OpenCV 2 REQUIRED)

include_directories(BEFORE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/buildW/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/CSparse/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/damascene/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/globalPb/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/log"
  "${CMAKE_CURRENT_SOURCE_DIR}/Spectra"
  "${INCLUDE_DIR}"
  "${MPI_C_INCLUDE_PATH}"
  "${OpenCV_INCLUDE_DIRS}"
)

set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations")
if(PROFILE)
  message(STATUS "Adding -pg to profile ${CMAKE_PROJECT_NAME}")
  set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} -pg")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()
add_definitions("-DLOG_USE_COLOR")

file(GLOB_RECURSE UCMMPI_SRC "${SOURCE_DIR}/*.cpp")

add_subdirectory(buildW)
add_subdirectory(CSparse)
add_subdirectory(damascene)
add_subdirectory(globalPb)

add_executable(ucm-mpi ${UCMMPI_SRC})
target_link_libraries(ucm-mpi
  buildW
  CSparse
  damascene
  globalPb
  ${ACML_LIBRARY}
  ${CUDA_CUBLAS_LIBRARIES}      # libcublas.so
  ${CUDA_CUDA_LIBRARY}          # libcuda.so
  ${CUDA_LIBRARIES}             # libcudart.so
  ${MPI_LIBRARIES}
  ${OpenCV_LIBS}
)
