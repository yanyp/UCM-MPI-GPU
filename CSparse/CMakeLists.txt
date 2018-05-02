project(CSparse)
cmake_minimum_required(VERSION 2.8)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories(
  ${INCLUDE_DIR}
)

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE CSparse_SRC "${SOURCE_DIR}/*.c")

add_library(CSparse ${CSparse_SRC})
