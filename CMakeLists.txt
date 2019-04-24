cmake_minimum_required(VERSION 2.8.3)
project(torchjpeg)

find_package(JPEG)

message(STATUS "JPEG_LIBRARIES = ${JPEG_LIBRARIES}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

add_library(torchjpeg MODULE torchjpeg.cpp jdatadst.h jdatadst.c)

target_include_directories(torchjpeg PUBLIC ${JPEG_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.6/site-packages/torch/lib/include
        ${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.6/site-packages/torch/lib/include/torch/csrc/api/include
        ${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.6/site-packages/torch/lib/include/TH
        ${CMAKE_CURRENT_SOURCE_DIR}/.venv/lib/python3.6/site-packages/torch/lib/include/THC
        /usr/include/python3.7m
)
target_link_libraries(torchjpeg PUBLIC ${JPEG_LIBRARIES})