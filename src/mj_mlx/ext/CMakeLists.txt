# Copyright 2026 the MilliJoule authors
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.27)
project(mj_stream LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import mlx, os; print(os.path.join(mlx.__path__[0], 'share', 'cmake', 'MLX'))"
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_DIR)
find_package(MLX CONFIG REQUIRED PATHS ${MLX_DIR} NO_DEFAULT_PATH)

set(MJ_NB_DOMAIN "" CACHE STRING "nanobind domain (must match MLX's)")
if(MJ_NB_DOMAIN)
  nanobind_add_module(_mj_stream NB_STATIC NB_DOMAIN ${MJ_NB_DOMAIN} mj_stream.cpp)
else()
  nanobind_add_module(_mj_stream NB_STATIC mj_stream.cpp)
endif()
target_link_libraries(_mj_stream PRIVATE mlx)
