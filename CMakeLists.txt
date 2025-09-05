cmake_minimum_required(VERSION 3.17)
project(CPU_DR_BCG VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CPU_DR_BCG_BUILD_EXAMPLES "Build examples" OFF)

add_subdirectory("lib")

if (CPU_DR_BCG_BUILD_EXAMPLES)
    add_subdirectory("examples")
endif()
