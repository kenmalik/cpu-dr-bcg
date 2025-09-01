cmake_minimum_required(VERSION 3.17)
project(CPU_DR_BCG VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory("lib")

add_subdirectory("examples")

