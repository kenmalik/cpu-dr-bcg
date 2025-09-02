add_library(dr_bcg_cpu "dr_bcg_cpu.cpp")
target_include_directories(dr_bcg_cpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

target_link_libraries(dr_bcg_cpu PUBLIC eigen)
