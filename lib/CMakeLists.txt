add_library(dr_bcg_cpu "dr_bcg_cpu.cpp")
target_include_directories(dr_bcg_cpu PUBLIC "${CMAKE_SOURCE_DIR}/include")

include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git"
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

target_compile_definitions(
    dr_bcg_cpu PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<BOOL:${CPU_DR_BCG_PROFILE_SOLVER}>:PROFILE_SOLVER>
)
target_link_libraries(dr_bcg_cpu PUBLIC eigen)
