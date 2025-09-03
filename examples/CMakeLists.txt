include(FetchContent)
FetchContent_Declare(
    suitesparse_matrix
    GIT_REPOSITORY git@github.com:kenmalik/suitesparse-matrix.git
    GIT_TAG main
)
FetchContent_MakeAvailable(suitesparse_matrix)

add_executable(example "example.cpp")
target_link_libraries(example PRIVATE dr_bcg_cpu suitesparse_matrix)
