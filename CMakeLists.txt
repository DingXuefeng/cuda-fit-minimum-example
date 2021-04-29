cmake_minimum_required(VERSION 3.16)
project(hello VERSION 1.0.0)
enable_language(CUDA)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 60)


find_package(ROOT REQUIRED)

add_executable(hello hello.cu)
target_link_libraries(hello PRIVATE ROOT::Minuit)
install(TARGETS hello DESTINATION .)
