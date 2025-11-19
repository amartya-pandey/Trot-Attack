cmake_minimum_required(VERSION 4.0)
project(gaven)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(Torch_DIR "/home/amartya/Development/libtorch/share/cmake/Torch/") # pytorch directory
# external libraries
find_package(Torch REQUIRED)
find_package(SDL2 REQUIRED CONFIG)


add_executable(gaven src/main.cpp
)

# link libraries
target_link_libraries(gaven
        PRIVATE
        "${TORCH_LIBRARIES}"
        SDL2::SDL2
        SDL2::SDL2main
)

target_compile_options(gaven PRIVATE -Wall -Wextra)