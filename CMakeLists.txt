cmake_minimum_required(VERSION 4.0)
project(gaven)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED Torch_DIR)
    # REPLACE THIS WITH YOUR LOCAL LINUX PATH
    set(Torch_DIR "/home/archuser/Development/libtorch/share/cmake/Torch")
endif()
# external libraries
find_package(Torch REQUIRED)

if(WIN32)
    set(SDL2_PATH "${CMAKE_SOURCE_DIR}/libs/SDL2")
    find_package(SDL2 REQUIRED CONFIG)
else()
    find_package(SDL2 REQUIRED CONFIG)
endif()


add_executable(gaven src/main.cpp
        src/Game.cpp
        src/Game.h
        src/Entity.cpp
        src/Entity.h
        src/NPC.cpp
        src/NPC.h
        src/AICore.cpp
        src/AICore.h
        src/Player.cpp
        src/Player.h
)

# link libraries
target_link_libraries(gaven
        PRIVATE
        "${TORCH_LIBRARIES}"
        SDL2::SDL2
        SDL2::SDL2main
)

target_compile_options(gaven PRIVATE -Wall -Wextra)