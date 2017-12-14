cmake_minimum_required(VERSION 3.8)
project(ComposeImage)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
find_package(OpenCV REQUIRED)
add_executable(ComposeImage ${SOURCE_FILES})
target_link_libraries(ComposeImage ${OpenCV_LIBS})