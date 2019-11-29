cmake_minimum_required(VERSION 3.15)
project(heatmap)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(heatmap main.cpp)
target_link_libraries(heatmap ${OpenCV_LIBS})