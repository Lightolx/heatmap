cmake_minimum_required(VERSION 3.15)
project(heatmap)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
add_definitions(${PCL_DEFINITIONS})

add_executable(heatmap main.cpp)
target_link_libraries(heatmap ${OpenCV_LIBS})

add_executable(laneMatching LaneMatching.cpp)
target_link_libraries(laneMatching ${OpenCV_LIBS} ${PCL_LIBRARIES})