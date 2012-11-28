cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(CV-3)


find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(segmentation src/segmentation.cpp)
target_link_libraries(segmentation ${PCL_LIBRARIES})

add_executable(example2 src/example2.cpp)
target_link_libraries(example2 ${PCL_LIBRARIES})
