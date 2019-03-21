CMAKE_MINIMUM_REQUIRED(VERSION 3.6)

project(ManTrapColor)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

#add_library(test SHARED include/test.cpp)
set(project_link_libs libRealSenseManager.dylib)
link_directories(/Users/aemass-cheanwu/dev/COPA/RealSenseManager/build/)
add_executable(ManTrapColor src/ManTrapColor.cpp)
target_link_libraries(ManTrapColor ${project_link_libs} ${OpenCV_LIBS})