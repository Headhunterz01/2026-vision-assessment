cmake_minimum_required(VERSION 3.16)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(qd_yolo_pose LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# --- 库目标 ---
add_library(qd_yolo_pose src/yolo_pose_openvino.cpp)

target_include_directories(qd_yolo_pose PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(qd_yolo_pose PUBLIC
    openvino::runtime
    ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(qd_yolo_pose PRIVATE /W4)
else()
    target_compile_options(qd_yolo_pose PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- 示例目标（仅在作为顶层项目时构建） ---
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_executable(yolo_pose_demo example/example.cpp)
    target_link_libraries(yolo_pose_demo PRIVATE qd_yolo_pose)

    add_executable(yolo_pose_minimal_example example/minimal_example.cpp)
    target_link_libraries(yolo_pose_minimal_example PRIVATE qd_yolo_pose)
endif()
