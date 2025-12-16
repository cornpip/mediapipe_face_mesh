# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(mediapipe_face_mesh_library VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mediapipe_face_mesh SHARED
  "../../src/mediapipe_face_mesh.cc"
  "../../src/face_detection_short_range.cc"
)

set_target_properties(mediapipe_face_mesh PROPERTIES
  PUBLIC_HEADER "../../src/mediapipe_face.h"
  OUTPUT_NAME "mediapipe_face_mesh"
)

target_compile_definitions(mediapipe_face_mesh PUBLIC DART_SHARED_LIB)
target_include_directories(mediapipe_face_mesh PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
)

if (ANDROID)
  # Support Android 15 16k page size
  target_link_options(mediapipe_face_mesh PRIVATE "-Wl,-z,max-page-size=16384")
  target_link_libraries(mediapipe_face_mesh PRIVATE log dl)
elseif (UNIX AND NOT APPLE)
  target_link_libraries(mediapipe_face_mesh PRIVATE dl)
endif()
