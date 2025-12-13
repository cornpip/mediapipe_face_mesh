# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(mediapipe_face_mesh_library VERSION 0.0.1 LANGUAGES C)

add_library(mediapipe_face_mesh SHARED
  "mediapipe_face_mesh.c"
)

set_target_properties(mediapipe_face_mesh PROPERTIES
  PUBLIC_HEADER mediapipe_face_mesh.h
  OUTPUT_NAME "mediapipe_face_mesh"
)

target_compile_definitions(mediapipe_face_mesh PUBLIC DART_SHARED_LIB)

if (ANDROID)
  # Support Android 15 16k page size
  target_link_options(mediapipe_face_mesh PRIVATE "-Wl,-z,max-page-size=16384")
endif()
