# The Flutter tooling requires that developers have a version of Visual Studio
# installed that includes CMake 3.14 or later. You should not increase this
# version, as doing so will cause the plugin to fail to compile for some
# customers of the plugin.
cmake_minimum_required(VERSION 3.14)

project(mediapipe_face_mesh_library VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(mediapipe_face_mesh SHARED
  "../src/mediapipe_face_mesh.cc"
  "../src/mediapipe_face_detector.cc"
  "../src/mediapipe_face_geometry.cc"
  "../src/mediapipe_face_blendshapes.cc"
)

set_target_properties(mediapipe_face_mesh PROPERTIES
  PUBLIC_HEADER "../src/mediapipe_face.h"
  OUTPUT_NAME "mediapipe_face_mesh"
)

# Flutter debug/profile app builds configure this target as a CMake Debug
# build, which compiles without optimization and makes the pixel-processing
# hot paths several times slower. Inference-adjacent code must always be
# optimized. MSVC Debug defaults (/Od /RTC1) conflict with /O2, so rewrite
# the debug flags for this target's directory instead of appending.
if (MSVC)
  string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REPLACE "/Od" "/O2" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
else()
  target_compile_options(mediapipe_face_mesh PRIVATE -O3 -ffp-contract=fast)
endif()

target_compile_definitions(mediapipe_face_mesh PUBLIC DART_SHARED_LIB)
target_compile_definitions(mediapipe_face_mesh PRIVATE NOMINMAX)
target_include_directories(mediapipe_face_mesh PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/include"
)

# Bundle the plugin library and the TensorFlow Lite C runtime next to the
# executable so `LoadLibrary("tensorflowlite_c.dll")` resolves at runtime.
set(mediapipe_face_mesh_bundled_libraries
  $<TARGET_FILE:mediapipe_face_mesh>
  "${CMAKE_CURRENT_SOURCE_DIR}/blobs/tensorflowlite_c.dll"
  PARENT_SCOPE
)
