// Replaces TensorFlow Lite's default tflite::CreateOpResolver() for the
// bundled mediapipe_face_mesh runtime: the builtin ops plus the MediaPipe
// custom ops that face_landmark_with_attention.tflite needs.
#include <memory>

#include "mediapipe/util/tflite/cpu_op_resolver.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {

std::unique_ptr<MutableOpResolver> CreateOpResolver() {  // NOLINT
  auto resolver = std::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  mediapipe::MediaPipe_RegisterTfLiteOpResolver(resolver.get());
  return resolver;
}

}  // namespace tflite
