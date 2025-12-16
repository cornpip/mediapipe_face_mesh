#include "mediapipe_face.h"
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if defined(__OBJC__) && TARGET_OS_IPHONE
#import <Foundation/Foundation.h>
#endif
#if TARGET_OS_IPHONE
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#else
#include "tensorflow/lite/c/c_api.h"
#endif
#else
#include "tensorflow/lite/c/c_api.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include "tflite_runtime.h"

#if defined(__ANDROID__)
#include <android/log.h>
#define FD_LOG_TAG "MediapipeFaceMesh"
#define FD_LOGI(...) __android_log_print(ANDROID_LOG_INFO, FD_LOG_TAG, __VA_ARGS__)
#define FD_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, FD_LOG_TAG, __VA_ARGS__)
#else
#define FD_LOGI(...) std::fprintf(stdout, "[INFO] " __VA_ARGS__)
#define FD_LOGE(...) std::fprintf(stderr, "[ERROR] " __VA_ARGS__)
#endif

namespace {

// Anchor generator parameters for MediaPipe Face Detection short-range.
struct AnchorOptions {
  int input_width = 128;
  int input_height = 128;
  float anchor_offset_x = 0.5f;
  float anchor_offset_y = 0.5f;
  bool fixed_anchor_size = true;
  std::vector<float> aspect_ratios{1.0f};
  std::vector<float> scales{0.1484375f, 0.2109375f, 0.2734375f, 0.3359375f,
                            0.3984375f};
  float interpolated_scale_aspect_ratio = 1.0f;
  std::vector<int> strides{8, 16, 16, 16};
};

struct Anchor {
  float x_center;
  float y_center;
  float w;
  float h;
};

struct Detection {
  float x_center;
  float y_center;
  float w;
  float h;
  float score;
  float keypoints[12];
  int keypoint_count;
};

struct TfLiteModelDeleter {
  TfLiteRuntime* runtime;
  void operator()(TfLiteModel* model) const {
    if (runtime && model) {
      runtime->ModelDelete(model);
    }
  }
};

struct TfLiteOptionsDeleter {
  TfLiteRuntime* runtime;
  void operator()(TfLiteInterpreterOptions* options) const {
    if (runtime && options) {
      runtime->InterpreterOptionsDelete(options);
    }
  }
};

struct TfLiteInterpreterDeleter {
  TfLiteRuntime* runtime;
  void operator()(TfLiteInterpreter* interpreter) const {
    if (runtime && interpreter) {
      runtime->InterpreterDelete(interpreter);
    }
  }
};

struct FaceDetectionContext {
  TfLiteRuntime runtime;
  std::unique_ptr<TfLiteModel, TfLiteModelDeleter> model{nullptr, {&runtime}};
  std::unique_ptr<TfLiteInterpreterOptions, TfLiteOptionsDeleter> options{
      nullptr, {&runtime}};
  std::unique_ptr<TfLiteInterpreter, TfLiteInterpreterDeleter> interpreter{
      nullptr, {&runtime}};

  TfLiteTensor* input_tensor = nullptr;
  const TfLiteTensor* regressors = nullptr;
  const TfLiteTensor* classificators = nullptr;

  int input_w = 0;
  int input_h = 0;
  float letterbox_scale = 1.0f;
  float letterbox_pad_x = 0.0f;
  float letterbox_pad_y = 0.0f;
  int threads = 2;
  float score_threshold = 0.5f;
  float nms_threshold = 0.3f;
  int max_detections = 1;

  std::vector<float> input_buffer;
  std::vector<float> regressors_buffer;
  std::vector<float> classificators_buffer;
  std::vector<Anchor> anchors;
  std::string last_error;

  bool Initialize(const std::string& model_path,
                  const MpFaceDetectionCreateOptions* options_in) {
    threads = (options_in && options_in->threads > 0) ? options_in->threads : 2;
    score_threshold =
        (options_in && options_in->score_threshold > 0.f) ? options_in->score_threshold : 0.5f;
    nms_threshold =
        (options_in && options_in->nms_threshold > 0.f) ? options_in->nms_threshold : 0.3f;
    max_detections =
        (options_in && options_in->max_detections > 0) ? options_in->max_detections : 1;

    const char* runtime_path = (options_in && options_in->tflite_library_path)
                                   ? options_in->tflite_library_path
                                   : nullptr;
    if (!runtime.Load(runtime_path)) {
      SetError("Failed to load TensorFlow Lite runtime: " + runtime.error());
      return false;
    }

    model.reset(runtime.ModelCreateFromFile(model_path.c_str()));
    if (!model) {
      SetError("Unable to load model file: " + model_path);
      return false;
    }

    options.reset(runtime.InterpreterOptionsCreate());
    if (!options) {
      SetError("Failed to allocate interpreter options.");
      return false;
    }
    runtime.InterpreterOptionsSetThreads(options.get(), threads);

    interpreter.reset(runtime.InterpreterCreate(model.get(), options.get()));
    if (!interpreter) {
      SetError("Failed to create interpreter.");
      return false;
    }
    if (runtime.InterpreterAllocateTensors(interpreter.get()) != kTfLiteOk) {
      SetError("Tensor allocation failed.");
      return false;
    }
    input_tensor = runtime.InterpreterGetInputTensor(interpreter.get(), 0);
    if (!input_tensor || runtime.TensorNumDims(input_tensor) != 4 ||
        runtime.TensorDim(input_tensor, 0) != 1 ||
        runtime.TensorDim(input_tensor, 3) != 3) {
      SetError("Model expects 1xHxWx3 input.");
      return false;
    }
    input_h = runtime.TensorDim(input_tensor, 1);
    input_w = runtime.TensorDim(input_tensor, 2);
    input_buffer.resize(static_cast<size_t>(input_h * input_w * 3));

    if (runtime.InterpreterGetOutputTensorCount(interpreter.get()) < 2) {
      SetError("Model outputs are missing.");
      return false;
    }
    regressors = runtime.InterpreterGetOutputTensor(interpreter.get(), 0);
    classificators = runtime.InterpreterGetOutputTensor(interpreter.get(), 1);
    if (!regressors || !classificators ||
        runtime.TensorType(regressors) != kTfLiteFloat32 ||
        runtime.TensorType(classificators) != kTfLiteFloat32) {
      SetError("Unexpected output tensor types.");
      return false;
    }
    const int reg_total = runtime.TensorByteSize(regressors) / sizeof(float);
    const int cls_total = runtime.TensorByteSize(classificators) / sizeof(float);
    regressors_buffer.resize(static_cast<size_t>(reg_total));
    classificators_buffer.resize(static_cast<size_t>(cls_total));

    anchors = BuildAnchors();
    if (anchors.size() * 16 != regressors_buffer.size()) {
      SetError("Anchor count does not match regressors output size.");
      return false;
    }
    return true;
  }

  bool Preprocess(const MpImage& image) {
    const int width = image.width;
    const int height = image.height;
    const int channels = 3;
    if (width <= 0 || height <= 0 || image.bytes_per_row <= 0 ||
        !image.data) {
      SetError("Invalid image.");
      return false;
    }
    // Resize with aspect ratio kept and bilinear sampling + replicated padding
    // to mimic MediaPipe's ImageToTensor letterbox behavior.
    letterbox_scale =
        std::min(static_cast<float>(input_w) / static_cast<float>(width),
                 static_cast<float>(input_h) / static_cast<float>(height));
    const int resized_w =
        static_cast<int>(std::round(static_cast<float>(width) * letterbox_scale));
    const int resized_h =
        static_cast<int>(std::round(static_cast<float>(height) * letterbox_scale));
    letterbox_pad_x = (static_cast<float>(input_w - resized_w)) * 0.5f;
    letterbox_pad_y = (static_cast<float>(input_h - resized_h)) * 0.5f;
    for (int y = 0; y < input_h; ++y) {
      const float src_y_f =
          (static_cast<float>(y) - letterbox_pad_y) / letterbox_scale;
      const int y0 = std::max(0, std::min(height - 1,
                                          static_cast<int>(std::floor(src_y_f))));
      const int y1 = std::max(0, std::min(height - 1, y0 + 1));
      const float wy = src_y_f - std::floor(src_y_f);
      for (int x = 0; x < input_w; ++x) {
        const float src_x_f =
            (static_cast<float>(x) - letterbox_pad_x) / letterbox_scale;
        const int x0 = std::max(0, std::min(width - 1,
                                            static_cast<int>(std::floor(src_x_f))));
        const int x1 = std::max(0, std::min(width - 1, x0 + 1));
        const float wx = src_x_f - std::floor(src_x_f);
        const uint8_t* row0 =
            image.data + static_cast<size_t>(y0) * image.bytes_per_row;
        const uint8_t* row1 =
            image.data + static_cast<size_t>(y1) * image.bytes_per_row;
        const uint8_t* p00 = row0 + static_cast<size_t>(x0) * 4;
        const uint8_t* p01 = row0 + static_cast<size_t>(x1) * 4;
        const uint8_t* p10 = row1 + static_cast<size_t>(x0) * 4;
        const uint8_t* p11 = row1 + static_cast<size_t>(x1) * 4;
        float r, g, b;
        if (image.format == MP_PIXEL_FORMAT_RGBA) {
          const float r0 = (1.0f - wx) * p00[0] + wx * p01[0];
          const float r1 = (1.0f - wx) * p10[0] + wx * p11[0];
          const float g0 = (1.0f - wx) * p00[1] + wx * p01[1];
          const float g1 = (1.0f - wx) * p10[1] + wx * p11[1];
          const float b0 = (1.0f - wx) * p00[2] + wx * p01[2];
          const float b1 = (1.0f - wx) * p10[2] + wx * p11[2];
          r = (1.0f - wy) * r0 + wy * r1;
          g = (1.0f - wy) * g0 + wy * g1;
          b = (1.0f - wy) * b0 + wy * b1;
        } else {
          const float r0 = (1.0f - wx) * p00[2] + wx * p01[2];
          const float r1 = (1.0f - wx) * p10[2] + wx * p11[2];
          const float g0 = (1.0f - wx) * p00[1] + wx * p01[1];
          const float g1 = (1.0f - wx) * p10[1] + wx * p11[1];
          const float b0 = (1.0f - wx) * p00[0] + wx * p01[0];
          const float b1 = (1.0f - wx) * p10[0] + wx * p11[0];
          r = (1.0f - wy) * r0 + wy * r1;
          g = (1.0f - wy) * g0 + wy * g1;
          b = (1.0f - wy) * b0 + wy * b1;
        }
        const int idx = (y * input_w + x) * channels;
        input_buffer[idx + 0] = r / 255.0f;
        input_buffer[idx + 1] = g / 255.0f;
        input_buffer[idx + 2] = b / 255.0f;
      }
    }
    return true;
  }

  MpFaceDetectionResult* Process(const MpImage& image) {
    if (!interpreter) {
      SetError("Interpreter is not initialized.");
      return nullptr;
    }
    if (!Preprocess(image)) {
      return nullptr;
    }
    const size_t bytes = input_buffer.size() * sizeof(float);
    if (runtime.TensorCopyFromBuffer(input_tensor, input_buffer.data(), bytes) !=
        kTfLiteOk) {
      SetError("Failed to copy input buffer.");
      return nullptr;
    }
    if (runtime.InterpreterInvoke(interpreter.get()) != kTfLiteOk) {
      SetError("Interpreter invocation failed.");
      return nullptr;
    }
    if (runtime.TensorCopyToBuffer(regressors, regressors_buffer.data(),
                                   regressors_buffer.size() * sizeof(float)) !=
        kTfLiteOk) {
      SetError("Unable to read regressors output.");
      return nullptr;
    }
    if (runtime.TensorCopyToBuffer(classificators,
                                   classificators_buffer.data(),
                                   classificators_buffer.size() *
                                       sizeof(float)) != kTfLiteOk) {
      SetError("Unable to read classificators output.");
      return nullptr;
    }
    std::vector<Detection> detections =
        DecodeDetections(regressors_buffer, classificators_buffer, anchors,
                         score_threshold);
    std::vector<Detection> keep =
        WeightedNonMaxSuppression(detections, nms_threshold, max_detections);

    auto* result = new MpFaceDetectionResult();
    if (!result) {
      SetError("Unable to allocate result.");
      return nullptr;
    }
    result->detections = new MpDetection[keep.size()];
    result->count = static_cast<int32_t>(keep.size());
    result->image_width = image.width;
    result->image_height = image.height;
    for (size_t i = 0; i < keep.size(); ++i) {
      const Detection& det = keep[i];
      MpDetection& out = result->detections[i];
      const float inv_scale = (letterbox_scale > 0.0f) ? 1.0f / letterbox_scale : 1.0f;
      const float x_center_orig = (det.x_center - letterbox_pad_x) * inv_scale;
      const float y_center_orig = (det.y_center - letterbox_pad_y) * inv_scale;
      const float w_orig = det.w * inv_scale;
      const float h_orig = det.h * inv_scale;
      out.box.x_center = x_center_orig / static_cast<float>(image.width);
      out.box.y_center = y_center_orig / static_cast<float>(image.height);
      out.box.width = w_orig / static_cast<float>(image.width);
      out.box.height = h_orig / static_cast<float>(image.height);
      out.score = det.score;
      out.keypoints_count = det.keypoint_count;
      for (int k = 0; k < det.keypoint_count * 2; ++k) {
        // Normalize keypoints to original image size.
        const bool is_x = (k % 2) == 0;
        const float mapped =
            (det.keypoints[k] - (is_x ? letterbox_pad_x : letterbox_pad_y)) * inv_scale;
        out.keypoints[k] =
            mapped / (is_x ? static_cast<float>(image.width)
                           : static_cast<float>(image.height));
      }
    }
    return result;
  }

  std::vector<Anchor> BuildAnchors() const {
    std::vector<Anchor> anchors_out;
    const AnchorOptions opt;
    const size_t num_layers = opt.strides.size();
    for (size_t i = 0; i < num_layers; ++i) {
      const int stride = opt.strides[i];
      const float scale = opt.scales[i];
      const float scale_next =
          (i + 1 < opt.scales.size()) ? opt.scales[i + 1] : opt.scales.back();
      const int fm_h = (opt.input_height + stride - 1) / stride;
      const int fm_w = (opt.input_width + stride - 1) / stride;
      for (int y = 0; y < fm_h; ++y) {
        for (int x = 0; x < fm_w; ++x) {
          for (float ar : opt.aspect_ratios) {
            Anchor a;
            a.x_center = (x + opt.anchor_offset_x) / fm_w;
            a.y_center = (y + opt.anchor_offset_y) / fm_h;
            if (opt.fixed_anchor_size) {
              a.w = scale * std::sqrt(ar);
              a.h = scale / std::sqrt(ar);
            } else {
              a.w = scale * std::sqrt(ar) / opt.input_width;
              a.h = scale / std::sqrt(ar) / opt.input_height;
            }
            anchors_out.push_back(a);
          }
          if (opt.interpolated_scale_aspect_ratio > 0.f) {
            Anchor a;
            a.x_center = (x + opt.anchor_offset_x) / fm_w;
            a.y_center = (y + opt.anchor_offset_y) / fm_h;
            const float interpolated_scale = std::sqrt(scale * scale_next);
            const float ratio = opt.interpolated_scale_aspect_ratio;
            if (opt.fixed_anchor_size) {
              a.w = interpolated_scale * std::sqrt(ratio);
              a.h = interpolated_scale / std::sqrt(ratio);
            } else {
              a.w = interpolated_scale * std::sqrt(ratio) / opt.input_width;
              a.h = interpolated_scale / std::sqrt(ratio) / opt.input_height;
            }
            anchors_out.push_back(a);
          }
        }
      }
    }
    return anchors_out;
  }

  std::vector<Detection> DecodeDetections(
      const std::vector<float>& regs,
      const std::vector<float>& cls,
      const std::vector<Anchor>& anchors_in,
      float score_thresh) const {
    const int num = static_cast<int>(anchors_in.size());
    std::vector<Detection> out;
    out.reserve(num);
    const float x_scale = 128.0f;
    const float y_scale = 128.0f;
    const float w_scale = 128.0f;
    const float h_scale = 128.0f;
    for (int i = 0; i < num; ++i) {
      const float score = Sigmoid(cls[i]);
      if (score < score_thresh) {
        continue;
      }
      const Anchor& a = anchors_in[i];
      const float* r = &regs[i * 16];
      // Box
      const float cx = r[0] / x_scale * a.w + a.x_center;
      const float cy = r[1] / y_scale * a.h + a.y_center;
      const float w = r[2] / w_scale * a.w;
      const float h = r[3] / h_scale * a.h;
      Detection det;
      det.x_center = cx * static_cast<float>(input_w);
      det.y_center = cy * static_cast<float>(input_h);
      det.w = w * static_cast<float>(input_w);
      det.h = h * static_cast<float>(input_h);
      det.score = score;
      det.keypoint_count = 6;
      for (int k = 0; k < det.keypoint_count; ++k) {
        const float kx = r[4 + k * 2] / x_scale * a.w + a.x_center;
        const float ky = r[4 + k * 2 + 1] / y_scale * a.h + a.y_center;
        det.keypoints[k * 2] = kx * static_cast<float>(input_w);
        det.keypoints[k * 2 + 1] = ky * static_cast<float>(input_h);
      }
      out.push_back(det);
    }
    return out;
  }

  std::vector<Detection> WeightedNonMaxSuppression(
      const std::vector<Detection>& dets, float iou_thresh,
      int max_keep) const {
    std::vector<int> indices(dets.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return dets[a].score > dets[b].score; });
    std::vector<Detection> out;
    while (!indices.empty() && static_cast<int>(out.size()) < max_keep) {
      const int idx = indices.front();
      indices.erase(indices.begin());
      Detection merged = dets[idx];
      float weight_sum = dets[idx].score;
      float max_score = dets[idx].score;
      float sum_x = dets[idx].x_center * dets[idx].score;
      float sum_y = dets[idx].y_center * dets[idx].score;
      float sum_w = dets[idx].w * dets[idx].score;
      float sum_h = dets[idx].h * dets[idx].score;
      float sum_kp[12] = {0};
      const int kp_count = dets[idx].keypoint_count;
      for (int k = 0; k < kp_count * 2; ++k) {
        sum_kp[k] = dets[idx].keypoints[k] * dets[idx].score;
      }

      std::vector<int> remaining;
      remaining.reserve(indices.size());
      for (int j : indices) {
        if (IoU(dets[idx], dets[j]) > iou_thresh) {
          const float w = dets[j].score;
          weight_sum += w;
          max_score = std::max(max_score, dets[j].score);
          sum_x += dets[j].x_center * w;
          sum_y += dets[j].y_center * w;
          sum_w += dets[j].w * w;
          sum_h += dets[j].h * w;
          for (int k = 0; k < kp_count * 2; ++k) {
            sum_kp[k] += dets[j].keypoints[k] * w;
          }
        } else {
          remaining.push_back(j);
        }
      }
      indices.swap(remaining);
      const float inv = (weight_sum > 0.0f) ? (1.0f / weight_sum) : 0.0f;
      merged.x_center = sum_x * inv;
      merged.y_center = sum_y * inv;
      merged.w = sum_w * inv;
      merged.h = sum_h * inv;
      merged.score = max_score;
      for (int k = 0; k < kp_count * 2; ++k) {
        merged.keypoints[k] = sum_kp[k] * inv;
      }
      out.push_back(merged);
    }
    return out;
  }

  float IoU(const Detection& a, const Detection& b) const {
    const float ax0 = a.x_center - a.w * 0.5f;
    const float ay0 = a.y_center - a.h * 0.5f;
    const float ax1 = a.x_center + a.w * 0.5f;
    const float ay1 = a.y_center + a.h * 0.5f;
    const float bx0 = b.x_center - b.w * 0.5f;
    const float by0 = b.y_center - b.h * 0.5f;
    const float bx1 = b.x_center + b.w * 0.5f;
    const float by1 = b.y_center + b.h * 0.5f;
    const float ix0 = std::max(ax0, bx0);
    const float iy0 = std::max(ay0, by0);
    const float ix1 = std::min(ax1, bx1);
    const float iy1 = std::min(ay1, by1);
    const float iw = std::max(0.0f, ix1 - ix0);
    const float ih = std::max(0.0f, iy1 - iy0);
    const float inter = iw * ih;
    const float union_area = a.w * a.h + b.w * b.h - inter;
    if (union_area <= 0.0f) {
      return 0.0f;
    }
    return inter / union_area;
  }

  float Sigmoid(float x) const { return 1.f / (1.f + std::exp(-x)); }

  void SetError(const std::string& message) { last_error = message; }
};

thread_local std::string g_fd_last_global_error;

void SetFdGlobalError(const std::string& message) {
  g_fd_last_global_error = message;
}

}  // namespace

extern "C" {

FFI_PLUGIN_EXPORT MpFaceDetectionContext* mp_face_detection_create(
    const char* model_path, const MpFaceDetectionCreateOptions* options) {
  if (!model_path) {
    SetFdGlobalError("Model path is null.");
    return nullptr;
  }
  auto* ctx = new FaceDetectionContext();
  if (!ctx) {
    SetFdGlobalError("Unable to allocate detection context.");
    return nullptr;
  }
  if (!ctx->Initialize(model_path, options)) {
    SetFdGlobalError(ctx->last_error);
    delete ctx;
    return nullptr;
  }
  return reinterpret_cast<MpFaceDetectionContext*>(ctx);
}

FFI_PLUGIN_EXPORT void mp_face_detection_destroy(
    MpFaceDetectionContext* context) {
  delete reinterpret_cast<FaceDetectionContext*>(context);
}

FFI_PLUGIN_EXPORT MpFaceDetectionResult* mp_face_detection_process(
    MpFaceDetectionContext* context, const MpImage* image) {
  if (!context || !image) {
    SetFdGlobalError("Context or image is null.");
    return nullptr;
  }
  auto* ctx = reinterpret_cast<FaceDetectionContext*>(context);
  auto* result = ctx->Process(*image);
  if (!result) {
    SetFdGlobalError(ctx->last_error);
  }
  return result;
}

FFI_PLUGIN_EXPORT void mp_face_detection_release_result(
    MpFaceDetectionResult* result) {
  if (!result) {
    return;
  }
  delete[] result->detections;
  delete result;
}

FFI_PLUGIN_EXPORT const char* mp_face_detection_last_error(
    const MpFaceDetectionContext* context) {
  const auto* ctx = reinterpret_cast<const FaceDetectionContext*>(context);
  return ctx ? ctx->last_error.c_str() : "";
}

FFI_PLUGIN_EXPORT const char* mp_face_detection_last_global_error(void) {
  return g_fd_last_global_error.c_str();
}

}  // extern "C"
