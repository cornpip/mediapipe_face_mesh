#include "mediapipe_face.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "mediapipe_face_geometry_data.h"

namespace {

using mediapipe_face_geometry_data::kCanonicalFaceLandmarkCount;
using mediapipe_face_geometry_data::kCanonicalFaceModel;
using mediapipe_face_geometry_data::kCanonicalFaceModelWithIris;
using mediapipe_face_geometry_data::kCanonicalFaceWithIrisLandmarkCount;
using mediapipe_face_geometry_data::kProcrustesWeights;
using mediapipe_face_geometry_data::kProcrustesWeightsWithIris;

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct Mat3 {
  float m[3][3] = {};
};

struct SimilarityTransform {
  Mat3 rotation;
  Vec3 translation;
  float scale = 1.0f;
};

struct ModelData {
  const mediapipe_face_geometry_data::CanonicalPoint* canonical_points = nullptr;
  const float* weights = nullptr;
  int count = 0;
};

struct Frustum {
  float left = 0.0f;
  float right = 0.0f;
  float bottom = 0.0f;
  float top = 0.0f;
  float near_plane = 1.0f;
  float far_plane = 10000.0f;
};

std::string& GeometryLastError() {
  static std::string error;
  return error;
}

float Dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 Add(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 Sub(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 Scale(const Vec3& a, float scale) {
  return {a.x * scale, a.y * scale, a.z * scale};
}

Vec3 Mul(const Mat3& mat, const Vec3& vec) {
  return {
      mat.m[0][0] * vec.x + mat.m[0][1] * vec.y + mat.m[0][2] * vec.z,
      mat.m[1][0] * vec.x + mat.m[1][1] * vec.y + mat.m[1][2] * vec.z,
      mat.m[2][0] * vec.x + mat.m[2][1] * vec.y + mat.m[2][2] * vec.z,
  };
}

Vec3 MulTranspose(const Mat3& mat, const Vec3& vec) {
  return {
      mat.m[0][0] * vec.x + mat.m[1][0] * vec.y + mat.m[2][0] * vec.z,
      mat.m[0][1] * vec.x + mat.m[1][1] * vec.y + mat.m[2][1] * vec.z,
      mat.m[0][2] * vec.x + mat.m[1][2] * vec.y + mat.m[2][2] * vec.z,
  };
}

float NormColumn0(const SimilarityTransform& transform) {
  const float x = transform.scale * transform.rotation.m[0][0];
  const float y = transform.scale * transform.rotation.m[1][0];
  const float z = transform.scale * transform.rotation.m[2][0];
  return std::sqrt(x * x + y * y + z * z);
}

void SetIdentity(Mat3& mat) {
  std::memset(mat.m, 0, sizeof(mat.m));
  mat.m[0][0] = 1.0f;
  mat.m[1][1] = 1.0f;
  mat.m[2][2] = 1.0f;
}

void NormalizeQuaternion(float q[4]) {
  float norm = 0.0f;
  for (int i = 0; i < 4; ++i) {
    norm += q[i] * q[i];
  }
  norm = std::sqrt(norm);
  if (norm <= 1e-12f) {
    q[0] = 1.0f;
    q[1] = q[2] = q[3] = 0.0f;
    return;
  }
  for (int i = 0; i < 4; ++i) {
    q[i] /= norm;
  }
}

void QuaternionToRotation(const float q[4], Mat3& rotation) {
  const float w = q[0];
  const float x = q[1];
  const float y = q[2];
  const float z = q[3];
  rotation.m[0][0] = 1.0f - 2.0f * (y * y + z * z);
  rotation.m[0][1] = 2.0f * (x * y - z * w);
  rotation.m[0][2] = 2.0f * (x * z + y * w);
  rotation.m[1][0] = 2.0f * (x * y + z * w);
  rotation.m[1][1] = 1.0f - 2.0f * (x * x + z * z);
  rotation.m[1][2] = 2.0f * (y * z - x * w);
  rotation.m[2][0] = 2.0f * (x * z - y * w);
  rotation.m[2][1] = 2.0f * (y * z + x * w);
  rotation.m[2][2] = 1.0f - 2.0f * (x * x + y * y);
}

void LargestEigenVectorSymmetric4x4(float a[4][4], float out[4]) {
  float v[4][4] = {};
  for (int i = 0; i < 4; ++i) {
    v[i][i] = 1.0f;
  }

  for (int iter = 0; iter < 64; ++iter) {
    int p = 0;
    int q = 1;
    float max_value = std::fabs(a[p][q]);
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        const float value = std::fabs(a[i][j]);
        if (value > max_value) {
          max_value = value;
          p = i;
          q = j;
        }
      }
    }
    if (max_value < 1e-9f) {
      break;
    }

    const float app = a[p][p];
    const float aqq = a[q][q];
    const float apq = a[p][q];
    const float theta = 0.5f * std::atan2(2.0f * apq, aqq - app);
    const float c = std::cos(theta);
    const float s = std::sin(theta);

    for (int k = 0; k < 4; ++k) {
      const float aik = a[p][k];
      const float aqk = a[q][k];
      a[p][k] = c * aik - s * aqk;
      a[q][k] = s * aik + c * aqk;
    }
    for (int k = 0; k < 4; ++k) {
      const float akp = a[k][p];
      const float akq = a[k][q];
      a[k][p] = c * akp - s * akq;
      a[k][q] = s * akp + c * akq;
    }
    for (int k = 0; k < 4; ++k) {
      const float vip = v[k][p];
      const float viq = v[k][q];
      v[k][p] = c * vip - s * viq;
      v[k][q] = s * vip + c * viq;
    }
  }

  int largest = 0;
  for (int i = 1; i < 4; ++i) {
    if (a[i][i] > a[largest][largest]) {
      largest = i;
    }
  }
  for (int i = 0; i < 4; ++i) {
    out[i] = v[i][largest];
  }
  NormalizeQuaternion(out);
}

bool SolveWeightedProcrustes(const std::vector<Vec3>& source,
                             const std::vector<Vec3>& target,
                             const ModelData& model,
                             SimilarityTransform& transform) {
  if (source.size() != target.size() ||
      source.size() != static_cast<size_t>(model.count)) {
    GeometryLastError() = "Invalid landmark count for Procrustes solve.";
    return false;
  }

  float total_weight = 0.0f;
  Vec3 source_center;
  Vec3 target_center;
  for (int i = 0; i < model.count; ++i) {
    const float weight = model.weights[i];
    if (weight <= 0.0f) {
      continue;
    }
    total_weight += weight;
    source_center = Add(source_center, Scale(source[i], weight));
    target_center = Add(target_center, Scale(target[i], weight));
  }
  if (total_weight <= 1e-9f) {
    GeometryLastError() = "Procrustes weights are empty.";
    return false;
  }
  source_center = Scale(source_center, 1.0f / total_weight);
  target_center = Scale(target_center, 1.0f / total_weight);

  float h[3][3] = {};
  float source_variance = 0.0f;
  for (int i = 0; i < model.count; ++i) {
    const float weight = model.weights[i];
    if (weight <= 0.0f) {
      continue;
    }
    const Vec3 p = Sub(source[i], source_center);
    const Vec3 q = Sub(target[i], target_center);
    source_variance += weight * Dot(p, p);
    h[0][0] += weight * p.x * q.x;
    h[0][1] += weight * p.x * q.y;
    h[0][2] += weight * p.x * q.z;
    h[1][0] += weight * p.y * q.x;
    h[1][1] += weight * p.y * q.y;
    h[1][2] += weight * p.y * q.z;
    h[2][0] += weight * p.z * q.x;
    h[2][1] += weight * p.z * q.y;
    h[2][2] += weight * p.z * q.z;
  }
  if (source_variance <= 1e-9f) {
    GeometryLastError() = "Source landmarks are too compact.";
    return false;
  }

  const float sxx = h[0][0];
  const float sxy = h[0][1];
  const float sxz = h[0][2];
  const float syx = h[1][0];
  const float syy = h[1][1];
  const float syz = h[1][2];
  const float szx = h[2][0];
  const float szy = h[2][1];
  const float szz = h[2][2];

  float n[4][4] = {
      {sxx + syy + szz, syz - szy, szx - sxz, sxy - syx},
      {syz - szy, sxx - syy - szz, sxy + syx, szx + sxz},
      {szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy},
      {sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz},
  };
  float q[4] = {};
  LargestEigenVectorSymmetric4x4(n, q);
  QuaternionToRotation(q, transform.rotation);

  float numerator = 0.0f;
  for (int i = 0; i < model.count; ++i) {
    const float weight = model.weights[i];
    if (weight <= 0.0f) {
      continue;
    }
    const Vec3 p = Sub(source[i], source_center);
    const Vec3 q_target = Sub(target[i], target_center);
    numerator += weight * Dot(q_target, Mul(transform.rotation, p));
  }
  transform.scale = numerator / source_variance;
  if (!(transform.scale > 1e-9f)) {
    GeometryLastError() = "Estimated Procrustes scale is too small.";
    return false;
  }
  transform.translation = Sub(
      target_center, Scale(Mul(transform.rotation, source_center),
                           transform.scale));
  return true;
}

void FillMatrix(const SimilarityTransform& transform, float matrix[16]) {
  matrix[0] = transform.scale * transform.rotation.m[0][0];
  matrix[1] = transform.scale * transform.rotation.m[0][1];
  matrix[2] = transform.scale * transform.rotation.m[0][2];
  matrix[3] = transform.translation.x;
  matrix[4] = transform.scale * transform.rotation.m[1][0];
  matrix[5] = transform.scale * transform.rotation.m[1][1];
  matrix[6] = transform.scale * transform.rotation.m[1][2];
  matrix[7] = transform.translation.y;
  matrix[8] = transform.scale * transform.rotation.m[2][0];
  matrix[9] = transform.scale * transform.rotation.m[2][1];
  matrix[10] = transform.scale * transform.rotation.m[2][2];
  matrix[11] = transform.translation.z;
  matrix[12] = 0.0f;
  matrix[13] = 0.0f;
  matrix[14] = 0.0f;
  matrix[15] = 1.0f;
}

Frustum MakeFrustum(const MpFaceGeometryOptions* options,
                    int image_width,
                    int image_height) {
  constexpr float kPi = 3.14159265358979323846f;
  const float fov_degrees =
      (options && options->vertical_fov_degrees > 0.0f)
          ? options->vertical_fov_degrees
          : 63.0f;
  const float near_plane =
      (options && options->near_plane > 0.0f) ? options->near_plane : 1.0f;
  const float far_plane =
      (options && options->far_plane > near_plane) ? options->far_plane
                                                   : 10000.0f;
  const float height_at_near =
      2.0f * near_plane * std::tan(0.5f * fov_degrees * kPi / 180.0f);
  const float width_at_near =
      image_height > 0
          ? static_cast<float>(image_width) * height_at_near /
                static_cast<float>(image_height)
          : height_at_near;
  return {-0.5f * width_at_near,
          0.5f * width_at_near,
          -0.5f * height_at_near,
          0.5f * height_at_near,
          near_plane,
          far_plane};
}

void ProjectXY(const Frustum& frustum,
               bool origin_top_left,
               std::vector<Vec3>& landmarks) {
  const float x_scale = frustum.right - frustum.left;
  const float y_scale = frustum.top - frustum.bottom;
  for (Vec3& point : landmarks) {
    if (origin_top_left) {
      point.y = 1.0f - point.y;
    }
    point.x = point.x * x_scale + frustum.left;
    point.y = point.y * y_scale + frustum.bottom;
    point.z = point.z * x_scale;
  }
}

void MoveAndRescaleZ(const Frustum& frustum,
                     float depth_offset,
                     float scale,
                     std::vector<Vec3>& landmarks) {
  for (Vec3& point : landmarks) {
    point.z = (point.z - depth_offset + frustum.near_plane) / scale;
  }
}

void UnprojectXY(const Frustum& frustum, std::vector<Vec3>& landmarks) {
  for (Vec3& point : landmarks) {
    point.x = point.x * point.z / frustum.near_plane;
    point.y = point.y * point.z / frustum.near_plane;
  }
}

void ChangeHandedness(std::vector<Vec3>& landmarks) {
  for (Vec3& point : landmarks) {
    point.z *= -1.0f;
  }
}

float MeanZ(const std::vector<Vec3>& landmarks) {
  float mean = 0.0f;
  for (size_t i = 0; i < landmarks.size(); ++i) {
    mean += (landmarks[i].z - mean) / static_cast<float>(i + 1);
  }
  return mean;
}

std::vector<Vec3> CanonicalLandmarks(const ModelData& model) {
  std::vector<Vec3> points(model.count);
  for (int i = 0; i < model.count; ++i) {
    points[i] = {model.canonical_points[i].x, model.canonical_points[i].y,
                 model.canonical_points[i].z};
  }
  return points;
}

bool SelectModel(int32_t landmarks_count, ModelData& model) {
  if (landmarks_count == kCanonicalFaceLandmarkCount) {
    model = {kCanonicalFaceModel, kProcrustesWeights,
             kCanonicalFaceLandmarkCount};
    return true;
  }
  if (landmarks_count == kCanonicalFaceWithIrisLandmarkCount) {
    model = {kCanonicalFaceModelWithIris, kProcrustesWeightsWithIris,
             kCanonicalFaceWithIrisLandmarkCount};
    return true;
  }
  GeometryLastError() =
      "Native geometry requires 468 or 478 landmarks.";
  return false;
}

bool EstimateOfficialGeometry(const MpLandmark* landmarks,
                              int32_t landmarks_count,
                              int32_t image_width,
                              int32_t image_height,
                              const MpFaceGeometryOptions* options,
                              std::vector<Vec3>& metric_landmarks,
                              SimilarityTransform& pose_transform) {
  if (!landmarks) {
    GeometryLastError() = "Landmarks pointer is null.";
    return false;
  }
  ModelData model;
  if (!SelectModel(landmarks_count, model)) {
    return false;
  }
  if (image_width <= 0 || image_height <= 0) {
    GeometryLastError() = "Image dimensions must be positive.";
    return false;
  }

  const bool origin_top_left = !options || options->origin_top_left != 0;
  const Frustum frustum = MakeFrustum(options, image_width, image_height);
  const std::vector<Vec3> canonical = CanonicalLandmarks(model);

  std::vector<Vec3> screen(model.count);
  for (int i = 0; i < model.count; ++i) {
    screen[i] = {landmarks[i].x, landmarks[i].y, landmarks[i].z};
  }

  ProjectXY(frustum, origin_top_left, screen);
  const float depth_offset = MeanZ(screen);

  std::vector<Vec3> intermediate = screen;
  ChangeHandedness(intermediate);

  SimilarityTransform first_transform;
  if (!SolveWeightedProcrustes(canonical, intermediate, model,
                               first_transform)) {
    return false;
  }
  const float first_scale = NormColumn0(first_transform);

  intermediate = screen;
  MoveAndRescaleZ(frustum, depth_offset, first_scale, intermediate);
  UnprojectXY(frustum, intermediate);
  ChangeHandedness(intermediate);

  SimilarityTransform second_transform;
  if (!SolveWeightedProcrustes(canonical, intermediate, model,
                               second_transform)) {
    return false;
  }
  const float second_scale = NormColumn0(second_transform);

  const float total_scale = first_scale * second_scale;
  MoveAndRescaleZ(frustum, depth_offset, total_scale, screen);
  UnprojectXY(frustum, screen);
  ChangeHandedness(screen);

  if (!SolveWeightedProcrustes(canonical, screen, model, pose_transform)) {
    return false;
  }

  metric_landmarks.resize(model.count);
  for (int i = 0; i < model.count; ++i) {
    const Vec3 translated = Sub(screen[i], pose_transform.translation);
    metric_landmarks[i] =
        Scale(MulTranspose(pose_transform.rotation, translated),
              1.0f / pose_transform.scale);
  }
  return true;
}

void ExtractEulerDegrees(const SimilarityTransform& transform,
                         float* yaw_degrees,
                         float* pitch_degrees,
                         float* roll_degrees) {
  constexpr float kRadiansToDegrees = 180.0f / 3.14159265358979323846f;
  const Mat3& r = transform.rotation;
  const float pitch = std::asin(std::max(-1.0f, std::min(1.0f, -r.m[1][2])));
  const float yaw = std::atan2(r.m[0][2], r.m[2][2]);
  const float roll = std::atan2(r.m[1][0], r.m[1][1]);
  *yaw_degrees = yaw * kRadiansToDegrees;
  *pitch_degrees = pitch * kRadiansToDegrees;
  *roll_degrees = roll * kRadiansToDegrees;
}

}  // namespace

extern "C" {

MpFaceGeometryResult* mp_face_geometry_estimate(
    const MpLandmark* landmarks,
    int32_t landmarks_count,
    int32_t image_width,
    int32_t image_height,
    const MpFaceGeometryOptions* options) {
  GeometryLastError().clear();

  std::vector<Vec3> metric_landmarks;
  SimilarityTransform pose_transform;
  SetIdentity(pose_transform.rotation);
  if (!EstimateOfficialGeometry(landmarks, landmarks_count, image_width,
                                image_height, options, metric_landmarks,
                                pose_transform)) {
    return nullptr;
  }

  auto* result = new MpFaceGeometryResult();
  result->metric_landmarks_count = static_cast<int32_t>(metric_landmarks.size());
  result->metric_landmarks = new MpLandmark[metric_landmarks.size()];
  if (!result->metric_landmarks) {
    delete result;
    GeometryLastError() = "Unable to allocate geometry landmarks.";
    return nullptr;
  }

  for (size_t i = 0; i < metric_landmarks.size(); ++i) {
    result->metric_landmarks[i].x = metric_landmarks[i].x;
    result->metric_landmarks[i].y = metric_landmarks[i].y;
    result->metric_landmarks[i].z = metric_landmarks[i].z;
  }
  FillMatrix(pose_transform, result->pose_transform_matrix);
  ExtractEulerDegrees(pose_transform, &result->yaw_degrees,
                      &result->pitch_degrees, &result->roll_degrees);
  result->scale = pose_transform.scale;
  return result;
}

void mp_face_geometry_release_result(MpFaceGeometryResult* result) {
  if (!result) {
    return;
  }
  delete[] result->metric_landmarks;
  result->metric_landmarks = nullptr;
  delete result;
}

const char* mp_face_geometry_last_error(void) {
  return GeometryLastError().c_str();
}

}  // extern "C"
