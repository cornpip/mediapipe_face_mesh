part of 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// A 3D point in the estimated face metric space.
class FaceMeshMetricPoint {
  /// Creates a point whose coordinates are expressed in centimeters.
  const FaceMeshMetricPoint({
    required this.xCm,
    required this.yCm,
    required this.zCm,
  });

  /// Horizontal coordinate in centimeters.
  final double xCm;

  /// Vertical coordinate in centimeters.
  final double yCm;

  /// Depth coordinate in centimeters.
  final double zCm;

  /// Euclidean distance to [other] in centimeters.
  double distanceTo(FaceMeshMetricPoint other) {
    final double dx = xCm - other.xCm;
    final double dy = yCm - other.yCm;
    final double dz = zCm - other.zCm;
    return math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  @override
  String toString() => 'FaceMeshMetricPoint(xCm: $xCm, yCm: $yCm, zCm: $zCm)';
}

/// Estimated yaw, pitch, and roll angles for the head.
class FaceMeshHeadPose {
  /// Creates a head pose angle bundle.
  const FaceMeshHeadPose({
    required this.yawDegrees,
    required this.pitchDegrees,
    required this.rollDegrees,
  });

  /// Left/right head rotation in degrees.
  final double yawDegrees;

  /// Up/down head rotation in degrees.
  final double pitchDegrees;

  /// In-plane head tilt in degrees.
  final double rollDegrees;

  @override
  String toString() =>
      'FaceMeshHeadPose(yawDegrees: $yawDegrees, '
      'pitchDegrees: $pitchDegrees, rollDegrees: $rollDegrees)';
}

/// A named face measurement in centimeters.
class FaceMeshMeasurement {
  /// Creates a measurement with the given [valueCm].
  const FaceMeshMeasurement(this.valueCm);

  /// Estimated value in centimeters.
  final double valueCm;

  @override
  String toString() => 'FaceMeshMeasurement($valueCm cm)';
}

/// Common face measurements derived from the estimated metric landmarks.
class FaceMeshMeasurements {
  /// Creates a preset measurement bundle.
  const FaceMeshMeasurements({
    required this.faceWidth,
    required this.faceHeight,
    required this.eyeOuterDistance,
    required this.eyeInnerDistance,
    required this.interpupillaryDistance,
    required this.mouthWidth,
    required this.noseWidth,
  });

  /// Estimated cheek-to-cheek face width.
  final FaceMeshMeasurement faceWidth;

  /// Estimated forehead-to-chin face height.
  final FaceMeshMeasurement faceHeight;

  /// Estimated distance between the outer eye corners.
  final FaceMeshMeasurement eyeOuterDistance;

  /// Estimated distance between the inner eye corners.
  final FaceMeshMeasurement eyeInnerDistance;

  /// Estimated interpupillary distance.
  ///
  /// `null` when iris landmarks are unavailable (468-landmark result).
  /// Enable iris via `FaceMeshProcessor.create(enableIris: true)` for a
  /// 478-landmark result with pupil-center landmarks.
  final FaceMeshMeasurement? interpupillaryDistance;

  /// Estimated mouth width.
  final FaceMeshMeasurement mouthWidth;

  /// Estimated nose width.
  final FaceMeshMeasurement noseWidth;
}

/// Estimated 3D geometry derived from a [FaceMeshResult].
class FaceMeshGeometry {
  const FaceMeshGeometry._({
    required this.metricLandmarks,
    required this.poseTransformMatrix,
    required this.headPose,
  });

  /// Landmarks in estimated centimeter coordinates.
  final List<FaceMeshMetricPoint> metricLandmarks;

  /// Row-major 4x4 transform matrix for the estimated face pose.
  final List<double> poseTransformMatrix;

  /// Estimated 3D head pose angles.
  final FaceMeshHeadPose headPose;

  /// Returns the 3D estimated centimeter distance between two landmarks.
  double distanceCm(int landmarkA, int landmarkB) {
    _validateLandmarkIndex(landmarkA, metricLandmarks.length);
    _validateLandmarkIndex(landmarkB, metricLandmarks.length);
    return metricLandmarks[landmarkA].distanceTo(metricLandmarks[landmarkB]);
  }

  /// Common face measurement presets.
  FaceMeshMeasurements get measurements {
    FaceMeshMeasurement measure(int a, int b) =>
        FaceMeshMeasurement(metricLandmarks[a].distanceTo(metricLandmarks[b]));

    final bool hasIris = metricLandmarks.length > 473;
    return FaceMeshMeasurements(
      faceWidth: measure(234, 454),
      faceHeight: measure(10, 152),
      eyeOuterDistance: measure(33, 263),
      eyeInnerDistance: measure(133, 362),
      interpupillaryDistance: hasIris ? measure(468, 473) : null,
      mouthWidth: measure(61, 291),
      noseWidth: measure(98, 327),
    );
  }
}

/// Distance and geometry helpers attached to [FaceMeshResult].
extension FaceMeshResultGeometry on FaceMeshResult {
  /// Returns the 2D pixel distance between two landmarks.
  double distancePixels(int landmarkA, int landmarkB) {
    _validateLandmarkIndex(landmarkA, landmarks.length);
    _validateLandmarkIndex(landmarkB, landmarks.length);
    final FaceMeshLandmark a = landmarks[landmarkA];
    final FaceMeshLandmark b = landmarks[landmarkB];
    final double dx = (a.x - b.x) * imageWidth;
    final double dy = (a.y - b.y) * imageHeight;
    return math.sqrt(dx * dx + dy * dy);
  }

  /// Estimates centimeter-scaled 3D geometry from normalized landmarks using
  /// the canonical face geometry model.
  FaceMeshGeometry estimateGeometry() {
    if (landmarks.length != 468 && landmarks.length != 478) {
      throw StateError('Geometry requires 468 or 478 landmarks.');
    }
    ffi.Pointer<MpLandmark> landmarksPtr = ffi.nullptr;
    ffi.Pointer<MpFaceGeometryOptions> optionsPtr = ffi.nullptr;
    ffi.Pointer<MpFaceGeometryResult> resultPtr = ffi.nullptr;
    try {
      landmarksPtr = pkg_ffi.calloc<MpLandmark>(landmarks.length);
      for (int i = 0; i < landmarks.length; ++i) {
        final FaceMeshLandmark landmark = landmarks[i];
        landmarksPtr[i]
          ..x = landmark.x
          ..y = landmark.y
          ..z = landmark.z;
      }
      optionsPtr = pkg_ffi.calloc<MpFaceGeometryOptions>();
      optionsPtr.ref
        ..vertical_fov_degrees = 63.0
        ..near_plane = 1.0
        ..far_plane = 10000.0
        ..origin_top_left = 1;

      resultPtr = faceBindings.mp_face_geometry_estimate(
        landmarksPtr,
        landmarks.length,
        imageWidth,
        imageHeight,
        optionsPtr,
      );
      if (resultPtr == ffi.nullptr) {
        final String? error = _nativeGeometryLastError();
        throw MediapipeFaceMeshException(
          error == null || error.isEmpty
              ? 'Native face geometry estimation is unavailable.'
              : error,
        );
      }
      final MpFaceGeometryResult nativeResult = resultPtr.ref;
      final int count = nativeResult.metric_landmarks_count;
      if (count <= 0 || nativeResult.metric_landmarks == ffi.nullptr) {
        throw MediapipeFaceMeshException(
          'Native face geometry returned empty landmarks.',
        );
      }
      final List<FaceMeshMetricPoint> metricLandmarks =
          List<FaceMeshMetricPoint>.generate(count, (int index) {
            final MpLandmark landmark =
                (nativeResult.metric_landmarks + index).ref;
            return FaceMeshMetricPoint(
              xCm: landmark.x,
              yCm: landmark.y,
              zCm: landmark.z,
            );
          }, growable: false);
      final List<double> matrix = List<double>.generate(
        16,
        (int index) => nativeResult.pose_transform_matrix[index],
        growable: false,
      );
      return FaceMeshGeometry._(
        metricLandmarks: metricLandmarks,
        poseTransformMatrix: matrix,
        headPose: FaceMeshHeadPose(
          yawDegrees: nativeResult.yaw_degrees,
          pitchDegrees: nativeResult.pitch_degrees,
          rollDegrees: nativeResult.roll_degrees,
        ),
      );
    } catch (e) {
      if (e is MediapipeFaceMeshException) rethrow;
      final String? nativeError = _nativeGeometryLastError();
      throw MediapipeFaceMeshException(
        nativeError == null || nativeError.isEmpty
            ? 'Native face geometry estimation failed.'
            : nativeError,
      );
    } finally {
      if (resultPtr != ffi.nullptr) {
        faceBindings.mp_face_geometry_release_result(resultPtr);
      }
      if (optionsPtr != ffi.nullptr) {
        pkg_ffi.calloc.free(optionsPtr);
      }
      if (landmarksPtr != ffi.nullptr) {
        pkg_ffi.calloc.free(landmarksPtr);
      }
    }
  }
}

String? _nativeGeometryLastError() {
  try {
    return _readCString(faceBindings.mp_face_geometry_last_error());
  } catch (_) {
    return null;
  }
}

void _validateLandmarkIndex(int landmark, int count) {
  if (landmark < 0 || landmark >= count) {
    throw RangeError.range(landmark, 0, count - 1, 'landmark');
  }
}
