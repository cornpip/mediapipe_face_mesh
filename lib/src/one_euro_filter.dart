import 'dart:math' as math;

/// A landmark position handled by [OneEuroLandmarksSmoother].
///
/// `x` and `y` are normalized to the range `0..1`; `z` uses the same
/// canonical units as `FaceMeshLandmark.z`.
typedef SmoothablePoint = ({double x, double y, double z});

/// Tuning parameters for OneEuro landmark smoothing.
///
/// The defaults are the values the official MediaPipe FaceLandmarker task
/// applies to face landmarks in stream (video) mode
/// (`face_landmarks_detector_graph.cc`).
class LandmarkSmoothingOptions {
  /// Creates a smoothing configuration.
  const LandmarkSmoothingOptions({
    this.minCutoff = 0.05,
    this.beta = 80.0,
    this.derivateCutoff = 1.0,
    this.initialFrequency = 30.0,
    this.minAllowedObjectScale = 1e-6,
    this.disableValueScaling = false,
  });

  /// Minimum cutoff frequency in Hz. Lower values remove more jitter while
  /// the landmarks are still, at the cost of more lag.
  final double minCutoff;

  /// Speed coefficient. Higher values track fast motion more closely.
  final double beta;

  /// Cutoff frequency for the derivative low-pass filter, in Hz.
  final double derivateCutoff;

  /// Frame frequency assumed before two timestamps have been observed.
  final double initialFrequency;

  /// Face scales (in pixels) below this bypass smoothing for the frame,
  /// matching the official calculator's `min_allowed_object_scale`.
  final double minAllowedObjectScale;

  /// Disables normalizing landmark velocity by the face's on-screen size.
  ///
  /// When scaling is enabled (the default), the same [beta] behaves
  /// consistently regardless of face size and image resolution.
  final bool disableValueScaling;
}

/// Dart port of MediaPipe's OneEuro filter
/// (`mediapipe/util/filtering/one_euro_filter.cc`), itself an implementation
/// of the 1€ filter by Casiez et al.
///
/// The filter adapts its cutoff frequency to the signal's speed: slow
/// changes are smoothed strongly (jitter removal) while fast changes pass
/// through with little lag.
class OneEuroFilter {
  /// Creates a scalar OneEuro filter.
  OneEuroFilter({
    double frequency = 30.0,
    this.minCutoff = 1.0,
    this.beta = 0.0,
    this.derivateCutoff = 1.0,
  }) : _frequency = frequency;

  /// See [LandmarkSmoothingOptions.minCutoff].
  final double minCutoff;

  /// See [LandmarkSmoothingOptions.beta].
  final double beta;

  /// See [LandmarkSmoothingOptions.derivateCutoff].
  final double derivateCutoff;

  double _frequency;
  int? _lastTimeMicros;
  final _LowPassFilter _x = _LowPassFilter();
  final _LowPassFilter _dx = _LowPassFilter();

  /// Filters [value] sampled at [timestamp].
  ///
  /// [timestamp] must be strictly increasing across calls; a non-increasing
  /// timestamp returns [value] unfiltered, like the official implementation.
  /// [valueScale] scales the velocity estimate — pass the inverse of the
  /// tracked object's size so [beta] is object-scale independent.
  double apply(Duration timestamp, double value, {double valueScale = 1.0}) {
    final int newTimeMicros = timestamp.inMicroseconds;
    final int? lastTimeMicros = _lastTimeMicros;
    if (lastTimeMicros != null) {
      if (newTimeMicros <= lastTimeMicros) {
        return value;
      }
      _frequency = 1e6 / (newTimeMicros - lastTimeMicros);
    }
    _lastTimeMicros = newTimeMicros;

    final double dvalue = _x.hasLastRawValue
        ? (value - _x.lastRawValue) * valueScale * _frequency
        : 0.0;
    final double edvalue = _dx.applyWithAlpha(dvalue, _alpha(derivateCutoff));
    final double cutoff = minCutoff + beta * edvalue.abs();
    return _x.applyWithAlpha(value, _alpha(cutoff));
  }

  double _alpha(double cutoff) {
    final double te = 1.0 / _frequency;
    final double tau = 1.0 / (2 * math.pi * cutoff);
    return 1.0 / (1.0 + tau / te);
  }
}

/// Exponential low-pass filter used by [OneEuroFilter]; port of
/// `mediapipe/util/filtering/low_pass_filter.cc`.
class _LowPassFilter {
  bool _initialized = false;
  double _storedValue = 0;
  double _rawValue = 0;

  bool get hasLastRawValue => _initialized;
  double get lastRawValue => _rawValue;

  double applyWithAlpha(double value, double alpha) {
    final double result = _initialized
        ? alpha * value + (1.0 - alpha) * _storedValue
        : value;
    _initialized = true;
    _rawValue = value;
    _storedValue = result;
    return result;
  }
}

/// Applies OneEuro smoothing to a full set of normalized landmarks, matching
/// the official `LandmarksSmoothingCalculator` OneEuro path
/// (`mediapipe/calculators/util/landmarks_smoothing_calculator_utils.cc`).
///
/// Landmarks are converted to pixel space, every axis of every landmark is
/// filtered by its own [OneEuroFilter] with the velocity normalized by the
/// face's pixel-space scale, and the result is converted back. Feed each
/// face its own smoother instance; timestamps must be strictly increasing.
class OneEuroLandmarksSmoother {
  /// Creates a landmark smoother; [options] defaults to the official
  /// FaceLandmarker stream-mode configuration.
  OneEuroLandmarksSmoother({this.options = const LandmarkSmoothingOptions()});

  /// The tuning parameters in effect.
  final LandmarkSmoothingOptions options;

  List<OneEuroFilter> _xFilters = <OneEuroFilter>[];
  List<OneEuroFilter> _yFilters = <OneEuroFilter>[];
  List<OneEuroFilter> _zFilters = <OneEuroFilter>[];

  /// Drops all filter state; the next [apply] passes its input through and
  /// starts a fresh smoothing sequence.
  void reset() {
    _xFilters = <OneEuroFilter>[];
    _yFilters = <OneEuroFilter>[];
    _zFilters = <OneEuroFilter>[];
  }

  /// Smooths [landmarks] sampled at [timestamp] on an
  /// [imageWidth]x[imageHeight] frame.
  ///
  /// Returns the input unchanged (without consuming filter state) when the
  /// landmark bounding box is degenerate. Filter state resets automatically
  /// when the landmark count changes.
  List<SmoothablePoint> apply({
    required List<SmoothablePoint> landmarks,
    required int imageWidth,
    required int imageHeight,
    required Duration timestamp,
  }) {
    if (landmarks.isEmpty || imageWidth <= 0 || imageHeight <= 0) {
      return landmarks;
    }

    // Pixel-space object scale: mean of the landmark bounding box sides,
    // like the official GetObjectScale.
    double minX = double.infinity;
    double minY = double.infinity;
    double maxX = double.negativeInfinity;
    double maxY = double.negativeInfinity;
    for (final SmoothablePoint landmark in landmarks) {
      minX = math.min(minX, landmark.x);
      minY = math.min(minY, landmark.y);
      maxX = math.max(maxX, landmark.x);
      maxY = math.max(maxY, landmark.y);
    }
    final double objectScale =
        ((maxX - minX) * imageWidth + (maxY - minY) * imageHeight) / 2.0;
    if (objectScale < options.minAllowedObjectScale) {
      return landmarks;
    }
    final double valueScale = options.disableValueScaling
        ? 1.0
        : 1.0 / objectScale;

    if (_xFilters.length != landmarks.length) {
      _xFilters = _makeFilters(landmarks.length);
      _yFilters = _makeFilters(landmarks.length);
      _zFilters = _makeFilters(landmarks.length);
    }

    final List<SmoothablePoint> smoothed = <SmoothablePoint>[];
    for (int i = 0; i < landmarks.length; i++) {
      final SmoothablePoint landmark = landmarks[i];
      final double x = _xFilters[i].apply(
        timestamp,
        landmark.x * imageWidth,
        valueScale: valueScale,
      );
      final double y = _yFilters[i].apply(
        timestamp,
        landmark.y * imageHeight,
        valueScale: valueScale,
      );
      // z uses image width for both conversions, like the official
      // normalized<->image landmark helpers.
      final double z = _zFilters[i].apply(
        timestamp,
        landmark.z * imageWidth,
        valueScale: valueScale,
      );
      smoothed.add((x: x / imageWidth, y: y / imageHeight, z: z / imageWidth));
    }
    return smoothed;
  }

  List<OneEuroFilter> _makeFilters(int count) => List<OneEuroFilter>.generate(
    count,
    (_) => OneEuroFilter(
      frequency: options.initialFrequency,
      minCutoff: options.minCutoff,
      beta: options.beta,
      derivateCutoff: options.derivateCutoff,
    ),
  );
}
