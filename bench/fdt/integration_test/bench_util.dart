import 'dart:convert';

/// Shared benchmark protocol constants. Keep in sync across bench apps.
const int kWarmupRuns = 10;
const int kMeasuredRuns = 100;
const int kStreamSettleFrames = 30;
const int kCooldownSeconds = 30;

/// Fixed idle pause before each measured config, so later configs in a run
/// are not measured on a hotter device than earlier ones.
Future<void> thermalCooldown() =>
    Future<void>.delayed(const Duration(seconds: kCooldownSeconds));

Map<String, double> stats(List<double> samplesMs) {
  final List<double> s = List<double>.from(samplesMs)..sort();
  double pct(double p) => s[((s.length - 1) * p).round()];
  final double mean = s.reduce((a, b) => a + b) / s.length;
  return <String, double>{
    'mean': mean,
    'median': pct(0.5),
    'p95': pct(0.95),
    'min': s.first,
    'max': s.last,
  };
}

/// Prints one result line. The aggregate script scans stdout for this marker.
void emitResult({
  required String app,
  required String suite,
  required Map<String, Object?> config,
  required List<double> samplesMs,
  Map<String, Object?> extra = const <String, Object?>{},
}) {
  final Map<String, Object?> payload = <String, Object?>{
    'app': app,
    'suite': suite,
    'config': config,
    'stats': stats(samplesMs),
    'samples': samplesMs.length,
    ...extra,
  };
  // ignore: avoid_print
  print('BENCH_JSON ${jsonEncode(payload)}');
}
