import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Diagnostic profile for the Windows multi-face UI jank report: per-stage
/// cost of the multi-face flow at camera-like resolutions, all on the same
/// isolate the example app uses.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  Future<FaceMeshImage> loadPortraitScaled(double scale) async {
    final ByteData data = await rootBundle.load('assets/portrait.jpg');
    final ui.Codec codec = await ui.instantiateImageCodec(
      data.buffer.asUint8List(),
      targetWidth: (820 * scale).round(),
      targetHeight: (1024 * scale).round(),
    );
    final ui.FrameInfo frame = await codec.getNextFrame();
    final ByteData? rgba = await frame.image.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    );
    final FaceMeshImage image = FaceMeshImage(
      pixels: rgba!.buffer.asUint8List(),
      width: frame.image.width,
      height: frame.image.height,
    );
    frame.image.dispose();
    codec.dispose();
    return image;
  }

  test('mesh cost by model and delegate', () async {
    final FaceMeshImage image = await loadPortraitScaled(1.0);
    final FaceDetectorProcessor detector = await FaceDetectorProcessor.create();
    final NormalizedRect roi =
        (detector.process(image).primaryDetection!.expandedFaceRect)!;
    detector.close();

    for (final FaceMeshModel model in FaceMeshModel.values) {
      for (final FaceMeshDelegate delegate in <FaceMeshDelegate>[
        FaceMeshDelegate.cpu,
        FaceMeshDelegate.xnnpack,
      ]) {
        final FaceMeshProcessor mesh = await FaceMeshProcessor.create(
          model: model,
          delegate: delegate,
        );
        mesh.process(image, roi: roi); // warmup
        final Stopwatch clock = Stopwatch()..start();
        for (int i = 0; i < 20; i++) {
          mesh.process(image, roi: roi);
        }
        // ignore: avoid_print
        print(
          'MODEL ${model.name} ${delegate.name} '
          '(active=${mesh.activeDelegate.name}): '
          '${(clock.elapsedMicroseconds / 20 / 1000).toStringAsFixed(2)}ms',
        );
        mesh.close();
      }
    }
  }, timeout: const Timeout(Duration(minutes: 10)));

  test('multi-face per-stage cost by resolution', () async {
    // XNNPACK matches the example app after the Windows jank fix.
    const FaceMeshDelegate delegate = FaceMeshDelegate.xnnpack;
    final FaceDetectorProcessor detector = await FaceDetectorProcessor.create(
      maxResults: 4,
      delegate: delegate,
    );
    final FaceMeshProcessor multiMesh =
        await FaceMeshProcessor.createForMultiFace(
          model: FaceMeshModel.v2,
          delegate: delegate,
        );
    final FaceMeshProcessor singleMesh = await FaceMeshProcessor.create(
      model: FaceMeshModel.v2,
      delegate: delegate,
    );
    final FaceBlendshapesProcessor blendshapes =
        await FaceBlendshapesProcessor.create(delegate: delegate);
    final FaceMeshInferencePipeline multiPipeline = FaceMeshInferencePipeline(
      detector: detector,
      mesh: multiMesh,
      landmarkSmoothing: const LandmarkSmoothingOptions(),
    );
    final FaceMeshInferencePipeline singlePipeline = FaceMeshInferencePipeline(
      detector: detector,
      mesh: singleMesh,
      landmarkSmoothing: const LandmarkSmoothingOptions(),
    );

    double time(int runs, void Function() body) {
      body(); // warmup
      final Stopwatch clock = Stopwatch()..start();
      for (int i = 0; i < runs; i++) {
        body();
      }
      return clock.elapsedMicroseconds / runs / 1000;
    }

    // 1.0 = source 820x1024; 2.0 ~ 1080p-class pixels; 3.6 ~ 4K-class.
    for (final double scale in <double>[1.0, 2.0, 3.6]) {
      final FaceMeshImage image = await loadPortraitScaled(scale);
      final NormalizedRect roi =
          (detector.process(image).primaryDetection!.expandedFaceRect)!;
      FaceMeshResult mesh = multiMesh.process(image, roi: roi);

      final double detectorMs = time(20, () => detector.process(image));
      final double roisMs = time(
        20,
        () => multiMesh.processRois(image, rois: <NormalizedRect>[roi]),
      );
      final double blendMs = time(20, () => blendshapes.process(mesh));
      singlePipeline.resetTracking();
      final double singleMs = time(20, () => singlePipeline.process(image));
      multiPipeline.resetTracking();
      final double multiMs = time(
        20,
        () => multiPipeline.processMultiFace(image, maxMeshFaces: 4),
      );

      // ignore: avoid_print
      print(
        'PROFILE ${image.width}x${image.height} '
        '(${(image.width * image.height / 1e6).toStringAsFixed(1)}MP): '
        'detector=${detectorMs.toStringAsFixed(2)}ms '
        'meshRois1=${roisMs.toStringAsFixed(2)}ms '
        'blendshapes=${blendMs.toStringAsFixed(2)}ms '
        'pipelineSingle=${singleMs.toStringAsFixed(2)}ms '
        'pipelineMulti4=${multiMs.toStringAsFixed(2)}ms',
      );
      mesh = multiMesh.process(image, roi: roi);
      expect(mesh.landmarks, hasLength(478));
    }

    blendshapes.close();
    singleMesh.close();
    multiMesh.close();
    detector.close();
  }, timeout: const Timeout(Duration(minutes: 10)));
}
