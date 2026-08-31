import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:mediapipe_face_mesh/mediapipe_face_mesh.dart';

/// Conversion tests for the [FaceMeshNv21Image] factories, covering the
/// tightly-packed, row-padded, and pixel-strided plane layouts the camera
/// plugins produce.
void main() {
  Uint8List sequential(int length, {int start = 0}) =>
      Uint8List.fromList(List<int>.generate(length, (int i) => (start + i) & 0xff));

  group('tryFromYAndInterleavedVuPlanes', () {
    test('tightly packed planes convert without reordering', () {
      const int width = 4;
      const int height = 4;
      final Uint8List y = sequential(width * height);
      final Uint8List vu = sequential(width * (height ~/ 2), start: 100);

      final FaceMeshNv21Image? image =
          FaceMeshNv21Image.tryFromYAndInterleavedVuPlanes(
            width: width,
            height: height,
            yPlane: FaceMeshImagePlane(bytes: y, bytesPerRow: width),
            vuPlane: FaceMeshImagePlane(bytes: vu, bytesPerRow: width),
          );

      expect(image, isNotNull);
      expect(image!.yPlane, y);
      expect(image.vuPlane, vu);
      expect(image.yBytesPerRow, width);
      expect(image.vuBytesPerRow, width);
    });

    test('row padding is stripped', () {
      const int width = 4;
      const int height = 2;
      const int rowStride = 6;
      final Uint8List padded = sequential(rowStride * height);

      final FaceMeshNv21Image? image =
          FaceMeshNv21Image.tryFromYAndInterleavedVuPlanes(
            width: width,
            height: height,
            yPlane: FaceMeshImagePlane(bytes: padded, bytesPerRow: rowStride),
            vuPlane: FaceMeshImagePlane(
              bytes: sequential(rowStride, start: 100),
              bytesPerRow: rowStride,
            ),
          );

      expect(image, isNotNull);
      expect(image!.yPlane, <int>[0, 1, 2, 3, 6, 7, 8, 9]);
      expect(image.vuPlane, <int>[100, 101, 102, 103]);
    });

    test('undersized plane returns null', () {
      const int width = 4;
      const int height = 4;
      final FaceMeshNv21Image? image =
          FaceMeshNv21Image.tryFromYAndInterleavedVuPlanes(
            width: width,
            height: height,
            yPlane: FaceMeshImagePlane(
              bytes: sequential(width * height - 1),
              bytesPerRow: width,
            ),
            vuPlane: FaceMeshImagePlane(
              bytes: sequential(width * (height ~/ 2)),
              bytesPerRow: width,
            ),
          );

      expect(image, isNull);
    });
  });

  group('tryFromYuv420Planes', () {
    test('planar chroma interleaves as VU', () {
      const int width = 4;
      const int height = 4;
      final Uint8List y = sequential(width * height);
      // 2x2 chroma planes.
      final Uint8List u = Uint8List.fromList(<int>[1, 2, 3, 4]);
      final Uint8List v = Uint8List.fromList(<int>[5, 6, 7, 8]);

      final FaceMeshNv21Image? image = FaceMeshNv21Image.tryFromYuv420Planes(
        width: width,
        height: height,
        yPlane: FaceMeshImagePlane(bytes: y, bytesPerRow: width),
        uPlane: FaceMeshImagePlane(bytes: u, bytesPerRow: width ~/ 2),
        vPlane: FaceMeshImagePlane(bytes: v, bytesPerRow: width ~/ 2),
      );

      expect(image, isNotNull);
      expect(image!.yPlane, y);
      expect(image.vuPlane, <int>[5, 1, 6, 2, 7, 3, 8, 4]);
    });

    test('semi-planar chroma with pixel stride 2 interleaves as VU', () {
      const int width = 4;
      const int height = 4;
      final Uint8List y = sequential(width * height);
      // Android camera semi-planar layout: U and V each strided by 2.
      final Uint8List u = Uint8List.fromList(<int>[1, 0, 2, 0, 3, 0, 4, 0]);
      final Uint8List v = Uint8List.fromList(<int>[5, 0, 6, 0, 7, 0, 8, 0]);

      final FaceMeshNv21Image? image = FaceMeshNv21Image.tryFromYuv420Planes(
        width: width,
        height: height,
        yPlane: FaceMeshImagePlane(bytes: y, bytesPerRow: width),
        uPlane: FaceMeshImagePlane(bytes: u, bytesPerRow: width, bytesPerPixel: 2),
        vPlane: FaceMeshImagePlane(bytes: v, bytesPerRow: width, bytesPerPixel: 2),
      );

      expect(image, isNotNull);
      expect(image!.vuPlane, <int>[5, 1, 6, 2, 7, 3, 8, 4]);
    });

    test('trailing row without stride padding is accepted', () {
      // The last chroma row of a strided buffer often omits the final
      // padding bytes; coverage is judged by the last sample, not the full
      // stride.
      const int width = 4;
      const int height = 4;
      final Uint8List y = sequential(width * height);
      final Uint8List u = Uint8List.fromList(<int>[1, 0, 2, 0, 3, 0, 4]);
      final Uint8List v = Uint8List.fromList(<int>[5, 0, 6, 0, 7, 0, 8]);

      final FaceMeshNv21Image? image = FaceMeshNv21Image.tryFromYuv420Planes(
        width: width,
        height: height,
        yPlane: FaceMeshImagePlane(bytes: y, bytesPerRow: width),
        uPlane: FaceMeshImagePlane(bytes: u, bytesPerRow: width, bytesPerPixel: 2),
        vPlane: FaceMeshImagePlane(bytes: v, bytesPerRow: width, bytesPerPixel: 2),
      );

      expect(image, isNotNull);
      expect(image!.vuPlane, <int>[5, 1, 6, 2, 7, 3, 8, 4]);
    });

    test('undersized chroma plane returns null', () {
      const int width = 4;
      const int height = 4;
      final FaceMeshNv21Image? image = FaceMeshNv21Image.tryFromYuv420Planes(
        width: width,
        height: height,
        yPlane: FaceMeshImagePlane(
          bytes: sequential(width * height),
          bytesPerRow: width,
        ),
        uPlane: FaceMeshImagePlane(bytes: Uint8List(3), bytesPerRow: 2),
        vPlane: FaceMeshImagePlane(bytes: Uint8List(4), bytesPerRow: 2),
      );

      expect(image, isNull);
    });

    test('odd dimensions return null', () {
      final FaceMeshNv21Image? image = FaceMeshNv21Image.tryFromYuv420Planes(
        width: 3,
        height: 4,
        yPlane: FaceMeshImagePlane(bytes: Uint8List(12), bytesPerRow: 3),
        uPlane: FaceMeshImagePlane(bytes: Uint8List(2), bytesPerRow: 1),
        vPlane: FaceMeshImagePlane(bytes: Uint8List(2), bytesPerRow: 1),
      );

      expect(image, isNull);
    });
  });
}
