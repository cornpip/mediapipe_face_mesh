# mediapipe_face_mesh

Flutter FFI 플러그인으로 [MediaPipe Face Mesh](https://developers.google.com/mediapipe/solutions/vision/face_mesh)를 CPU 상에서 추론할 수 있도록 구성했습니다.  
TFLite 모델 로딩/추론은 네이티브 C++에서 처리하고, 입력 영상 전처리(ROI 정규화, 192×192 RGB 텐서 생성)와 추론 결과 후처리(468개 랜드마크 복원, ROI 업데이트) 또한 MediaPipe에서 공개한 로직을 참고하여 구현했습니다.

> **중요**  
> 저장소에는 빈 `mediapipe_face_mesh.tflite` 자리표시자만 포함되어 있습니다. 실제 MediaPipe Face Mesh 모델과 TensorFlow Lite 런타임(`libtensorflowlite_c`) 라이브러리를 각 플랫폼에 맞게 추가해야 실행할 수 있습니다.

## 준비 사항

1. **Face Mesh TFLite 모델 배치**  
   - `assets/models/mediapipe_face_mesh.tflite` 파일을 실제 모델로 교체하세요.  
   - 앱에서 다른 경로의 모델을 사용하고 싶다면 `MediapipeFaceMesh.create(modelFilePath: '...')` 또는 `modelAssetPath` 옵션을 활용할 수 있습니다.

2. **TensorFlow Lite C 런타임 제공**  
   - **Android**: 각 ABI(`arm64-v8a`, `armeabi-v7a` 등)에 맞는 `libtensorflowlite_c.so`를 `android/src/main/jniLibs/<abi>/` 폴더에 넣어 함께 배포하세요.  
   - **iOS**: `libtensorflowlite_c.dylib` 또는 `TensorFlowLiteC.xcframework`를 Pod 타겟에 링크해야 합니다.
   - 옵션으로 `MediapipeFaceMesh.create(tfliteLibraryPath: '/path/to/libtensorflowlite_c.so')` 로 직접 경로를 전달할 수 있습니다.

3. **카메라 프레임/이미지 준비**  
   - 전처리는 RGBA 또는 BGRA `Uint8List` 버퍼를 입력으로 받습니다.  
   - ROI가 없다면 기본적으로 전체 프레임을 대상으로 Face Mesh 추적을 시작합니다.

## 사용 방법

```dart
final MediapipeFaceMesh mesh = await MediapipeFaceMesh.create(
  threads: 2,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
);

final FaceMeshImage image = FaceMeshImage(
  pixels: rgbaBytes,   // width * height * 4 바이트
  width: frameWidth,
  height: frameHeight,
  pixelFormat: FaceMeshPixelFormat.rgba,
);

final FaceMeshResult result = mesh.process(image);
for (final landmark in result.landmarks) {
  debugPrint('x=${landmark.x}, y=${landmark.y}, z=${landmark.z}');
}

mesh.close();
```

`FaceMeshResult` 는

- 468개의 3D 랜드마크(`x`, `y`는 입력 이미지 기준 정규화 좌표, `z`는 ROI 폭 기준 깊이),
- MediaPipe `NormalizedRect` 형태의 최신 ROI,
- 모델 confidence score

를 포함합니다.

## Example

`example/` 앱은 플러그인을 초기화하고 더미 RGBA 버퍼를 넣어 호출하는 흐름을 보여줍니다. 실제 추론을 위해서는 위 준비 사항을 충족해야 합니다. 실패 시 UI 상단에 에러 메시지가 표시됩니다.
