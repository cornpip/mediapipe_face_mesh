# Bundled TensorFlow Lite runtime: iOS device slice

How `ios/mediapipe_face_mesh/Frameworks/TensorFlowLiteC.xcframework/ios-arm64/libTensorFlowLiteC.a`
is built. The other Apple slices (iOS simulator, macOS) were built earlier
from the same kind of target and are not covered by this recipe yet.

The runtime is the TensorFlow Lite C API built inside the MediaPipe Bazel
workspace, with two changes from a stock build:

- `create_op_resolver.cc` replaces the default `tflite::CreateOpResolver()`
  so the MediaPipe custom operators are registered.
- XNNPACK's Arm SME, SME2, and KleidiAI kernels are compiled out. iPhone 16
  and 17 (A18, A19) select those kernels and one of them executes an SVE
  instruction outside streaming mode, which raises `EXC_BAD_INSTRUCTION`.

## Requirements

- macOS with Xcode and `bazelisk`. The workspace pins its Bazel version in
  `.bazelversion`.
- A JDK that Bazel can find. If none is installed system-wide, point
  `JAVA_HOME` at one for the build, for example the JDK inside Android
  Studio.

## Build

MediaPipe commit used: `e2f6a19ba4de1d21780d7e73bc101ce0d9b3306e`
(2026-07-13). It pins XNNPACK `68167d1` and TensorFlow `fad6b3c`.

```
git init mediapipe && cd mediapipe
git remote add origin https://github.com/google-ai-edge/mediapipe.git
git fetch --depth 1 origin e2f6a19ba4de1d21780d7e73bc101ce0d9b3306e
git checkout FETCH_HEAD

mkdir mpfm_runtime
cp <this dir>/BUILD <this dir>/create_op_resolver.cc mpfm_runtime/

bazelisk build -c opt --config=ios_arm64 \
  --define=xnn_enable_arm_sme=false \
  --define=xnn_enable_arm_sme2=false \
  --define=xnn_enable_kleidiai=false \
  //mpfm_runtime:TensorFlowLiteC_framework
```

The output is `bazel-bin/mpfm_runtime/TensorFlowLiteC_framework.zip`, a
static framework whose binary is an archive of every object file.

All three defines are needed at this XNNPACK commit. Turning off only the
two SME defines and keeping KleidiAI does not compile
(`gemm-config.c: use of undeclared identifier 'XNN_ENABLE_ARM_SME'`), so
KleidiAI's NEON kernels for quantized models are lost along with its SME
ones. Revisit this when the pinned XNNPACK moves.

## Hide symbols and pack

Only the C API stays exported. Everything else is made local so it cannot
collide with another copy of TensorFlow Lite's internals.

```
unzip bazel-bin/mpfm_runtime/TensorFlowLiteC_framework.zip -d out && cd out
printf '_TfLite*\n*AcquireFlexDelegate*\n' > allowlist.txt
xcrun ld -r -arch arm64 -all_load TensorFlowLiteC.framework/TensorFlowLiteC \
  -exported_symbols_list allowlist.txt -x -o TensorFlowLiteC.o
xcrun libtool -static -o libTensorFlowLiteC.a TensorFlowLiteC.o
```

## Verify

No SME instruction may remain, and the exported symbols must be the same
list as in the binary being replaced.

```
OD=$(xcrun -f llvm-objdump)
$OD -d --no-show-raw-insn --mattr=+sve,+sme,+sme2 libTensorFlowLiteC.a \
  | grep -c -w -E 'smstart|smstop|addvl|rdsvl|fmopa|smopa|bfmopa'   # 0

OLD=<repo>/ios/mediapipe_face_mesh/Frameworks/TensorFlowLiteC.xcframework/ios-arm64/libTensorFlowLiteC.a
diff <(nm -gU "$OLD" | awk '$2=="T"{print $3}' | sort) \
     <(nm -gU libTensorFlowLiteC.a | awk '$2=="T"{print $3}' | sort)   # no output
```

## After replacing the binary

- Run `tool/prefix_apple_tflite_symbols.sh`. A freshly built archive exports
  the original `TfLite*` names, and the plugin's headers expect the `mpfm_`
  ones. Run the Verify step above before this one, since it compares
  unprefixed names.
- Re-sync `Headers/` in every xcframework slice from `src/include` if the
  TensorFlow commit changed, and keep the slices' headers identical.
- Update `THIRD_PARTY_NOTICES.md` and the attribution prose in `NOTICES`.
- Add the CHANGELOG bullet.
