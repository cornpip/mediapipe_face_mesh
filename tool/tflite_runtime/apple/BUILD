load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")

package(default_visibility = ["//visibility:private"])

cc_library(
    name = "create_op_resolver",
    srcs = ["create_op_resolver.cc"],
    deps = [
        "//mediapipe/util/tflite:cpu_op_resolver",
        "@org_tensorflow//tensorflow/lite:framework_stable",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_lite_c",
    deps = [
        ":create_op_resolver",
        "@org_tensorflow//tensorflow/lite/c:c_api_experimental_without_op_resolver",
        "@org_tensorflow//tensorflow/lite/c:c_api_without_op_resolver",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@org_tensorflow//tensorflow/lite/profiling/telemetry/c:profiler",
        "@org_tensorflow//tensorflow/lite/profiling/telemetry/c:telemetry_setting",
    ],
)

# Build with the three --define flags from README.md. Without them the SME
# kernels are compiled back in:
#   bazel build -c opt --config=ios_arm64 \
#     --define=xnn_enable_arm_sme=false \
#     --define=xnn_enable_arm_sme2=false \
#     --define=xnn_enable_kleidiai=false \
#     //mpfm_runtime:TensorFlowLiteC_framework
ios_static_framework(
    name = "TensorFlowLiteC_framework",
    bundle_name = "TensorFlowLiteC",
    minimum_os_version = "12.0",
    deps = [":tensorflow_lite_c"],
)
