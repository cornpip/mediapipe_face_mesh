// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "mediapipe_face_mesh",
    platforms: [
        .iOS("13.0")
    ],
    products: [
        // Dynamic, so the plugin stays its own framework and Dart opens it by name.
        .library(name: "mediapipe-face-mesh", type: .dynamic, targets: ["mediapipe_face_mesh"])
    ],
    dependencies: [
        .package(name: "FlutterFramework", path: "../FlutterFramework")
    ],
    targets: [
        .target(
            name: "mediapipe_face_mesh",
            dependencies: [
                .product(name: "FlutterFramework", package: "FlutterFramework"),
                "TensorFlowLiteC",
            ],
            linkerSettings: [
                .linkedFramework("CoreFoundation"),
                .linkedFramework("Foundation")
            ]
        ),
        .binaryTarget(
            name: "TensorFlowLiteC",
            path: "Frameworks/TensorFlowLiteC.xcframework"
        )
    ],
    cxxLanguageStandard: .cxx17
)
