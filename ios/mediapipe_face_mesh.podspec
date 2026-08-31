#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint mediapipe_face_mesh.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'mediapipe_face_mesh'
  s.version          = '2.9.0'
  s.summary          = 'MediaPipe Face Mesh for Flutter.'
  s.description      = <<-DESC
Real-time face mesh detection for Flutter with bundled MediaPipe face mesh,
face detector, and TensorFlow Lite runtime binaries for Android and iOS.
                       DESC
  s.homepage         = 'https://github.com/cornpip/mediapipe_face_mesh.git'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'mediapipe_face_mesh contributors' => 'cornpip7777@gmail.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  # `.cc` files are pulled in via the ObjC++ forwarders in Classes/ to avoid
  # double compilation; only headers are exposed here.
  s.source_files = 'Classes/**/*', '../src/**/*.h'
  s.dependency 'Flutter'
  # The arm64 simulator slice of TensorFlowLiteC.xcframework has a minimum of
  # iOS 14.0 (an Apple constraint for arm64 simulators). Apps targeting less
  # than that still link against it; the linker may warn about the newer
  # minimum.
  s.platform = :ios, '13.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'HEADER_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)/../src/include" $(inherited)'
  }
  s.swift_version = '5.0'

  # Bundle the TensorFlow Lite C runtime copied into ios/Frameworks.
  # An xcframework, so the device slice (ios-arm64) and the simulator slice
  # (ios-arm64_x86_64-simulator) can coexist. A plain fat framework cannot hold
  # both: arm64 device and arm64 simulator differ by platform, not by arch.
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteC.xcframework'
end
