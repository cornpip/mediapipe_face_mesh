#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint mediapipe_face_mesh.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'mediapipe_face_mesh'
  s.version          = '3.1.0'
  s.summary          = 'MediaPipe Face Mesh for Flutter.'
  s.description      = <<-DESC
Real-time face mesh detection for Flutter with bundled MediaPipe face mesh,
face detector, and TensorFlow Lite runtime binaries.
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
  s.dependency 'FlutterMacOS'
  # The arm64 slice of TensorFlowLiteC.xcframework has a minimum of macOS
  # 11.0 (the first release for Apple Silicon); the x86_64 slice has 10.15.
  s.platform = :osx, '10.15'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'HEADER_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)/../src/include" $(inherited)'
  }
  s.swift_version = '5.0'

  # The TensorFlow Lite runtime references CoreFoundation (time zone lookup in
  # its bundled abseil).
  s.frameworks = 'CoreFoundation'

  # Bundle the TensorFlow Lite C runtime copied into macos/Frameworks: a static
  # universal (arm64 + x86_64) framework, linked into the plugin framework.
  s.vendored_frameworks = 'Frameworks/TensorFlowLiteC.xcframework'
end
