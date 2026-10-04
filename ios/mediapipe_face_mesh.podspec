#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint mediapipe_face_mesh.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'mediapipe_face_mesh'
  s.version          = '3.2.1'
  s.summary          = 'MediaPipe Face Mesh for Flutter.'
  s.description      = <<-DESC
Real-time face mesh detection for Flutter with bundled MediaPipe face mesh,
face detector, and TensorFlow Lite runtime binaries.
                       DESC
  s.homepage         = 'https://github.com/cornpip/mediapipe_face_mesh.git'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'mediapipe_face_mesh contributors' => 'cornpip7777@gmail.com' }

  # Sources/ is shared with Package.swift: ObjC++ forwarders that include
  # src/*.cc by relative path, since neither build system compiles files
  # outside the plugin directory. Only headers are listed from src/.
  s.source           = { :path => '.' }
  s.source_files = 'mediapipe_face_mesh/Sources/mediapipe_face_mesh/**/*', '../src/*.h'
  s.dependency 'Flutter'
  # The arm64 simulator slice of TensorFlowLiteC.xcframework has a minimum of
  # iOS 14.0 (an Apple constraint for arm64 simulators). Apps targeting less
  # than that still link against it; the linker may warn about the newer
  # minimum.
  s.platform = :ios, '13.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }
  s.swift_version = '5.0'

  # Referenced by the bundled TensorFlow Lite runtime.
  s.frameworks = 'CoreFoundation', 'Foundation'

  # Bundle the TensorFlow Lite C runtime as a library xcframework with its
  # headers, so the device slice (ios-arm64) and the simulator slice
  # (ios-arm64_x86_64-simulator) can coexist. A plain fat framework cannot hold
  # both: arm64 device and arm64 simulator differ by platform, not by arch.
  s.vendored_frameworks = 'mediapipe_face_mesh/Frameworks/TensorFlowLiteC.xcframework'
end
