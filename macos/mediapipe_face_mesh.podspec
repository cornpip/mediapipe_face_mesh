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

  # Sources/ is shared with Package.swift: ObjC++ forwarders that include
  # src/*.cc by relative path, since neither build system compiles files
  # outside the plugin directory. Only headers are listed from src/.
  s.source           = { :path => '.' }
  s.source_files = 'mediapipe_face_mesh/Sources/mediapipe_face_mesh/**/*', '../src/*.h'
  s.dependency 'FlutterMacOS'
  # The arm64 slice of TensorFlowLiteC.xcframework has a minimum of macOS
  # 11.0 (the first release for Apple Silicon); the x86_64 slice has 10.15.
  s.platform = :osx, '10.15'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES'
  }
  s.swift_version = '5.0'

  # Referenced by the bundled TensorFlow Lite runtime.
  s.frameworks = 'CoreFoundation'

  # Bundle the TensorFlow Lite C runtime: a static universal (arm64 +
  # x86_64) library xcframework with its headers, linked into the plugin
  # framework. Shared with the Swift package in mediapipe_face_mesh/.
  s.vendored_frameworks = 'mediapipe_face_mesh/Frameworks/TensorFlowLiteC.xcframework'
end
