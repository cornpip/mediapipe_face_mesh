#include <flutter/dart_project.h>
#include <flutter/flutter_view_controller.h>
#include <windows.h>

#include <algorithm>

#include "flutter_window.h"
#include "utils.h"

int APIENTRY wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE prev,
                      _In_ wchar_t *command_line, _In_ int show_command) {
  // Attach to console when present (e.g., 'flutter run') or create a
  // new console when running with a debugger.
  if (!::AttachConsole(ATTACH_PARENT_PROCESS) && ::IsDebuggerPresent()) {
    CreateAndAttachConsole();
  }

  // Initialize COM, so that it is available for use in the library and/or
  // plugins.
  ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

  flutter::DartProject project(L"data");

  std::vector<std::string> command_line_arguments =
      GetCommandLineArguments();

  project.set_dart_entrypoint_arguments(std::move(command_line_arguments));

  FlutterWindow window(project);
  // Portrait window centered on the primary monitor. Win32Window::Create
  // scales logical coordinates by the monitor DPI, so convert the physical
  // screen metrics back to logical units before centering.
  Win32Window::Size size(1200, 1000);
  const UINT dpi = ::GetDpiForSystem();
  const int screen_width = ::GetSystemMetrics(SM_CXSCREEN) * 96 / dpi;
  const int screen_height = ::GetSystemMetrics(SM_CYSCREEN) * 96 / dpi;
  Win32Window::Point origin(
      std::max(0, (screen_width - static_cast<int>(size.width)) / 2),
      std::max(0, (screen_height - static_cast<int>(size.height)) / 2));
  if (!window.Create(L"mediapipe_face_mesh_example", origin, size)) {
    return EXIT_FAILURE;
  }
  window.SetQuitOnClose(true);

  ::MSG msg;
  while (::GetMessage(&msg, nullptr, 0, 0)) {
    ::TranslateMessage(&msg);
    ::DispatchMessage(&msg);
  }

  ::CoUninitialize();
  return EXIT_SUCCESS;
}
