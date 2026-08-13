# Bench design notes and pitfalls

Deeper background for bench/README.md: why the suite is shaped this way,
and the pitfalls that cost time while building it. Read this before
changing the harness or interpreting surprising numbers.

## Design decisions

- **One app per package.** Each measured package bundles its own TFLite
  native binaries; putting two in one app risks symbol/loader collisions
  and pollutes measurement (shared thread pools, GPU context, allocator
  state). Separate apps also make the app-size delta measurable at all.
- **Identical assets in every app.** Asset weight then cancels out of
  every size comparison against the template baseline app; deltas are pure
  package weight. (No baseline app is checked in; `flutter create` a
  fresh one to measure size deltas.)
- **Assets are generated, not committed.** Only the source video and two
  photos are committed; `tool/prepare_assets.py` extracts the 300 frames
  and copies assets into each app (all gitignored). New checkout: run it
  once (needs opencv-python).
- **Stats over stdout.** Tests print `BENCH_JSON {...}` lines;
  `tool/aggregate.py` folds any number of run logs into one markdown
  table. No result files to sync from devices.
- **`activeDelegate` is recorded in every result.** Delegate requests can
  silently fall back to CPU; without recording what actually ran, a "GPU"
  number can be a CPU number with a wrong label. This caught exactly that
  once.

## Pitfalls (each of these cost real time)

1. **`flutter test integration_test` builds the app in debug.** The
   plugin's CMake now forces optimization for its own C++ regardless of
   build type, but the Dart side still runs JIT in debug. Confirm headline
   numbers with a release/profile run before publishing anywhere.
   Flutter's profile mode also configures plugin CMake as Debug, so
   without the forced flags, profile runs measured -O0 native code.
2. **Single-image numbers swing with device thermal state** (observed
   2.6 to 4.5ms for the same config across sessions on the same device).
   Streaming steady-state is stable run-to-run (1.46 to 1.51ms across
   every session). Treat streaming as the headline metric; treat
   single-image deltas under ~2x with suspicion unless measured
   back-to-back.
3. **PowerShell `Tee-Object` writes UTF-16 logs.** `aggregate.py` detects
   the BOM and handles it; if you parse logs with anything else, convert
   first.
4. **`flutter test` occasionally fails at teardown** with a
   `PathNotFoundException` on a `flutter_tools.*` temp path while all
   real work succeeded. Rerun; it is tooling flakiness, not a bench
   failure.
5. **Keep decode and IO outside the stopwatch.** Prepare input buffers
   (decode, color conversion) before the measured call, and be aware that
   some APIs cache decode across identical inputs, so a same-image loop
   may not measure what its input type suggests. Use distinct images if
   decode cost matters to the comparison.
6. **The GPU delegate is intentionally out of the matrix.** The bundled
   runtimes do not export the GPU delegate symbols (`gpuV2` always falls
   back to CPU; deprecated, removed in 3.0.0). A one-off experiment with a
   GPU-enabled runtime measured the 192x192 mesh several times slower on
   GPU than CPU/XNNPACK on a 2025 flagship: dispatch/transfer overhead
   dominates at this model size.
7. **cpu vs xnnpack parity is expected, not a bug.** The bundled runtime
   is built with XNNPACK and TFLite applies it automatically in cpu mode,
   so both delegates execute the same kernels. The matrix keeps one
   xnnpack config only to demonstrate the parity.
8. **Lock files:** the bench apps' pubspec.locks are independent of the
   package's own example lock; regenerating them with a current Flutter is
   fine and does not need to match the package's SDK pin.

## Resuming on a fresh machine

1. `python3 bench/tool/prepare_assets.py` (one-time; needs opencv-python)
2. `flutter pub get` in each of `mine/ mlkit/ fdt/`
   (first fdt build also downloads opencv_dart binaries and is slow)
3. Run per bench/README.md; device-bound numbers are only comparable on
   the same device model.
