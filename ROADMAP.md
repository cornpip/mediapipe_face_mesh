# Roadmap

## Models

### Face Mesh
- Add iris landmark model support for refined eye and iris points.

### Face Detection
- Add full-range dense face detection model support.
- Add full-range sparse face detection model support.

## API

### High-Level Inference API
- Add a high-level API that combines face detection and face mesh inference in one flow.
- Expose the detector result, selected ROI/box, and mesh result through one unified interface.

### Multi-Face Mesh Helpers
- Add convenience APIs for running face mesh inference over multiple detector results.
- Return `List<FaceMeshResult>` from detector-driven multi-face flows.

### Delegate Diagnostics
- Expose the active delegate selected by each processor after fallback handling.
- Make it easier to confirm whether CPU, XNNPACK, or GPU V2 is actually being used.
