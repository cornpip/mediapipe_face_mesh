# Roadmap

## API

### Multi-Face Mesh Helpers
- Add convenience APIs for running face mesh inference over multiple detector results.
- Return `List<FaceMeshResult>` from detector-driven multi-face flows.

### Delegate Diagnostics
- Expose the active delegate selected by each processor after fallback handling.
- Make it easier to confirm whether CPU, XNNPACK, or GPU V2 is actually being used.
