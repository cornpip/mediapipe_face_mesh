"""Builds the tiny models embedded in integration_test/interpreter_test.dart.

Each model is one ADD operator, adding its input to itself or its two inputs
to each other. Prints the models as base64. Needs only the `flatbuffers` package:

    python tool/make_interpreter_test_models.py
"""

import base64

import flatbuffers

# TensorType values from the TFLite schema.
INT32 = 2
UINT8 = 3
# BuiltinOptions union value of AddOptions.
ADD_OPTIONS = 11


def _int32_vector(b, values):
    b.StartVector(4, len(values), 4)
    for value in reversed(values):
        b.PrependInt32(value)
    return b.EndVector()


def _offset_vector(b, offsets):
    b.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        b.PrependUOffsetTRelative(offset)
    return b.EndVector()


def _quantization(b, scale, zero_point):
    b.StartVector(4, 1, 4)
    b.PrependFloat32(scale)
    scales = b.EndVector()
    b.StartVector(8, 1, 8)
    b.PrependInt64(zero_point)
    zero_points = b.EndVector()
    b.StartObject(4)
    b.PrependUOffsetTRelativeSlot(2, scales, 0)
    b.PrependUOffsetTRelativeSlot(3, zero_points, 0)
    return b.EndObject()


def _tensor(b, name, tensor_type, shape, signature, quantization):
    name_offset = b.CreateString(name)
    shape_offset = _int32_vector(b, shape)
    signature_offset = _int32_vector(b, signature)
    quantization_offset = (
        _quantization(b, *quantization) if quantization else None
    )
    b.StartObject(8)
    b.PrependUOffsetTRelativeSlot(0, shape_offset, 0)
    b.PrependInt8Slot(1, tensor_type, 0)
    b.PrependUint32Slot(2, 0, 0)
    b.PrependUOffsetTRelativeSlot(3, name_offset, 0)
    if quantization_offset is not None:
        b.PrependUOffsetTRelativeSlot(4, quantization_offset, 0)
    b.PrependUOffsetTRelativeSlot(7, signature_offset, 0)
    return b.EndObject()


def build(
    tensor_type, in_quantization=None, out_quantization=None, two_inputs=False
):
    b = flatbuffers.Builder(512)

    # One empty buffer: index 0 means "no data".
    b.StartObject(1)
    buffers = _offset_vector(b, [b.EndObject()])

    # ADD is builtin code 0, the default of every OperatorCode field.
    b.StartObject(4)
    operator_codes = _offset_vector(b, [b.EndObject()])

    names = ["x", "y"] if two_inputs else ["x"]
    tensors = _offset_vector(
        b,
        [
            _tensor(b, name, tensor_type, [1, 3], [1, -1], in_quantization)
            for name in names
        ]
        + [_tensor(b, "sum", tensor_type, [1, 3], [1, -1], out_quantization)],
    )
    sum_index = len(names)

    b.StartObject(1)
    add_options = b.EndObject()
    op_inputs = _int32_vector(b, [0, 1] if two_inputs else [0, 0])
    op_outputs = _int32_vector(b, [sum_index])
    b.StartObject(5)
    b.PrependUint32Slot(0, 0, 0)
    b.PrependUOffsetTRelativeSlot(1, op_inputs, 0)
    b.PrependUOffsetTRelativeSlot(2, op_outputs, 0)
    b.PrependUint8Slot(3, ADD_OPTIONS, 0)
    b.PrependUOffsetTRelativeSlot(4, add_options, 0)
    operators = _offset_vector(b, [b.EndObject()])

    graph_inputs = _int32_vector(b, list(range(sum_index)))
    graph_outputs = _int32_vector(b, [sum_index])
    graph_name = b.CreateString("main")
    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(0, tensors, 0)
    b.PrependUOffsetTRelativeSlot(1, graph_inputs, 0)
    b.PrependUOffsetTRelativeSlot(2, graph_outputs, 0)
    b.PrependUOffsetTRelativeSlot(3, operators, 0)
    b.PrependUOffsetTRelativeSlot(4, graph_name, 0)
    subgraphs = _offset_vector(b, [b.EndObject()])

    description = b.CreateString("interpreter test model")
    b.StartObject(5)
    b.PrependUint32Slot(0, 3, 0)
    b.PrependUOffsetTRelativeSlot(1, operator_codes, 0)
    b.PrependUOffsetTRelativeSlot(2, subgraphs, 0)
    b.PrependUOffsetTRelativeSlot(3, description, 0)
    b.PrependUOffsetTRelativeSlot(4, buffers, 0)
    b.Finish(b.EndObject(), file_identifier=b"TFL3")
    return bytes(b.Output())


if __name__ == "__main__":
    models = {
        "int32 add": build(INT32),
        # real = (q - 3) * 0.5 in, (q - 6) * 1.0 out, so the output is q + 3.
        "uint8 quantized add": build(UINT8, (0.5, 3), (1.0, 6)),
        "int32 add of two inputs": build(INT32, two_inputs=True),
    }
    for name, data in models.items():
        print(f"{name} ({len(data)} bytes)")
        print(base64.b64encode(data).decode())
