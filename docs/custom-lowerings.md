# Custom Lowerings

When your model contains ops that coreai-onnx does not support — custom-domain ops,
experimental ONNX ops, or proprietary layers — you can register a **custom lowering**
to teach the converter how to translate them.

## The `register_onnx_lowering` decorator

```python
from pathlib import Path

import onnx
import coreai_onnx
from coreai._compiler.dialects import coreai
from coreai._compiler.ir import Location, Value

converter = coreai_onnx.OnnxConverter()

@converter.register_onnx_lowering("mycompany::ChannelShuffle")
def lower_channel_shuffle(
    values_map: dict[str, Value],
    node: onnx.NodeProto,
    loc: Location,
) -> Value | list[Value]:
    # Pull the single input tensor.
    x = values_map[node.input[0]]

    # Read any op attributes you need.
    groups = next(
        (a.i for a in node.attribute if a.name == "groups"),
        2,  # default
    )

    # Emit Core AI ops and return the result Value(s).
    # Here we decompose ChannelShuffle as reshape → transpose → reshape.
    # (Shapes are symbolic; use coreai ops that accept dynamic dims.)
    reshaped = coreai.reshape(x, [...], loc=loc)
    transposed = coreai.transpose(reshaped, perm=[0, 2, 1, 3], loc=loc)
    out = coreai.reshape(transposed, [...], loc=loc)
    return out  # single Value; wrap in a list for multi-output ops

# Now convert as usual.
converter.add_onnx_model("model.onnx", entrypoint_name="main")
ai_program = converter.to_coreai()
ai_program.save_asset(Path("model.aimodel"))
```

## Lowering function signature

```
(values_map, node, loc) -> Value | list[Value]
```

| Parameter    | Type                          | Description |
|--------------|-------------------------------|-------------|
| `values_map` | `dict[str, Value]`            | Maps ONNX value names to live Core AI `Value`s; write outputs here (the converter does it automatically when you return). |
| `node`       | `onnx.NodeProto`              | The ONNX node being lowered. Read `node.input`, `node.output`, and `node.attribute`. |
| `loc`        | `coreai._compiler.ir.Location`| Source location for diagnostics; pass through to every Core AI op you emit. |

The function must return:

- A single `Value` if the node has one output.
- A `list[Value]` with one entry per non-empty output slot if the node has multiple
  outputs.

## Reserved domains

The following domains **cannot** be overridden:

| Domain      | Reason |
|-------------|--------|
| `ai.onnx`   | Built-in ONNX standard ops — all handled internally |
| `coreai`    | Core AI dialect — not valid as an ONNX domain |

Attempting to register a lowering for either domain raises `ValueError`.

## Overriding a built-in lowering

Pass `allow_override=True` to replace an existing lowering (useful for patching
incorrect behaviour in the interim):

```python
@converter.register_onnx_lowering("Relu", allow_override=True)
def patched_relu(values_map, node, loc):
    x = values_map[node.input[0]]
    return coreai.relu(x, loc=loc)
```

## Importing the Core AI dialect

All Core AI ops live in `coreai._compiler.dialects.coreai`:

```python
from coreai._compiler.dialects import coreai
```

Consult the `coreai-core` package for the full op catalogue. The `Value` and
`Location` types are in `coreai._compiler.ir`.

## UnsupportedOpError — aggregated report

If any op in the model has no lowering (built-in or custom), conversion fails with
`UnsupportedOpError` **before emitting any IR**. The error message aggregates every
missing op into a single report so you can fix them all at once:

```
UnsupportedOpError: The following ONNX ops have no Core AI lowering:
  mycompany::ChannelShuffle (3 node(s), e.g. node_4, node_9, node_14)
  mycompany::FusedGELU (1 node(s), e.g. node_22)

Register a custom lowering to proceed:
    @converter.register_onnx_lowering("mycompany::ChannelShuffle")
    def lower(values_map, node, loc): ...
Run `coreai-onnx inspect <model>` for a full coverage report.
```

Use `coreai-onnx inspect model.onnx` before converting to get a full histogram of
which ops are present and whether each is covered. See the [CLI reference](cli.md)
for details.
