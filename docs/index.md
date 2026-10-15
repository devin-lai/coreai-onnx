# coreai-onnx

**coreai-onnx** converts ONNX models into Apple's Core AI `.aimodel` format for
on-device inference on Apple silicon. It supports 143 built-in ONNX operator
lowerings, graph optimization, and numerical verification against ONNX Runtime.

Convert models exported from PyTorch and other ONNX-compatible frameworks through
the Python API, command-line interface (CLI), or Model Context Protocol (MCP)
server. Conversion runs on macOS, Linux, and Windows; executing `.aimodel` bundles
requires macOS 27+ / iOS 27+ with the Core AI framework.

[Get started](quickstart.md) · [Supported ONNX operators](coverage.md) ·
[MCP server](mcp.md) · [GitHub](https://github.com/devin-lai/coreai-onnx) ·
[PyPI](https://pypi.org/project/coreai-onnx/)

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
cli
agents
mcp
coverage
custom-lowerings
repair
```
