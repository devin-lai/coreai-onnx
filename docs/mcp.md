# MCP server

coreai-onnx ships an [MCP](https://modelcontextprotocol.io) server so agent
frameworks can call the converter as native tools.

## Install and configure

```bash
pip install "coreai-onnx[mcp]"
```

Add to your MCP client configuration (Claude Code, Claude Desktop, etc.):

```json
{
  "mcpServers": {
    "coreai-onnx": {
      "command": "coreai-onnx-mcp"
    }
  }
}
```

The server speaks stdio. It exposes four tools:

| Tool | Parameters | Wraps |
|---|---|---|
| `inspect_model` | `model_path` | `coreai-onnx inspect` |
| `convert_model` | `model_path`, `output_path`, `optimize=true`, `validate=true`, `verify=true`, `rtol`, `atol`, `min_psnr`, `compute_unit`, `seed`, `entrypoint` | `coreai-onnx convert` |
| `verify_model` | `model_path`, `aimodel_path`, `rtol`, `atol`, `min_psnr`, `compute_unit`, `seed`, `entrypoint` | `coreai-onnx verify` |
| `get_schema` | — | `coreai-onnx schema` |

## One contract

Every tool returns the same envelope the CLI emits with `--json`
(`schema_version`, `command`, `status`, `result`, `warnings`, `error`) — see
[Machine-readable output](cli.md). Domain failures (unsupported ops, bad
files, precision failures) come back as `status: "error"` envelopes with the
documented [error codes](cli.md), never as MCP protocol errors; branch on
`error.code` exactly as you would for the CLI. Exit codes do not exist over
MCP — `status`/`error.code` carry the same information. Boolean parameters
map to the CLI's negative flags (`optimize=false` ≙ `--no-optimize`);
`entrypoint` ≙ `--name`.
