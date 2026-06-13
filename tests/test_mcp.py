# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""MCP server tests, driven over real in-memory client sessions. The tools
must return byte-for-byte the same envelope contract as the CLI's --json
mode; see docs/mcp.md."""

import json

import pytest

from coreai_onnx._mcp import _build_server
from coreai_onnx._mcp import main as mcp_main
from tests.helpers import (
    COREAI_CONVERSION_MARKS,
    ENVELOPE_KEYS,
    coreai_runtime_test,
    det_model_file,
    relu_model_file,
)

pytestmark = [*COREAI_CONVERSION_MARKS]

_mcp_memory = pytest.importorskip("mcp.shared.memory")
connect = _mcp_memory.create_connected_server_and_client_session


async def _call(name: str, args: dict) -> dict:
    """Call one tool over an in-memory session; assert envelope shape."""
    server = _build_server()
    async with connect(server._mcp_server) as client:
        res = await client.call_tool(name, args)
    assert res.isError is False, f"unexpected protocol error: {res.content}"
    env = res.structuredContent
    assert env is not None
    assert set(env) == ENVELOPE_KEYS
    return env


async def test_list_tools_exposes_four_tools():
    server = _build_server()
    async with connect(server._mcp_server) as client:
        tools = (await client.list_tools()).tools
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {
        "inspect_model",
        "convert_model",
        "verify_model",
        "get_schema",
    }
    convert_props = by_name["convert_model"].inputSchema["properties"]
    assert {
        "model_path",
        "output_path",
        "optimize",
        "validate",
        "verify",
        "repair",
        "rtol",
        "atol",
        "seed",
        "entrypoint",
    } <= set(convert_props)
    # every tool must carry a non-empty description (it's the agent-facing doc)
    assert all(t.description for t in tools)


async def test_convert_model_repair_promotes_float16(tmp_path):
    """The repair flag surfaces the same auto-repair as the CLI --repair."""
    import numpy as np
    import onnx

    from tests.helpers import single_op_model

    model = single_op_model(
        "Mul",
        {"x": np.zeros((2, 3), dtype=np.float16)},
        initializers={"w": np.ones((3,), dtype=np.float16)},
    )
    model_path = str(tmp_path / "fp16.onnx")
    onnx.save(model, model_path)

    env = await _call(
        "convert_model",
        {
            "model_path": model_path,
            "output_path": str(tmp_path / "out.aimodel"),
            "repair": True,
        },
    )
    assert env["status"] == "ok", env["error"]
    assert [r["name"] for r in env["result"]["repairs"]] == [
        "promote_float16_to_float32"
    ]


async def test_inspect_model_convertible(tmp_path):
    env = await _call("inspect_model", {"model_path": relu_model_file(tmp_path)})
    assert env["command"] == "inspect"
    assert env["status"] == "ok"
    assert env["result"]["convertible"] is True


async def test_inspect_model_unconvertible_is_ok_status(tmp_path):
    env = await _call("inspect_model", {"model_path": det_model_file(tmp_path)})
    assert env["status"] == "ok"
    assert env["result"]["convertible"] is False
    assert "Det" in env["result"]["unsupported"]


async def test_convert_model_unsupported_ops_envelope_not_protocol_error(tmp_path):
    env = await _call(
        "convert_model",
        {
            "model_path": det_model_file(tmp_path),
            "output_path": str(tmp_path / "out.aimodel"),
        },
    )
    assert env["command"] == "convert"
    assert env["status"] == "error"
    assert env["error"]["code"] == "unsupported_ops"
    assert env["error"]["hint"]


async def test_convert_model_missing_file_is_io_error(tmp_path):
    env = await _call(
        "convert_model",
        {
            "model_path": str(tmp_path / "missing.onnx"),
            "output_path": str(tmp_path / "out.aimodel"),
        },
    )
    assert env["status"] == "error"
    assert env["error"]["code"] == "io_error"


async def test_get_schema_matches_cli_schema():
    env = await _call("get_schema", {})
    assert env["command"] == "schema"
    r = env["result"]
    assert {c["name"] for c in r["commands"]} == {
        "convert",
        "inspect",
        "verify",
        "schema",
    }
    assert len(r["supported_ops"]) > 100


async def test_envelope_parity_with_cli_json(tmp_path, capsys):
    from coreai_onnx._cli import main as cli_main

    path = relu_model_file(tmp_path)
    assert cli_main(["inspect", path, "--json"]) == 0
    cli_env = json.loads(capsys.readouterr().out)
    mcp_env = await _call("inspect_model", {"model_path": path})
    assert mcp_env == cli_env


@coreai_runtime_test
async def test_convert_model_end_to_end(tmp_path):
    # Proves the asyncio.run-in-worker-thread design: the precision check
    # inside _run_convert calls asyncio.run, which would crash on the MCP
    # event-loop thread.
    out_path = tmp_path / "out.aimodel"
    env = await _call(
        "convert_model",
        {
            "model_path": relu_model_file(tmp_path),
            "output_path": str(out_path),
        },
    )
    assert env["status"] == "ok"
    assert env["result"]["precision"]["passed"] is True
    assert out_path.exists()


@coreai_runtime_test
async def test_verify_model_end_to_end(tmp_path):
    from coreai_onnx._cli import main as cli_main

    model_path = relu_model_file(tmp_path)
    out_path = str(tmp_path / "out.aimodel")
    assert cli_main(["convert", model_path, "-o", out_path, "--no-verify"]) == 0
    env = await _call(
        "verify_model", {"model_path": model_path, "aimodel_path": out_path}
    )
    assert env["command"] == "verify"
    assert env["status"] == "ok"
    assert env["result"]["passed"] is True


async def test_convert_ort_missing_warning_parity_with_cli(
    tmp_path, capsys, monkeypatch
):
    """MCP envelope warnings must include 'onnxruntime_missing' when ORT is
    absent and a domain error occurs — exactly matching the CLI's --json envelope
    (the 'same envelope' guarantee documented in _mcp.py's module docstring)."""
    import coreai_onnx._service as service
    from coreai_onnx._cli import main as cli_main

    monkeypatch.setattr(service, "_onnxruntime_available", lambda: False)

    path = det_model_file(tmp_path)
    out = str(tmp_path / "out.aimodel")

    # MCP path
    mcp_env = await _call(
        "convert_model",
        {"model_path": path, "output_path": out},
    )
    mcp_codes = [w["code"] for w in mcp_env["warnings"]]
    assert "onnxruntime_missing" in mcp_codes, (
        f"MCP envelope missing 'onnxruntime_missing'; warnings={mcp_env['warnings']}"
    )

    # CLI path — convert failed before writing anything, so reuse the same output path.
    cli_main(["convert", path, "-o", out, "--json"])
    cli_env = json.loads(capsys.readouterr().out)

    assert mcp_env == cli_env, (
        f"MCP envelope differs from CLI envelope.\n"
        f"MCP: {json.dumps(mcp_env, indent=2)}\n"
        f"CLI: {json.dumps(cli_env, indent=2)}"
    )


def test_main_without_mcp_extra_exits_2(monkeypatch, capsys):
    import coreai_onnx._mcp as m

    def _no_mcp():
        raise ImportError(
            'the MCP server requires the [mcp] extra: pip install "coreai-onnx[mcp]"'
        )

    monkeypatch.setattr(m, "_build_server", _no_mcp)
    rc = mcp_main()
    err = capsys.readouterr().err
    assert rc == 2
    assert "coreai-onnx[mcp]" in err
