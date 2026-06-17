# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Drift guards for the machine-readable contract, plus discoverability-asset
sanity checks. If these fail, the contract tables, the docs, and the emitting
code have diverged — fix the divergence, not the test."""

from pathlib import Path

import pytest

from coreai_onnx import _service

REPO_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTED_ERROR_CODES = {
    "unsupported_ops",
    "model_validation_failed",
    "conversion_failed",
    "compiler_failed",
    "precision_check_failed",
    "precision_check_error",
    "invalid_model_file",
    "io_error",
    "platform_unsupported",
}
DOCUMENTED_WARNING_CODES = {
    "onnxruntime_missing",
    "platform_no_runtime",
    "reference_nonfinite",
    "precision_benign_noise",
    "precision_hardware_divergence",
}
DOCUMENTED_EXIT_CODES = {0, 1, 2, 3}


def test_error_code_table_matches_contract():
    assert set(_service._ERROR_CODES) == DOCUMENTED_ERROR_CODES
    for code, entry in _service._ERROR_CODES.items():
        assert entry["meaning"], f"{code} has empty meaning"


def test_warning_code_table_matches_contract():
    assert set(_service._WARNING_CODES) == DOCUMENTED_WARNING_CODES
    assert all(_service._WARNING_CODES.values())


def test_exit_code_table_matches_contract():
    assert set(_service._EXIT_CODES) == DOCUMENTED_EXIT_CODES
    assert all(_service._EXIT_CODES.values())


def test_error_constructor_rejects_unknown_code():
    with pytest.raises(RuntimeError):
        _service._error("not_a_real_code", "boom")


def test_warning_constructor_rejects_unknown_code():
    with pytest.raises(RuntimeError):
        _service._warning("not_a_real_code", "boom")


def test_docs_cli_md_mentions_every_code():
    text = (REPO_ROOT / "docs" / "cli.md").read_text()
    for code in DOCUMENTED_ERROR_CODES | DOCUMENTED_WARNING_CODES:
        assert f"`{code}`" in text, f"docs/cli.md is missing {code}"
    for code in DOCUMENTED_EXIT_CODES:
        assert f"| `{code}` |" in text, f"docs/cli.md exit-code table is missing {code}"


def test_llms_txt_exists_and_links_resolve():
    import re

    text = (REPO_ROOT / "llms.txt").read_text()
    assert text.startswith("# coreai-onnx")
    assert "AGENTS.md" in text
    # Every docs-site page link must correspond to a real source page, so a
    # docs rename cannot silently 404 the agent-facing index.
    site = "https://devin-lai.github.io/coreai-onnx/"
    pages = re.findall(re.escape(site) + r"([\w-]+)\.html", text)
    assert pages, "llms.txt should link into the docs site"
    for page in pages:
        assert (REPO_ROOT / "docs" / f"{page}.md").exists(), (
            f"llms.txt links to {page}.html but docs/{page}.md does not exist"
        )


def test_schema_description_matches_pyproject():
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    import argparse

    outcome = _service._run_schema(argparse.Namespace())
    assert outcome.result is not None
    assert outcome.result["tool"]["description"] == data["project"]["description"]


def test_agents_md_covers_every_error_code():
    text = (REPO_ROOT / "AGENTS.md").read_text()
    for code in DOCUMENTED_ERROR_CODES:
        assert f"`{code}`" in text, f"AGENTS.md has no recovery guidance for {code}"
    for code in DOCUMENTED_WARNING_CODES:
        assert f"`{code}`" in text, f"AGENTS.md does not mention warning {code}"
    assert "schema --json" in text
    assert "exit code" in text.lower()


def test_pyproject_metadata_complete():
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = data["project"]
    classifiers = project["classifiers"]
    assert project["name"] == "coreai-onnx"
    assert project["version"] == "1.1.1"
    assert "Development Status :: 5 - Production/Stable" in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers
    assert "Programming Language :: Python :: 3.13" in classifiers
    assert project["requires-python"] == ">=3.11,<3.14"
    assert any(c.startswith("Topic ::") for c in classifiers)
    assert "onnx" in project["keywords"]
    urls = project["urls"]
    assert {"Homepage", "Documentation", "Repository", "Issues"} <= set(urls)
    assert data["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert data["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    test_deps = data["project"]["optional-dependencies"]["test"]
    assert any(dep.startswith("pytest-cov") for dep in test_deps)
    dev_deps = data["project"]["optional-dependencies"]["dev"]
    assert any(dep.startswith("vulture") for dep in dev_deps)
    lint_select = data["tool"]["ruff"]["lint"]["select"]
    assert {"B", "UP", "SIM", "C4", "PERF", "RET", "RUF", "PT"} <= set(lint_select)
    coverage = data["tool"]["coverage"]
    assert coverage["run"]["branch"] is True
    assert coverage["run"]["source"] == ["coreai_onnx"]
    assert coverage["report"]["show_missing"] is True
    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    assert "apple: Requires Apple Core AI / macOS 27 / Xcode 27 beta" in markers
    assert "coreai: Exercises Core AI conversion/compiler/runtime behavior" in markers
    assert "requires_macos27: Requires local macOS 27 / Xcode 27 beta" in markers
    assert (
        "integration: Integration tests that depend on local Apple SDK/runtime"
        in markers
    )


def test_sdist_manifest_includes_project_governance_files():
    text = (REPO_ROOT / "MANIFEST.in").read_text()
    for path in [
        ".editorconfig",
        ".gitattributes",
        ".pre-commit-config.yaml",
        "AGENTS.md",
        "CHANGELOG.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "llms.txt",
        "pyproject.toml",
    ]:
        assert f"include {path}" in text


def test_repository_has_line_ending_and_binary_attribute_policy():
    text = (REPO_ROOT / ".gitattributes").read_text()
    assert "* text=auto eol=lf" in text
    for pattern in ["*.onnx binary", "*.aimodel/** binary", "*.mlirb binary"]:
        assert pattern in text


def test_ci_runs_lightweight_cross_platform_checks_and_docs_on_prs():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "permissions:" in ci
    assert "contents: read" in ci
    assert "concurrency:" in ci
    assert "lightweight:" in ci
    assert ci.count("timeout-minutes:") == 1
    assert ci.count("cache: pip") == 1
    assert ci.count("cache-dependency-path: pyproject.toml") == 1
    assert "python -m pip install -e \".[test]\" build ruff twine" in ci
    assert "ruff check ." in ci
    assert "python -m compileall src tests" in ci
    assert 'pytest -m "not apple and not integration"' in ci
    assert "python -m build" in ci
    assert "twine check dist/*" in ci
    assert "pre-commit run --all-files" not in ci
    assert "mypy --ignore-missing-imports src/coreai_onnx" not in ci
    assert "vulture src/coreai_onnx tests --min-confidence 80" not in ci
    assert "python -m pip install dist/*.whl" not in ci
    assert "coreai-onnx schema --json" not in ci
    assert "coverage:" not in ci
    assert 'pytest -m "not slow"' not in ci
    assert "--cov=coreai_onnx" not in ci
    assert "--cov-report=term-missing" not in ci
    assert "--cov-report=xml" not in ci
    assert 'pytest -n auto -m "not slow"' not in ci
    assert 'pytest -n auto -m "slow"' not in ci
    assert "macos-latest" not in ci

    docs = (REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text()
    assert "pull_request:" in docs
    assert docs.count("timeout-minutes:") >= 2
    assert "pages: write" in docs
    assert "id-token: write" in docs
    assert "enablement: true" in docs
    assert "cache: pip" in docs
    assert "cache-dependency-path: pyproject.toml" in docs
    assert "sphinx-build -b html docs docs/_build/html" in docs
    assert "if: github.event_name == 'push'" in docs


def test_publish_workflow_uses_trusted_publishing_and_validates_artifacts():
    publish = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text()
    # Publishes on v* tag push, validates the artifacts, then ships via PyPI
    # Trusted Publishing (OIDC) - never an API token, username, or password.
    assert "push:" in publish
    assert "tags:" in publish
    assert '- "v*"' in publish
    assert "release:" not in publish
    assert "published" not in publish
    assert "environment:" in publish
    assert "name: pypi" in publish
    assert "id-token: write" in publish
    assert "pypa/gh-action-pypi-publish" in publish
    assert "concurrency:" in publish
    assert publish.count("timeout-minutes:") >= 2
    assert "cache: pip" in publish
    assert "cache-dependency-path: pyproject.toml" in publish
    assert "python -m pip install -U build twine" in publish
    assert "python -m build" in publish
    assert "twine check dist/*" in publish
    assert "python -m pip install dist/*.whl" not in publish
    assert "coreai-onnx schema --json" not in publish
    assert "if-no-files-found: error" in publish
    # OIDC only: no secret-based auth may creep into the publish workflow.
    assert "PYPI_TOKEN" not in publish
    assert "password:" not in publish
    assert "username:" not in publish
    # The old tag-triggered release.yml is removed so there is exactly one
    # publisher (a second workflow would fail OIDC and double-trigger).
    assert not (REPO_ROOT / ".github" / "workflows" / "release.yml").exists()


def test_codeql_workflow_scans_python_on_prs_and_schedule():
    codeql = (REPO_ROOT / ".github" / "workflows" / "codeql.yml").read_text()
    assert "pull_request:" in codeql
    assert "schedule:" in codeql
    assert "security-events: write" in codeql
    assert "concurrency:" in codeql
    assert "timeout-minutes:" in codeql
    assert "github/codeql-action/init@v4" in codeql
    assert "github/codeql-action/analyze@v4" in codeql
    assert "languages: python" in codeql
    assert "security-extended,security-and-quality" in codeql


def test_github_issue_templates_are_structured_forms():
    template_dir = REPO_ROOT / ".github" / "ISSUE_TEMPLATE"
    for name in ["bug_report", "feature_request", "op_request"]:
        path = template_dir / f"{name}.yml"
        assert path.is_file(), f"missing structured issue form: {path.name}"
        text = path.read_text()
        assert "body:" in text
        assert "validations:" in text
        assert "required: true" in text
        assert not (template_dir / f"{name}.md").exists()

    config = (template_dir / "config.yml").read_text()
    assert "blank_issues_enabled: false" in config
    assert "security/policy" in config


def test_dependabot_groups_low_risk_dependency_updates():
    text = (REPO_ROOT / ".github" / "dependabot.yml").read_text()
    assert "package-ecosystem: pip" in text
    assert "package-ecosystem: github-actions" in text
    assert text.count("open-pull-requests-limit: 5") == 2
    assert "production-dependencies:" in text
    assert "dependency-type: production" in text
    assert "development-dependencies:" in text
    assert "dependency-type: development" in text
    assert "github-actions:" in text
    assert '          - "*"' in text
    assert text.count("minor") >= 2
    assert text.count("patch") >= 2
    assert "major" not in text


def test_readme_has_agent_section():
    text = (REPO_ROOT / "README.md").read_text()
    assert "AGENTS.md" in text
    assert "schema --json" in text


def test_pyproject_mcp_extra_and_script():
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert any(dep.startswith("mcp") for dep in extras.get("mcp", [])), (
        "pyproject must declare the [mcp] extra"
    )
    scripts = data["project"]["scripts"]
    assert scripts.get("coreai-onnx-mcp") == "coreai_onnx._mcp:main"


def test_runtime_imports_use_declared_dependencies_only():
    for path in (REPO_ROOT / "src" / "coreai_onnx").rglob("*.py"):
        assert "typing_extensions" not in path.read_text(), (
            f"{path.relative_to(REPO_ROOT)} imports typing_extensions, "
            "which is not a declared runtime dependency"
        )


def test_schema_introspection_does_not_reference_private_argparse_classes():
    text = (REPO_ROOT / "src" / "coreai_onnx" / "_service.py").read_text()
    assert "argparse._" not in text
    assert "type: ignore[attr-defined]" not in text


def test_production_code_has_no_type_ignore_or_noqa_suppressions():
    for path in (REPO_ROOT / "src" / "coreai_onnx").rglob("*.py"):
        text = path.read_text()
        assert "type: ignore" not in text, (
            f"{path.relative_to(REPO_ROOT)} should isolate third-party typing "
            "gaps with typed helpers or casts instead of suppressions"
        )
        assert "# noqa" not in text, (
            f"{path.relative_to(REPO_ROOT)} should not carry lint suppressions"
        )


MCP_TOOL_NAMES = {"inspect_model", "convert_model", "verify_model", "get_schema"}


def test_docs_mcp_page_covers_all_tools():
    text = (REPO_ROOT / "docs" / "mcp.md").read_text()
    for tool in MCP_TOOL_NAMES:
        assert f"`{tool}`" in text, f"docs/mcp.md is missing {tool}"
    assert "coreai-onnx-mcp" in text


def test_agents_md_mentions_mcp_server():
    text = (REPO_ROOT / "AGENTS.md").read_text()
    assert "coreai-onnx-mcp" in text
    for tool in MCP_TOOL_NAMES:
        assert f"`{tool}`" in text, f"AGENTS.md is missing MCP tool {tool}"
