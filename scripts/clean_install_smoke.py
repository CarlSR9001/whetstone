"""Prove that the built wheel works without a source checkout or optional deps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap
import venv
from pathlib import Path


def run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")

    with tempfile.TemporaryDirectory(prefix="whetstone-clean-install-") as raw:
        root = Path(raw)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        if sys.platform == "win32":
            python = environment / "Scripts" / "python.exe"
            whetstone = environment / "Scripts" / "whetstone.exe"
        else:
            python = environment / "bin" / "python"
            whetstone = environment / "bin" / "whetstone"

        run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", str(wheel)],
            cwd=root,
        )
        run(
            [
                str(whetstone),
                "--root",
                str(root / "bank"),
                "--config",
                str(root / "whetstone.toml"),
                "init",
            ],
            cwd=root,
        )
        run(
            [
                str(whetstone),
                "--root",
                str(root / "bank"),
                "--config",
                str(root / "whetstone.toml"),
                "status",
            ],
            cwd=root,
        )

        smoke = textwrap.dedent(
            """
            import json
            import threading
            import urllib.request

            import bcv
            from bcv.toolbox_service import make_server

            server = make_server("127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                health = json.load(urllib.request.urlopen(base + "/api/health", timeout=5))
                assert health["status"] == "ok"
                assert health["version"] == bcv.__version__
                assert health["tools"] == 8
                assert health["private_bank_loaded"] is False
                assert health["open_bench"]["ready"] is True
                assert health["open_bench"]["raw_tasks_persisted"] is False
                assert health["open_bench"]["raw_answers_persisted"] is False
                assert health["build_commit"] == "development" or len(health["build_commit"]) == 40

                evidence = json.load(urllib.request.urlopen(base + "/api/evidence", timeout=5))
                assert evidence["cross_scale"]["models"] == 8
                assert evidence["source_receipts_sha256"]

                for path, marker in (
                    ("/", b"Before you ship an AI change"),
                    ("/benchmark", b"Compare the version you have"),
                    ("/benchmark.js", b"/api/open-bench/submit"),
                    ("/skill.md", b"Whetstone"),
                    ("/app.js", b"serviceVersion"),
                    ("/styles.css", b"--cyan"),
                    ("/og.png", b"\\x89PNG\\r\\n\\x1a\\n"),
                    ("/open-bench-og.png", b"\\x89PNG\\r\\n\\x1a\\n"),
                ):
                    body = urllib.request.urlopen(base + path, timeout=5).read()
                    assert marker in body, (path, len(body))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
            """
        )
        run([str(python), "-c", smoke], cwd=root)

        receipt = {
            "wheel": wheel.name,
            "python": str(python),
            "version": subprocess.check_output(
                [str(python), "-c", "import bcv; print(bcv.__version__)"],
                cwd=root,
                text=True,
            ).strip(),
            "status": "PASS",
        }
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
