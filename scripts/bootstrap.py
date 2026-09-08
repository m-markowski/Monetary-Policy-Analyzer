"""Prepare the local Windows environment and launch Monetary Policy Analyzer."""

import argparse
import getpass
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import runpy
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / ".runtime"
VENV = ROOT / ".venv"
ENV_FILE = ROOT / "config" / ".env"
UV = RUNTIME / "uv" / "uv.exe"
KEY_PATTERN = re.compile(r"[a-z0-9]{32}")
KEY_LINE = re.compile(r"^\s*(?:export\s+)?FRED_API_KEY\s*=")
FRED_KEYS_URL = "https://fred.stlouisfed.org/docs/api/api_key.html"
BINARY_PACKAGES = (
    "arch",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "pyarrow",
    "shap",
    "numba",
    "llvmlite",
    "tensorflow",
    "tensorflow-intel",
    "msvc-runtime",
    "h5py",
    "ml-dtypes",
    "grpcio",
    "curl-cffi",
)
_DLL_HANDLES = []


class SetupError(Exception):
    pass


class InvalidKey(SetupError):
    pass


def say(message: str) -> None:
    print(message, flush=True)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def write_text(path: Path, text: str) -> None:
    """Replace a local configuration file only after the complete write succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_env_text(path: Path) -> str:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return ""
    try:
        encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        return data.decode(encoding)
    except UnicodeError as exc:
        raise SetupError(
            f"{path} is not a valid text file. Remove the unused template and run start.bat again."
        ) from exc


def read_env_key(path: Path = ENV_FILE) -> str:
    key = ""
    for line in read_env_text(path).splitlines():
        if KEY_LINE.match(line):
            value = line.split("=", 1)[1].split("#", 1)[0].strip().strip("'\"")
            key = value if KEY_PATTERN.fullmatch(value) else ""
    return key


def save_env_key(key: str, path: Path = ENV_FILE) -> None:
    if not KEY_PATTERN.fullmatch(key):
        raise InvalidKey("The key must contain 32 lowercase letters or digits.")
    lines = []
    replaced = False
    for line in read_env_text(path).splitlines():
        if KEY_LINE.match(line):
            if not replaced:
                lines.append(f"FRED_API_KEY={key}")
                replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"FRED_API_KEY={key}")
    write_text(path, "\n".join(lines) + "\n")


def validate_key_online(key: str) -> None:
    query = urllib.parse.urlencode({"series_id": "FEDFUNDS", "api_key": key, "file_type": "json"})
    request = urllib.request.Request(
        f"https://api.stlouisfed.org/fred/series?{query}",
        headers={"User-Agent": "Monetary-Policy-Analyzer/0.1"},
    )
    for attempt in range(1, 4):
        say(f"Checking the key with FRED ({attempt}/3)...")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.load(response)
            if not isinstance(data, dict) or not data.get("seriess"):
                raise SetupError("FRED returned an unexpected response. The key has not been changed.")
            return
        except urllib.error.HTTPError as exc:
            try:
                message = str(json.loads(exc.read()).get("error_message", "")).lower()
            except (ValueError, AttributeError):
                message = ""
            if exc.code in (400, 401, 403) and ("api_key" in message or "api key" in message):
                raise InvalidKey("FRED rejected this API key. Check that it is a registered, active key.") from None
            reason = f"FRED returned HTTP {exc.code}"
            if exc.code not in (408, 429) and exc.code < 500:
                raise SetupError(
                    f"{reason}. The key has not been changed; check network/proxy access to FRED."
                ) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            reason = "FRED could not be reached over a verified HTTPS connection"
        except ValueError:
            reason = "FRED did not return a valid JSON response"
        if attempt < 3:
            say(f"{reason}. Retrying...")
            time.sleep(2)
    raise SetupError(f"{reason}. Check your connection and run start.bat again. No new key was saved.")


def record_valid_key(key: str) -> None:
    digest = hashlib.sha256(key.encode()).hexdigest()
    write_text(RUNTIME / "fred-validation.json", json.dumps({"key_hash": digest}) + "\n")


def ensure_key(reset: bool = False) -> str:
    say("\n[3/5] Checking config/.env...")
    key = read_env_key()
    validated = read_json(RUNTIME / "fred-validation.json").get("key_hash")
    if key and not reset and validated == hashlib.sha256(key.encode()).hexdigest():
        # Normalize files written by older PowerShell commands to UTF-8.
        save_env_key(key)
        say("Using the previously verified FRED key. No online check is needed.")
        return key
    if key and not reset:
        try:
            validate_key_online(key)
        except InvalidKey as exc:
            say(str(exc))
        else:
            save_env_key(key)
            record_valid_key(key)
            say("The existing FRED key is valid.")
            return key
    if not sys.stdin.isatty():
        raise SetupError("Double-click start.bat in File Explorer to enter your FRED key interactively.")
    say(f"Create or view your own FRED API key here:\n{FRED_KEYS_URL}")
    say("Paste only the key and press Enter. Input is hidden; pasted characters will not appear.")
    say("An existing key will not be replaced unless the new key passes validation.")
    while True:
        key = getpass.getpass("FRED API key: ").strip()
        if not KEY_PATTERN.fullmatch(key):
            say("Enter all 32 lowercase letters/digits. Empty input is not accepted.")
            continue
        try:
            validate_key_online(key)
        except InvalidKey as exc:
            say(str(exc))
            continue
        save_env_key(key)
        record_valid_key(key)
        say("Verified. The key was saved to config/.env. No further setup input is required.")
        return key


def venv_python() -> Path:
    return VENV / "Scripts" / "python.exe"


def python_command(*arguments: str) -> list[str]:
    return [str(venv_python()), "-I", "-X", "utf8", *arguments]


def failure_message(code: int) -> str:
    hint = {
        0xC000001D: "A native library used an unsupported CPU instruction. TensorFlow requires a compatible AVX-capable CPU.",
        0xC0000135: "A required native DLL could not be loaded. Run start.bat --repair and check antivirus quarantine.",
        0xC0000005: "A native library crashed. Run start.bat --repair and review the import printed immediately above.",
    }.get(code & 0xFFFFFFFF, "Review the output above and run start.bat again.")
    return f"This step failed (exit code {code}). {hint}"


def run_step(command: list[str], env: dict[str, str], message: str) -> None:
    say(message)
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode:
        raise SetupError(failure_message(result.returncode))


def process_environment(key: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "FRED_API_KEY": key,
            "UV_PROJECT_ENVIRONMENT": str(VENV),
            "UV_CACHE_DIR": str(RUNTIME / "uv-cache"),
            "UV_PYTHON_INSTALL_DIR": str(RUNTIME / "python"),
            "KERAS_BACKEND": "tensorflow",
            "KERAS_HOME": str(RUNTIME / "keras"),
            "MPLCONFIGDIR": str(RUNTIME / "matplotlib"),
            "NUMBA_CACHE_DIR": str(RUNTIME / "numba"),
            "XDG_CACHE_HOME": str(RUNTIME / "cache"),
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "PATH": str(VENV / "Scripts") + os.pathsep + env.get("PATH", ""),
        }
    )
    return env


def prepare_environment(env: dict[str, str], repair: bool = False) -> None:
    say("\n[4/5] Checking the application environment...")
    receipt = RUNTIME / "runtime-checked.txt"
    lock_digest = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    checked = receipt.read_text(encoding="utf-8").strip() if receipt.is_file() else ""
    needs_check = repair or checked != lock_digest or not venv_python().is_file()
    if needs_check:
        receipt.unlink(missing_ok=True)
    command = [str(UV), "sync", "--locked", "--python", sys.executable]
    for name in BINARY_PACKAGES:
        command.extend(["--no-build-package", name])
    if repair:
        command.append("--reinstall")
    # uv compares .venv with uv.lock itself and does nothing when they already match.
    run_step(command, env, "Synchronising dependencies from uv.lock...")
    if needs_check:
        run_step(
            python_command(str(Path(__file__).resolve()), "--check-runtime"),
            env,
            "Checking imports, native libraries, the editable install and the neural networks...",
        )
        write_text(receipt, lock_digest + "\n")
    say("Environment is ready.")


def configure_dlls() -> None:
    directories = {VENV, VENV / "Scripts"}
    runtime = importlib.metadata.distribution("msvc-runtime")
    for file in runtime.files or ():
        if str(file).lower().endswith(".dll"):
            directories.add(Path(runtime.locate_file(file)).resolve().parent)
    for directory in sorted(directories):
        if directory.is_dir():
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))


def check_runtime() -> None:
    configure_dlls()
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "pyarrow",
        "sklearn",
        "statsmodels.api",
        "arch",
        "lightgbm",
        "xgboost",
        "shap",
        "optuna",
        "plotly",
        "streamlit",
        "jinja2",
        "joblib",
        "yaml",
        "dotenv",
        "fredapi",
        "yfinance",
        "tensorflow",
        "keras",
        "config.settings",
        "src.models.pipeline",
        "utils.plots",
    ):
        say(f"  Importing {name}...")
        importlib.import_module(name)
    metadata = importlib.metadata.distribution("monetary-policy-analyzer")
    direct_url = json.loads(metadata.read_text("direct_url.json") or "{}")
    if not direct_url.get("dir_info", {}).get("editable"):
        raise SetupError("The project was not installed in editable mode.")
    for name in ("config", "src", "utils"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != ROOT / name:
            raise SetupError(f"The import of {name} resolves outside this project.")
    importlib.metadata.version("ruff")
    numpy = importlib.import_module("numpy")
    neural = importlib.import_module("src.models.neural")
    for kind, shape in (("mlp", (3,)), ("lstm", (3, 2))):
        network = neural.build_network(kind, shape, 1, "regression", 8, 0.0, 0.0, 0.001)
        output = numpy.asarray(network(numpy.zeros((1, *shape), dtype="float32")))
        if output.shape != (1, 1) or not numpy.isfinite(output).all():
            raise SetupError(f"The {kind.upper()} runtime check returned invalid output.")
        say(f"  {kind.upper()} forward pass: OK")
    say("Runtime checks passed.")


def serve(port: int) -> None:
    configure_dlls()
    yfinance = importlib.import_module("yfinance")
    yfinance.set_tz_cache_location(str(RUNTIME / "yfinance"))
    sys.argv = [
        "streamlit",
        "run",
        str(ROOT / "app/Main_Page.py"),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.showEmailPrompt=false",
        "--browser.gatherUsageStats=false",
    ]
    runpy.run_module("streamlit", run_name="__main__")


def available_port() -> int:
    for port in range(8501, 8521):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SetupError("Ports 8501-8520 are in use. Close an unused local server and try again.")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT)
        process.wait(timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
        process.wait()


def wait_for_server(process: subprocess.Popen) -> None:
    import msvcrt

    while process.poll() is None:
        if sys.stdin.isatty() and msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in ("\x00", "\xe0"):
                msvcrt.getwch()
            elif key.lower() == "q" or key == "\x03":
                say("\nStopping the application...")
                return
        time.sleep(0.2)
    if process.returncode:
        raise SetupError(f"Streamlit stopped. {failure_message(process.returncode)}")


def start_server(env: dict[str, str], no_browser: bool = False) -> None:
    port = available_port()
    url = f"http://127.0.0.1:{port}"
    say(f"\n[5/5] Starting the application at {url} ...")
    process = subprocess.Popen(
        python_command(str(Path(__file__).resolve()), "--serve", str(port)),
        cwd=ROOT,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise SetupError(f"Streamlit stopped during startup. {failure_message(process.returncode)}")
            try:
                with opener.open(f"{url}/_stcore/health", timeout=1) as response:
                    ready = response.status == 200 and response.read().strip() == b"ok"
            except (urllib.error.URLError, TimeoutError, OSError):
                ready = False
            if ready:
                break
            time.sleep(0.4)
        else:
            raise SetupError("Streamlit did not become ready. Review its messages above.")
        say(f"\nApplication server is ready: {url}")
        say("Keep this window open while using the application.")
        say("Press Q in this window to stop. Closing the browser tab does not stop the server.")
        if not no_browser:
            try:
                opened = webbrowser.open(url)
            except webbrowser.Error:
                opened = False
            if not opened:
                say("The browser could not be opened automatically. Open the address above manually.")
        wait_for_server(process)
    except KeyboardInterrupt:
        say("\nStopping the application...")
    finally:
        stop_server(process)
    say("The application has stopped. Your environment and API key are saved for the next launch.")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--check-runtime":
        check_runtime()
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        serve(int(sys.argv[2]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true", help="prepare the environment without opening the app")
    parser.add_argument("--reset-key", action="store_true", help="prompt for a replacement FRED API key")
    parser.add_argument("--repair", action="store_true", help="reinstall dependencies and rerun runtime checks")
    parser.add_argument("--no-browser", action="store_true", help="start the server without opening a browser")
    args = parser.parse_args()
    os.chdir(ROOT)
    if sys.version_info[:2] != (3, 12) or not Path(sys.executable).resolve().is_relative_to(RUNTIME / "python"):
        raise SetupError("Use start.bat so that the repository-local Python 3.12 is used.")
    key = ensure_key(reset=args.reset_key)
    env = process_environment(key)
    prepare_environment(env, repair=args.repair)
    if args.setup_only:
        say("\n[5/5] Setup completed. The application was not started (--setup-only).")
        say("Double-click start.bat to open the application.")
    else:
        start_server(env, no_browser=args.no_browser)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        say("\nSetup cancelled. Run start.bat again to continue.")
        raise SystemExit(130) from None
    except (SetupError, OSError, ValueError, ImportError) as exc:
        say(f"\nERROR: {exc}")
        say("The launcher stopped. Review the message above and run start.bat again.")
        raise SystemExit(1) from None
