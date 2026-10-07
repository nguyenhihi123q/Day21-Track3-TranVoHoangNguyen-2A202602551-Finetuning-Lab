"""Build a standalone Colab notebook from this checkout's tracked lab files."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import pathlib
import subprocess
import textwrap
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    paths = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(filter(None, paths)):
            path = pathlib.PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or any(
                part in {".git", ".env", ".aws", ".codex"} for part in path.parts
            ):
                raise ValueError(f"Refusing to package {name}")
            local = ROOT / name
            if local.is_file():
                entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, local.read_bytes())
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()

    cells = []

    def cell(kind: str, source: str) -> None:
        source = textwrap.dedent(source).strip() + "\n"
        item = {"cell_type": kind, "metadata": {},
                "source": source.splitlines(keepends=True),
                "id": hashlib.sha1(source.encode()).hexdigest()[:8]}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", """
        # Lab 21 — bản chạy đầy đủ trên Colab T4

        Notebook chứa bản sao mã lab trong thư mục của bạn, không cần Colab tải notebook từ GitHub.
        Chọn **Runtime → Change runtime type → T4 GPU**, rồi chạy lần lượt từ trên xuống.
        Cấu hình nộp bài: **EPOCHS=2**, **toàn bộ tập eval**, **assistant-only**.
        Core NB1–NB5 mất khoảng 100–130 phút trên T4 theo tài liệu repo; NB6 là phần thưởng +3.

        Sau mỗi giai đoạn có ZIP sao lưu trong Files để tải về. ZIP gồm kết quả, log và adapter;
        không chứa trọng số base/merged. Máy ảo Colab mất dữ liệu khi runtime bị xoá.
        Nếu một ô lỗi, dừng và gửi log. Chỉ chạy lại giai đoạn chưa hoàn thành;
        không chạy lại NB2 sau khi đã train để thay đổi baseline.
        Report vẫn cần viết theo số liệu thật và phản tư của bạn trước khi nộp.
    """)
    cell("code", f'''
# @title 1. Setup — giải nén bản lab và cài dependency
import base64, hashlib, io, json, os, pathlib, subprocess, sys, zipfile

ROOT = pathlib.Path("/content/Day21-Finetuning-Lab")
SNAPSHOT_SHA = "{digest}"
SOURCE_COMMIT = "{commit}"
payload = base64.b64decode("{encoded}")
assert hashlib.sha256(payload).hexdigest() == SNAPSHOT_SHA
marker = ROOT / ".colab_snapshot_sha"
if ROOT.exists() and not marker.exists():
    raise RuntimeError("Thư mục đã tồn tại nhưng không thuộc snapshot này; dùng runtime mới.")
if marker.exists() and marker.read_text().strip() != SNAPSHOT_SHA:
    # ZIP metadata may differ between notebook builds while the source is identical.
    # Reuse identical source without replacing generated results or the report.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        changed = []
        for member in archive.infolist():
            name = pathlib.PurePosixPath(member.filename)
            if name.parts[0] in ("results", "adapters", "submission"):
                continue
            existing = ROOT / member.filename
            if not existing.is_file() or existing.read_bytes() != archive.read(member):
                changed.append(member.filename)
    if changed:
        raise RuntimeError("Source khác snapshot; giữ nguyên baseline. File khác: " + str(changed[:5]))
    marker.write_text(SNAPSHOT_SHA)
if not marker.exists():
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            target = (ROOT / member.filename).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                raise ValueError("Unsafe archive path")
        archive.extractall(ROOT)
    marker.write_text(SNAPSHOT_SHA)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
os.environ["COMPUTE_TIER"] = "T4"
os.environ["EPOCHS"] = "2"
os.environ["MASK_MODE"] = "assistant-only"
for key in ("EVAL_LIMIT", "BASE_MODEL", "ONLY", "FORCE_RETRAIN"):
    os.environ.pop(key, None)

import torch
if not torch.cuda.is_available():
    raise RuntimeError("Chưa có GPU. Chọn Runtime > Change runtime type > T4 GPU.")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), flush=True)
print("Source commit:", SOURCE_COMMIT, flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
print("Setup hoàn tất. Nếu Colab yêu cầu restart runtime, restart rồi chạy lại ô này.")
''')
    cell("code", '''
        # @title 2. Kiểm tra môi trường và chuẩn bị sao lưu
        import datetime, signal
        from google.colab import files
        from IPython.display import FileLink, display

        RUN_NB6 = True  # @param {type:"boolean"}
        AUTO_DOWNLOAD = False  # @param {type:"boolean"}
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "logs").mkdir(exist_ok=True)
        metadata = {"source_commit": SOURCE_COMMIT, "snapshot_sha256": SNAPSHOT_SHA,
                    "compute_tier": "T4", "epochs": 2, "mask_mode": "assistant-only",
                    "eval_limit": None, "gpu": torch.cuda.get_device_name(0)}
        (ROOT / "results" / "colab_run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")

        def backup(stage):
            archive_path = pathlib.Path("/content") / f"lab21_{stage}_checkpoint.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for folder in ("results", "logs", "submission", "notebooks", "src", "scripts", "data"):
                    for path in (ROOT / folder).rglob("*"):
                        if path.is_file() and "__pycache__" not in path.parts:
                            archive.write(path, path.relative_to(ROOT).as_posix())
                for name in ("requirements.txt", "requirements-cpu.txt", "pyproject.toml", "rubric.md"):
                    archive.write(ROOT / name, name)
                for path in (ROOT / "adapters").rglob("*"):
                    if path.is_file() and "merged" not in path.relative_to(ROOT / "adapters").parts:
                        archive.write(path, path.relative_to(ROOT).as_posix())
            print(f"Backup: {archive_path.name} ({archive_path.stat().st_size / 1024**2:.1f} MB)")
            print("Tải ZIP bằng thanh Files bên trái → menu của file → Download.")
            if AUTO_DOWNLOAD:
                try:
                    files.download(str(archive_path))
                except Exception as exc:
                    print("Không tự tải được ZIP; tải trong Files:", exc)
            return archive_path

        def run_stage(stage):
            prerequisites = {
                "nb2": ["results/mask_proof.json"],
                "nb3": ["results/baselines_frozen.json"],
                "nb4": ["results/baselines_frozen.json", "adapters/correct/adapter_model.safetensors"],
                "nb5": ["results/baselines_frozen.json", *[
                    f"adapters/{name}/adapter_model.safetensors"
                    for name in ("correct", "attn_only", "wrong_lr", "qlora")]],
                "nb6": ["results/verdict.json", "results/autopsy.json"],
            }
            missing = [name for name in prerequisites.get(stage, []) if not (ROOT / name).is_file()]
            if missing:
                raise RuntimeError(f"Chưa đủ điều kiện chạy {stage}: {missing}. Chạy đúng thứ tự NB1–NB5.")
            if stage == "nb2" and (ROOT / "results/baselines_frozen.json").exists():
                print("Baseline đã đóng băng; giữ nguyên kết quả NB2.")
                return
            if stage == "nb2" and (ROOT / "results/runs.csv").exists():
                raise RuntimeError("Đã có training run; không đo lại baseline sau train.")
            if stage in ("nb3", "nb4"):
                frozen = json.loads((ROOT / "results/baselines_frozen.json").read_text())
                if frozen.get("smoke_mode") or frozen.get("eval_limit"):
                    raise RuntimeError("Baseline đang là smoke run; cần full eval trước khi train.")
                if frozen["baseline_b"]["target"] <= frozen["baseline_a"]["target"]:
                    raise RuntimeError("Baseline (b) chưa mạnh hơn (a); xử lý trước khi train.")
            log = ROOT / "logs" / f"{stage}_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt"
            process = None
            try:
                with log.open("w", encoding="utf-8") as output:
                    process = subprocess.Popen(
                        [sys.executable, "-u", "scripts/colab_run.py", stage], cwd=ROOT,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, start_new_session=True,
                        env={**os.environ, "PYTHONUNBUFFERED": "1"})
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        output.write(line)
                    rc = process.wait()
                if rc:
                    raise RuntimeError(f"{stage} lỗi (exit {rc}); xem log phía trên. Dừng ở đây.")
            except BaseException:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
            finally:
                backup(stage)

        subprocess.run([sys.executable, "scripts/verify.py", "--smoke"], check=True)
    ''')
    stages = [("nb1", "Dữ liệu, template và mask proof"),
              ("nb2", "Đo và đóng băng baseline TRƯỚC KHI TRAIN"),
              ("nb3", "Train cấu hình correct"),
              ("nb4", "Ba run đối chứng cùng ngân sách step"),
              ("nb5", "Bốn nhóm đánh giá và phán quyết")]
    for number, (name, title) in enumerate(stages, 3):
        cell("code", f'# @title {number}. {title}\nrun_stage("{name}")')
    cell("code", '''
        # @title 8. NB6 — merge và hot-swap (+3 điểm nếu đủ bằng chứng)
        if RUN_NB6:
            run_stage("nb6")
        else:
            print("Đã bỏ qua NB6 theo lựa chọn RUN_NB6.")
    ''')
    cell("code", '''
        # @title 9. Xem kết quả và kiểm tra trước khi viết report
        result = subprocess.run([sys.executable, "scripts/verify.py"], check=False)
        if result.returncode:
            print("Đọc từng FAIL/WARN. REPORT.md hiện còn mẫu nên chưa đủ điều kiện nộp.")
        for name in ("runs.csv", "autopsy.json", "verdict.json", "merge_check.json"):
            path = ROOT / "results" / name
            if path.exists():
                print("\\n---", name, "---\\n", path.read_text(encoding="utf-8"))
        backup("final")
        print("Mở Files bên trái, chọn /content/lab21_final_checkpoint.zip → Download.")
        print("Tải ZIP final về máy để viết REPORT.md và kiểm tra lại trước khi nộp.")
    ''')
    notebook = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"name": "Lab21_COLAB_FULL.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    destination = ROOT / "colab" / "Lab21_COLAB_FULL.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
