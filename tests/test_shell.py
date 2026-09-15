import time

from app import config
from app import shell


def test_echo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    out = shell.run_shell("echo hello")
    assert "hello" in out and "Exit code: 0" in out


def test_stderr_and_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    out = shell.run_shell("ls /definitely-not-here")
    assert "STDERR" in out and "Exit code: 2" in out


def test_working_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    out = shell.run_shell("pwd")
    assert str(tmp_path) in out
    (tmp_path / "sub").mkdir()
    out = shell.run_shell("pwd", working_dir="sub")
    assert str(tmp_path / "sub") in out


def test_working_dir_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    assert shell.run_shell("pwd", working_dir="/etc").startswith("error")


def test_workspace_writes_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    out = shell.run_shell("mkdir -p sub && rm -rf sub")
    assert "Exit code: 0" in out


def test_timeout_kills_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    t0 = time.monotonic()
    out = shell.run_shell("sleep 30", timeout=1)
    assert time.monotonic() - t0 < 5, "timeout did not stop the command"
    assert "timed out" in out


def test_deny_patterns(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    for cmd in [
        "rm -rf /",
        "sudo rm -rf / --no-preserve-root",
        "rm -rf ~",
        "rm -rf /*",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/workspace/x",
        "echo x > /dev/sda1",
        "shutdown now",
        "reboot",
        ":(){ :|:& };:",
    ]:
        assert shell.run_shell(cmd).startswith("error: command denied"), cmd


def test_protected_dirs_readonly(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    for cmd in [
        "echo x > /data/out.txt",
        "rm -rf /app",
        "cp /etc/hostname /models/x",
        "sed -i s/a/b/ /data/conversation.json",
    ]:
        assert shell.run_shell(cmd).startswith("error"), cmd
    # plain reads of protected dirs stay allowed
    assert "Exit code: 0" in shell.run_shell("cat /app/config.py | head -1")


def test_output_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    monkeypatch.setattr(config, "EXEC_MAX_OUTPUT", 2000)
    out = shell.run_shell("seq 1 5000")
    assert "chars truncated" in out
    assert len(out) < 2400
    assert "5000" in out, "tail of output missing"


def test_missing_command(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path))
    assert shell.run_shell("").startswith("error")
