from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, Signal

from .adapters.backtest_adapter import BacktestCommand


class BacktestRunner(QObject):
    log_received = Signal(str)
    progress_changed = Signal(int)
    status_changed = Signal(str)
    result_path_changed = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.process: QProcess | None = None
        self.result_path: str = ""
        self.rebalance_total: int = 0
        self.rebalance_done: int = 0

    def start(self, command: BacktestCommand, working_dir: Path) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.failed.emit("已有回测任务正在运行")
            return
        self.result_path = ""
        self.rebalance_total = 0
        self.rebalance_done = 0
        self.process = QProcess(self)
        self.process.setProgram(command.program)
        self.process.setArguments(command.arguments)
        self.process.setWorkingDirectory(str(working_dir))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)
        self.status_changed.emit("Running...")
        self.progress_changed.emit(1)
        self.log_received.emit(f"COMMAND: {command.program} {' '.join(command.arguments)}")
        self.process.start()
        if not self.process.waitForStarted(3000):
            self.status_changed.emit("Failed")
            self.failed.emit(f"回测进程未能启动: {self.process.errorString()}")
            return

    def stop(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()

    def _read_output(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return
        self.log_received.emit(data.rstrip())
        for line in data.splitlines():
            self._parse_progress_line(line.strip())

    def _parse_progress_line(self, line: str) -> None:
        if not line:
            return
        m = re.search(r"PROGRESS:\s*(\d+)", line)
        if m:
            self.progress_changed.emit(max(0, min(100, int(m.group(1)))))
            return
        m = re.search(r"(?:RESULT_PATH|BACKTEST_RUN_DIR)=(.+)", line)
        if m:
            self.result_path = m.group(1).strip()
            self.result_path_changed.emit(self.result_path)
            self.progress_changed.emit(100)
            return
        m = re.search(r"result_dir=([^ ]+)", line)
        if m and not self.result_path:
            self.result_path = m.group(1).strip()
            self.result_path_changed.emit(self.result_path)
            self.progress_changed.emit(max(5, 5))
        if "cache hit:" in line:
            self.progress_changed.emit(max(15, self._current_progress_floor()))
        elif "download/cache build start" in line:
            self.progress_changed.emit(8)
        elif "download/cache build done" in line or "provider_loaded_from_local_cache" in line:
            self.progress_changed.emit(20)
        m = re.search(r"rebalance_dates count=(\d+)", line)
        if m:
            self.rebalance_total = max(1, int(m.group(1)))
            self.rebalance_done = 0
            self.progress_changed.emit(25)
        if "[rebalance-debug]" in line and "signal_date=" in line:
            if self.rebalance_total <= 0:
                self.progress_changed.emit(min(95, self._current_progress_floor() + 5))
            else:
                self.rebalance_done += 1
                progress = 25 + int(65 * min(self.rebalance_done, self.rebalance_total) / self.rebalance_total)
                self.progress_changed.emit(min(95, progress))
        if "Backtest Performance Summary" in line or "outputs_saved" in line:
            self.progress_changed.emit(98)

    def _current_progress_floor(self) -> int:
        return 1

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        if exit_code == 0:
            self.progress_changed.emit(100)
            self.status_changed.emit("Completed")
            self.finished_ok.emit(self.result_path)
        else:
            self.status_changed.emit("Failed")
            self.failed.emit(f"回测进程失败，退出码: {exit_code}")

    def _on_error(self, _error: QProcess.ProcessError) -> None:
        message = self.process.errorString() if self.process is not None else "Unknown QProcess error"
        self.status_changed.emit("Failed")
        self.failed.emit(message)
