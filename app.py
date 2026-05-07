from __future__ import annotations

import math
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import sounddevice as sd
import soundfile as sf


APP_TITLE = "噪音反击系统 v4"
SAMPLE_RATE = 44_100
BLOCK_SIZE = 4_096
WINDOW_SECONDS = 4.0
CALIBRATION_SECONDS = 5.0
DEFAULT_REQUIRED_ABOVE_DB = 27.0
DEFAULT_LOW_RATIO_PERCENT = 23
DEFAULT_HIGH_RATIO_PERCENT = 42
DEFAULT_STABLE_BLOCKS = 3
DEFAULT_AUDIO_PATH = Path(__file__).resolve().parent / "assets" / "default_alert.wav"


@dataclass
class NoiseMetrics:
    above_floor: float = 0.0
    low_ratio: float = 0.0
    voice_ratio: float = 0.0
    scream_ratio: float = 0.0
    impact: float = 0.0
    score: float = 0.0
    low_score: float = 0.0
    voice_score: float = 0.0
    scream_score: float = 0.0
    rms_db: float = -90.0
    low_ratio_raw: float = 0.0
    high_ratio_raw: float = 0.0
    voice_ratio_raw: float = 0.0
    scream_ratio_raw: float = 0.0
    centroid_hz: float = 0.0
    zcr: float = 0.0
    harmonicity: float = 0.0


class CounterNoiseWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(980, 880)
        self.setMinimumSize(860, 720)

        self.devices: list[dict] = []
        self.input_devices: list[tuple[int, str]] = []
        self.output_devices: list[tuple[int, str]] = []
        self.audio_path: Path | None = DEFAULT_AUDIO_PATH if DEFAULT_AUDIO_PATH.exists() else None

        self.metrics = NoiseMetrics()
        self.event_times: list[float] = []
        self.running = False
        self.started_at = 0.0
        self.last_fire_at = 0.0
        self.calibrating_until = 0.0
        self.trigger_total = 0
        self.noise_floor_db = -55.0
        self.calibration_values: list[float] = []
        self.previous_spectrum: np.ndarray | None = None
        self.clear_block_counts: dict[str, int] = {}
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stream: sd.InputStream | None = None

        self._build_ui()
        self._refresh_devices(log=False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(120)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        shell = QVBoxLayout(root)
        shell.setContentsMargins(18, 14, 18, 18)
        shell.setSpacing(14)

        title = QLabel(APP_TITLE)
        title.setFont(QFont("Arial", 20, QFont.Bold))
        subtitle = QLabel("先选设备和音频，再开始监听；需要微调时只看左侧规则。")
        subtitle.setObjectName("subtitle")
        header = QVBoxLayout()
        header.setSpacing(2)
        header.addWidget(title)
        header.addWidget(subtitle)
        shell.addLayout(header)

        main = QHBoxLayout()
        main.setSpacing(14)
        shell.addLayout(main, 1)

        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setMinimumWidth(390)
        left_panel.setMaximumWidth(460)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(16, 16, 16, 16)
        left.setSpacing(14)

        right_panel = QFrame()
        right_panel.setObjectName("rightPanel")
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(16, 16, 16, 16)
        right.setSpacing(14)

        main.addWidget(left_panel)
        main.addWidget(right_panel, 1)

        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.refresh_button = QPushButton("刷新设备")
        self.refresh_button.clicked.connect(self._refresh_devices)

        device_box, device_grid = self._section_grid()
        device_grid.addWidget(self._group_title("设备"), 0, 0, 1, 3)
        device_grid.addWidget(QLabel("输入设备 (麦克风)"), 1, 0)
        device_grid.addWidget(self.input_combo, 2, 0, 1, 3)
        device_grid.addWidget(QLabel("输出设备 (音箱)"), 3, 0)
        device_grid.addWidget(self.output_combo, 4, 0, 1, 3)
        device_grid.addWidget(self.refresh_button, 5, 0, 1, 3)
        device_grid.setColumnStretch(1, 1)
        left.addWidget(device_box)

        default_file_name = self.audio_path.name if self.audio_path else "未选择音频"
        self.file_label = QLabel(default_file_name)
        self.file_label.setObjectName("fileName")
        choose_button = QPushButton("选择文件")
        choose_button.clicked.connect(self._choose_file)
        file_box, file_layout = self._section_hbox()
        file_layout.setSpacing(10)
        file_layout.addWidget(self._group_title("反击音频"))
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(choose_button)
        left.addWidget(file_box)

        self.low_target_check = QCheckBox("低频冲击")
        self.voice_target_check = QCheckBox("人声")
        self.scream_target_check = QCheckBox("尖叫声")
        self.low_target_check.setChecked(True)
        target_box = QFrame()
        target_box.setObjectName("inlineBox")
        target_layout = QHBoxLayout(target_box)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setSpacing(12)
        target_layout.addWidget(self.low_target_check)
        target_layout.addWidget(self.voice_target_check)
        target_layout.addWidget(self.scream_target_check)
        target_layout.addStretch(1)
        self.sensitivity_slider, self.sensitivity_value = self._make_slider(1, 10, 2)
        self.confirm_slider, self.confirm_value = self._make_slider(1, 6, 4)
        self.cooldown_slider, self.cooldown_value = self._make_slider(1, 45, 12)
        param_box, param_grid = self._section_grid()
        param_grid.addWidget(self._group_title("基础参数"), 0, 0, 1, 5)
        param_grid.addWidget(QLabel("监听目标"), 1, 0)
        param_grid.addWidget(target_box, 1, 1, 1, 4)
        self._add_slider_row(param_grid, 2, "灵敏度", "保守", self.sensitivity_slider, "敏感", self.sensitivity_value)
        self._add_slider_row(param_grid, 3, "确认次数", "", self.confirm_slider, "", self.confirm_value)
        self._add_slider_row(param_grid, 4, "冷却时间", "", self.cooldown_slider, "", self.cooldown_value)
        param_grid.setColumnStretch(2, 1)
        left.addWidget(param_box)

        self.above_db_slider, self.above_db_value = self._make_slider(12, 45, round(DEFAULT_REQUIRED_ABOVE_DB))
        self.low_ratio_slider, self.low_ratio_value = self._make_slider(5, 60, DEFAULT_LOW_RATIO_PERCENT)
        self.high_ratio_slider, self.high_ratio_value = self._make_slider(10, 80, DEFAULT_HIGH_RATIO_PERCENT)
        self.stable_blocks_slider, self.stable_blocks_value = self._make_slider(1, 8, DEFAULT_STABLE_BLOCKS)
        advanced_box, advanced_grid = self._section_grid()
        advanced_grid.addWidget(self._group_title("高级规则"), 0, 0, 1, 5)
        self._add_slider_row(advanced_grid, 1, "分贝门槛", "低", self.above_db_slider, "高", self.above_db_value)
        self._add_slider_row(advanced_grid, 2, "低频下限", "低", self.low_ratio_slider, "高", self.low_ratio_value)
        self._add_slider_row(advanced_grid, 3, "高频上限", "少", self.high_ratio_slider, "多", self.high_ratio_value)
        self._add_slider_row(advanced_grid, 4, "连续块数", "短", self.stable_blocks_slider, "长", self.stable_blocks_value)
        advanced_grid.setColumnStretch(2, 1)
        left.addWidget(advanced_box)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        start_button = QPushButton("▶ 开始监听")
        calibrate_button = QPushButton("◎ 重新校准")
        stop_button = QPushButton("■ 停止")
        start_button.setObjectName("primaryButton")
        start_button.clicked.connect(self._start)
        calibrate_button.clicked.connect(self._begin_calibration)
        stop_button.clicked.connect(self._stop)
        controls.addWidget(start_button)
        controls.addWidget(calibrate_button)
        controls.addWidget(stop_button)
        left.addStretch(1)
        left.addLayout(controls)

        self.status_label = QLabel("就绪 - 选择音频文件后点击开始")
        self.status_label.setObjectName("statusText")
        self.above_floor_bar = self._make_bar()
        self.low_ratio_bar = self._make_bar()
        self.high_ratio_bar = self._make_bar()
        self.voice_ratio_bar = self._make_bar()
        self.scream_ratio_bar = self._make_bar()
        self.impact_bar = self._make_bar()
        self.score_bar = self._make_bar()
        self.floor_label = QLabel("底噪: -- dB")
        self.trigger_label = QLabel("0/3")
        self.trigger_total_label = QLabel("0")
        self.runtime_label = QLabel("00:00:00")

        status_box, status_grid = self._section_grid()
        status_grid.addWidget(self._group_title("运行监控"), 0, 0, 1, 4)
        status_grid.addWidget(self.status_label, 1, 0, 1, 4)
        self._add_metric_row(status_grid, 2, 0, "超过底噪", self.above_floor_bar)
        self._add_metric_row(status_grid, 3, 0, "低频占比", self.low_ratio_bar)
        self._add_metric_row(status_grid, 4, 0, "高频占比", self.high_ratio_bar)
        self._add_metric_row(status_grid, 5, 0, "人声占比", self.voice_ratio_bar)
        self._add_metric_row(status_grid, 6, 0, "尖叫占比", self.scream_ratio_bar)
        self._add_metric_row(status_grid, 7, 0, "突变强度", self.impact_bar)
        self._add_metric_row(status_grid, 8, 0, "综合评分", self.score_bar)
        status_grid.addWidget(self.floor_label, 9, 0, 1, 4)
        status_grid.setColumnStretch(1, 1)
        right.addWidget(status_box)

        stats_box, stats_grid = self._section_grid()
        stats_grid.addWidget(self._group_title("触发统计"), 0, 0, 1, 4)
        stats_grid.addWidget(QLabel("冲击累积"), 1, 0)
        stats_grid.addWidget(self.trigger_label, 1, 1)
        stats_grid.addWidget(QLabel("触发次数"), 1, 2)
        stats_grid.addWidget(self.trigger_total_label, 1, 3)
        stats_grid.addWidget(QLabel("运行时长"), 2, 0)
        stats_grid.addWidget(self.runtime_label, 2, 1, 1, 3)
        stats_grid.setColumnStretch(1, 1)
        stats_grid.setColumnStretch(3, 1)
        right.addWidget(stats_box)

        log_box = QFrame()
        log_box.setObjectName("card")
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(14, 12, 14, 14)
        log_layout.setSpacing(10)
        log_layout.addWidget(self._group_title("日志"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Menlo", 13))
        log_layout.addWidget(self.log, 1)
        right.addWidget(log_box, 1)

        self._apply_style()
        self._sync_labels()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f5f7; color: #242424; font: 15px Arial; }
            QLabel#subtitle { color: #666666; font-size: 13px; }
            QFrame#leftPanel, QFrame#rightPanel {
                background: #ffffff;
                border: 1px solid #e2e4e8;
                border-radius: 10px;
            }
            QFrame#card {
                background: #f8f8f9;
                border: 1px solid #e6e7eb;
                border-radius: 8px;
            }
            QFrame#inlineBox {
                background: transparent;
                border: 0;
            }
            QCheckBox {
                background: transparent;
                spacing: 6px;
            }
            QLabel#groupTitle {
                color: #111111;
                font-size: 15px;
                font-weight: 700;
                padding-bottom: 2px;
            }
            QLabel#fileName {
                color: #333333;
                background: transparent;
            }
            QLabel#statusText {
                color: #111111;
                font-size: 18px;
                font-weight: 700;
                padding: 6px 0 10px 0;
            }
            QComboBox, QPushButton {
                background: #ffffff;
                border: 1px solid #d4d4d4;
                border-radius: 7px;
                padding: 6px 10px;
                min-height: 24px;
            }
            QPushButton#primaryButton {
                background: #1f6feb;
                color: #ffffff;
                border-color: #1f6feb;
            }
            QPushButton:hover { border-color: #7eb8f0; }
            QPushButton:pressed { background: #eeeeee; }
            QSlider::groove:horizontal {
                height: 4px;
                background: #d1d1d1;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #cccccc;
                width: 20px;
                margin: -8px 0;
                border-radius: 10px;
            }
            QProgressBar {
                background: #dcdcdc;
                border: 0;
                border-radius: 2px;
                height: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #7eb8f0;
                border-radius: 2px;
            }
            QTextEdit {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 3px;
            }
            """
        )

    def _section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("class", "sectionTitle")
        return label

    def _group_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("groupTitle")
        return label

    def _section_grid(self) -> tuple[QFrame, QGridLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QGridLayout(frame)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)
        return frame, layout

    def _section_hbox(self) -> tuple[QFrame, QHBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        return frame, layout

    def _make_slider(self, minimum: int, maximum: int, value: int) -> tuple[QSlider, QLabel]:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        value_label = QLabel()
        slider.valueChanged.connect(self._sync_labels)
        return slider, value_label

    def _add_slider_row(
        self,
        layout: QGridLayout,
        row: int,
        label: str,
        left: str,
        slider: QSlider,
        right: str,
        value: QLabel,
    ) -> None:
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(QLabel(left), row, 1)
        layout.addWidget(slider, row, 2)
        layout.addWidget(QLabel(right), row, 3)
        layout.addWidget(value, row, 4)

    def _make_bar(self) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setTextVisible(False)
        return bar

    def _add_metric_row(self, layout: QGridLayout, row: int, col: int, label: str, bar: QProgressBar) -> None:
        layout.addWidget(QLabel(label), row, col)
        layout.addWidget(bar, row, col + 1)

    def _sync_labels(self) -> None:
        self.sensitivity_value.setText(str(self.sensitivity_slider.value()))
        self.confirm_value.setText(f"{self.confirm_slider.value()}次/4秒")
        self.cooldown_value.setText(f"{self.cooldown_slider.value()}秒")
        self.above_db_value.setText(f"{self.above_db_slider.value()} dB")
        self.low_ratio_value.setText(f"{self.low_ratio_slider.value()}%")
        self.high_ratio_value.setText(f"{self.high_ratio_slider.value()}%")
        self.stable_blocks_value.setText(f"{self.stable_blocks_slider.value()}块")
        self.trigger_label.setText(f"{len(self.event_times)}/{self.confirm_slider.value()}")

    def _begin_calibration(self) -> None:
        if not self.running:
            self.status_label.setText("请先开始监听，再校准底噪")
            self._log("校准需要先开始监听")
            return
        self.calibration_values.clear()
        self.event_times.clear()
        self.previous_spectrum = None
        self.clear_block_counts.clear()
        self.calibrating_until = time.monotonic() + CALIBRATION_SECONDS
        self.status_label.setText("正在校准底噪 - 保持环境安静")
        self._log(f"开始 {CALIBRATION_SECONDS:.0f} 秒底噪校准")

    def _refresh_devices(self, log: bool = True) -> None:
        try:
            self.devices = list(sd.query_devices())
            default_in, default_out = sd.default.device
        except Exception as exc:
            QMessageBox.critical(self, "设备读取失败", str(exc))
            return

        self.input_devices.clear()
        self.output_devices.clear()
        self.input_combo.clear()
        self.output_combo.clear()

        for index, device in enumerate(self.devices):
            name = str(device["name"])
            if int(device["max_input_channels"]) > 0:
                suffix = " (默认)" if index == default_in else ""
                label = f"[{index}] {name}{suffix}"
                self.input_devices.append((index, label))
                self.input_combo.addItem(label)
            if int(device["max_output_channels"]) > 0:
                suffix = " (默认)" if index == default_out else ""
                label = f"[{index}] {name}{suffix}"
                self.output_devices.append((index, label))
                self.output_combo.addItem(label)

        self._select_default(self.input_combo, self.input_devices, default_in)
        self._select_default(self.output_combo, self.output_devices, default_out)

        if log:
            self._log("设备列表已刷新")

    def _select_default(self, combo: QComboBox, choices: list[tuple[int, str]], default_index: int | None) -> None:
        for row, (index, _) in enumerate(choices):
            if index == default_index:
                combo.setCurrentIndex(row)
                return
        if choices:
            combo.setCurrentIndex(0)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择反击音频",
            "",
            "音频文件 (*.mp3 *.wav *.aiff *.aif *.flac *.ogg);;所有文件 (*.*)",
        )
        if not path:
            return
        self.audio_path = Path(path)
        self.file_label.setText(self.audio_path.name)
        self.status_label.setText("就绪 - 点击开始监听")
        self._log(f"已选择音频: {self.audio_path.name}")

    def _selected_index(self, choices: list[tuple[int, str]], combo: QComboBox) -> int | None:
        row = combo.currentIndex()
        if 0 <= row < len(choices):
            return choices[row][0]
        return None

    def _start(self) -> None:
        if self.running:
            return
        if self.audio_path is None:
            QMessageBox.information(self, "请选择音频", "先选择一个反击音频文件，或恢复 assets/default_alert.wav。")
            return
        input_index = self._selected_index(self.input_devices, self.input_combo)
        if input_index is None:
            QMessageBox.critical(self, "没有输入设备", "请刷新并选择麦克风。")
            return

        self.running = True
        self.event_times.clear()
        self.started_at = time.monotonic()
        self.last_fire_at = 0.0
        self.previous_spectrum = None
        self.clear_block_counts.clear()
        self._begin_calibration()
        self._log("开始监听，先自动校准底噪")

        try:
            self.stream = sd.InputStream(
                device=input_index,
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=BLOCK_SIZE,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            self.running = False
            self.status_label.setText("启动失败")
            QMessageBox.critical(self, "监听启动失败", str(exc))

    def _stop(self) -> None:
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.status_label.setText("已停止")
        self.event_times.clear()
        self.trigger_label.setText(f"0/{self.confirm_slider.value()}")
        self._log("已停止监听")

    def _audio_callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            self.audio_queue.put(np.zeros(BLOCK_SIZE, dtype=np.float32))
        self.audio_queue.put(indata[:, 0].copy())

    def _tick(self) -> None:
        self._process_audio()
        self._update_ui()

    def _process_audio(self) -> None:
        processed = 0
        while processed < 8:
            try:
                samples = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            self.metrics = self._analyze(samples)
            if self.running:
                if self._handle_calibration(self.metrics):
                    self._maybe_register_hit(self.metrics)

    def _analyze(self, samples: np.ndarray) -> NoiseMetrics:
        if len(samples) == 0:
            return NoiseMetrics()
        samples = samples.astype(np.float32)
        samples = samples - float(np.mean(samples))
        windowed = samples * np.hanning(len(samples))
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(len(windowed), d=1 / SAMPLE_RATE)

        total = float(np.sum(spectrum) + 1e-9)
        low = self._band_energy(spectrum, freqs, 40, 260)
        very_low = self._band_energy(spectrum, freqs, 25, 120)
        high = self._band_energy(spectrum, freqs, 2_000, 8_000)
        voice = self._band_energy(spectrum, freqs, 300, 3_400)
        scream = self._band_energy(spectrum, freqs, 1_500, 6_000)
        rms = float(np.sqrt(np.mean(samples * samples)))
        rms_db = 20 * math.log10(max(rms, 1e-7))
        above_db = max(0.0, rms_db - self.noise_floor_db)
        signal_gate = self._ramp(above_db, 4.0, 18.0)
        centroid_hz = float(np.sum(freqs * spectrum) / total)
        zcr = self._zero_crossing_rate(samples)
        harmonicity = self._harmonicity(samples)

        normalized = spectrum / total
        if self.previous_spectrum is None:
            flux = 0.0
        else:
            previous = self.previous_spectrum
            flux = float(np.sum(np.maximum(normalized - previous, 0.0)))
        self.previous_spectrum = normalized
        flux *= signal_gate

        low_ratio_raw = low / total
        high_ratio_raw = high / total
        voice_ratio_raw = voice / total
        scream_ratio_raw = scream / total
        above_floor = min(1.0, above_db / 36.0)
        low_ratio = min(1.0, low_ratio_raw * 2.8) * signal_gate
        voice_ratio = min(1.0, voice_ratio_raw * 1.6) * signal_gate
        scream_ratio = min(1.0, scream_ratio_raw * 1.55) * signal_gate
        impact = min(1.0, above_db / 38.0 + flux * 3.2)

        loud_score = self._ramp(above_db, 8.0, 32.0)
        onset_score = min(1.0, flux * 4.0)
        low_shape = self._bell(centroid_hz, 70.0, 700.0)
        voice_shape = self._bell(centroid_hz, 450.0, 2_800.0)
        scream_shape = self._bell(centroid_hz, 1_800.0, 7_000.0)
        zcr_voice = self._bell(zcr, 0.018, 0.18)
        zcr_scream = self._ramp(zcr, 0.055, 0.24)
        low_score = self._clamp01(
            loud_score * 0.35
            + onset_score * 0.24
            + min(1.0, low_ratio_raw * 4.0) * 0.27
            + low_shape * 0.14
            - min(0.30, high_ratio_raw * 0.75)
        ) * signal_gate
        voice_score = self._clamp01(
            loud_score * 0.25
            + min(1.0, voice_ratio_raw * 1.8) * 0.32
            + harmonicity * 0.25
            + zcr_voice * 0.12
            + voice_shape * 0.06
            - min(0.22, very_low / total * 1.1)
        ) * signal_gate
        scream_score = self._clamp01(
            loud_score * 0.28
            + onset_score * 0.14
            + min(1.0, scream_ratio_raw * 1.7) * 0.30
            + zcr_scream * 0.12
            + scream_shape * 0.10
            + min(1.0, high_ratio_raw * 2.2) * 0.06
            - min(0.18, low_ratio_raw * 0.9)
        ) * signal_gate
        score = max(low_score, voice_score, scream_score)
        return NoiseMetrics(
            above_floor,
            low_ratio,
            voice_ratio,
            scream_ratio,
            impact,
            score,
            low_score,
            voice_score,
            scream_score,
            rms_db,
            low_ratio_raw,
            high_ratio_raw,
            voice_ratio_raw,
            scream_ratio_raw,
            centroid_hz,
            zcr,
            harmonicity,
        )

    def _band_energy(self, spectrum: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
        return float(np.sum(spectrum[(freqs >= low_hz) & (freqs <= high_hz)]))

    def _zero_crossing_rate(self, samples: np.ndarray) -> float:
        if len(samples) < 2:
            return 0.0
        signs = np.signbit(samples)
        return float(np.mean(signs[1:] != signs[:-1]))

    def _harmonicity(self, samples: np.ndarray) -> float:
        if len(samples) < 512:
            return 0.0
        clipped = samples[: min(len(samples), 4096)]
        corr = np.correlate(clipped, clipped, mode="full")[len(clipped) - 1 :]
        base = float(corr[0] + 1e-9)
        min_lag = max(1, int(SAMPLE_RATE / 360))
        max_lag = min(len(corr) - 1, int(SAMPLE_RATE / 75))
        if max_lag <= min_lag:
            return 0.0
        peak = float(np.max(corr[min_lag:max_lag]))
        return self._clamp01(peak / base)

    def _ramp(self, value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return self._clamp01((value - low) / (high - low))

    def _bell(self, value: float, low: float, high: float) -> float:
        if value <= low or value >= high:
            return 0.0
        center = (low + high) / 2.0
        half_width = (high - low) / 2.0
        return self._clamp01(1.0 - abs(value - center) / half_width)

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _handle_calibration(self, metrics: NoiseMetrics) -> bool:
        now = time.monotonic()
        if now < self.calibrating_until:
            self.calibration_values.append(metrics.rms_db)
            return False

        if self.calibration_values:
            values = np.array(self.calibration_values, dtype=np.float32)
            self.noise_floor_db = float(np.percentile(values, 85))
            self.calibration_values.clear()
            self.status_label.setText("监听中")
            self._log(f"底噪校准完成: {self.noise_floor_db:.1f} dB")
            return False

        # Slow adaptive floor: follow stable ambience, ignore sudden loud blocks.
        if metrics.rms_db < self.noise_floor_db + 6:
            self.noise_floor_db = self.noise_floor_db * 0.995 + metrics.rms_db * 0.005
        return True

    def _maybe_register_hit(self, metrics: NoiseMetrics) -> None:
        now = time.monotonic()
        sensitivity = self.sensitivity_slider.value()
        targets = self._selected_targets()
        if not targets:
            self.clear_block_counts.clear()
            return
        in_cooldown = now - self.last_fire_at < self.cooldown_slider.value()
        matched_target = ""
        for target in targets:
            if self._matches_target(target, metrics, sensitivity):
                self.clear_block_counts[target] = self.clear_block_counts.get(target, 0) + 1
            else:
                self.clear_block_counts[target] = 0
            if self.clear_block_counts[target] >= self.stable_blocks_slider.value():
                matched_target = target

        if matched_target and not in_cooldown:
            if not self.event_times or now - self.event_times[-1] > 0.65:
                self.event_times.append(now)
                self.clear_block_counts[matched_target] = 0
                self._log_detection(matched_target, metrics)

        self.event_times = [item for item in self.event_times if now - item <= WINDOW_SECONDS]
        if len(self.event_times) >= self.confirm_slider.value() and not in_cooldown:
            self.event_times.clear()
            self.last_fire_at = now
            self.trigger_total += 1
            self.trigger_total_label.setText(str(self.trigger_total))
            self.status_label.setText("已触发 - 播放反击音频")
            self._log("达到确认次数，播放反击音频")
            threading.Thread(target=self._play_audio, daemon=True).start()

    def _selected_targets(self) -> list[str]:
        targets = []
        if self.low_target_check.isChecked():
            targets.append("低频冲击")
        if self.voice_target_check.isChecked():
            targets.append("人声")
        if self.scream_target_check.isChecked():
            targets.append("尖叫声")
        return targets

    def _log_detection(self, target: str, metrics: NoiseMetrics) -> None:
        self._log(
            f"确认一次{target}: 高于底噪 {metrics.rms_db - self.noise_floor_db:.1f} dB，"
            f"低频 {metrics.low_ratio_raw:.2f}，人声 {metrics.voice_ratio_raw:.2f}，"
            f"尖叫 {metrics.scream_ratio_raw:.2f}"
        )

    def _matches_target(self, target: str, metrics: NoiseMetrics, sensitivity: int) -> bool:
        above_db = metrics.rms_db - self.noise_floor_db
        required_above_db = self.above_db_slider.value() - (sensitivity - 2) * 0.6
        allowed_high_ratio = self.high_ratio_slider.value() / 100.0
        required_low_ratio = max(0.04, self.low_ratio_slider.value() / 100.0 - (sensitivity - 2) * 0.008)
        threshold = 0.82 - sensitivity * 0.025

        if target == "人声":
            voice_above_db = max(10.0, required_above_db - 12.0)
            voice_threshold = max(0.50, threshold - 0.10)
            return (
                above_db >= voice_above_db
                and metrics.voice_score >= voice_threshold
                and metrics.harmonicity >= 0.18
                and 0.012 <= metrics.zcr <= 0.20
                and metrics.low_ratio_raw <= 0.35
                and metrics.high_ratio_raw <= max(0.62, allowed_high_ratio)
            )

        if target == "尖叫声":
            scream_above_db = max(14.0, required_above_db - 8.0)
            scream_threshold = max(0.54, threshold - 0.08)
            return (
                above_db >= scream_above_db
                and metrics.scream_score >= scream_threshold
                and metrics.scream_ratio_raw >= max(0.30, 0.52 - sensitivity * 0.018)
                and metrics.centroid_hz >= 1_300
                and metrics.high_ratio_raw >= 0.20
                and metrics.low_ratio_raw <= 0.22
                and metrics.impact >= 0.28
            )

        low_threshold = max(0.56, threshold - 0.04)
        return (
            above_db >= required_above_db
            and metrics.impact >= 0.42
            and metrics.low_score >= low_threshold
            and metrics.low_ratio_raw >= required_low_ratio
            and metrics.high_ratio_raw <= allowed_high_ratio
        )

    def _play_audio(self) -> None:
        if self.audio_path is None:
            return
        output_index = self._selected_index(self.output_devices, self.output_combo)
        try:
            data, samplerate = sf.read(self.audio_path, always_2d=True, dtype="float32")
            sd.play(data, samplerate=samplerate, device=output_index, blocking=True)
            return
        except Exception as exc:
            self._log(f"指定设备播放失败，回退到系统输出: {exc}")

        try:
            subprocess.run(["afplay", str(self.audio_path)], check=False)
        except Exception as exc:
            self._log(f"播放失败: {exc}")

    def _update_ui(self) -> None:
        self.above_floor_bar.setValue(round(self.metrics.above_floor * 1000))
        self.low_ratio_bar.setValue(round(self.metrics.low_ratio * 1000))
        self.high_ratio_bar.setValue(round(min(1.0, self.metrics.high_ratio_raw) * self._display_signal_gate() * 1000))
        self.voice_ratio_bar.setValue(round(self.metrics.voice_ratio * 1000))
        self.scream_ratio_bar.setValue(round(self.metrics.scream_ratio * 1000))
        self.impact_bar.setValue(round(self.metrics.impact * 1000))
        self.score_bar.setValue(round(self._current_target_score() * 1000))
        self.floor_label.setText(f"底噪: {self.noise_floor_db:.1f} dB    当前: {self.metrics.rms_db:.1f} dB")
        self.trigger_label.setText(f"{len(self.event_times)}/{self.confirm_slider.value()}")

        if self.running:
            elapsed = int(time.monotonic() - self.started_at)
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            self.runtime_label.setText(f"{hours:02}:{minutes:02}:{seconds:02}")

            if time.monotonic() < self.calibrating_until:
                left = max(1, math.ceil(self.calibrating_until - time.monotonic()))
                self.status_label.setText(f"正在校准底噪 - {left}秒")
            elif self.last_fire_at and time.monotonic() - self.last_fire_at < self.cooldown_slider.value():
                left = max(0, self.cooldown_slider.value() - int(time.monotonic() - self.last_fire_at))
                self.status_label.setText(f"冷却中 - {left}秒")
            elif self.status_label.text().startswith(("冷却中", "已触发")):
                self.status_label.setText("监听中")

    def _current_target_score(self) -> float:
        scores = []
        if self.low_target_check.isChecked():
            scores.append(self.metrics.low_score)
        if self.voice_target_check.isChecked():
            scores.append(self.metrics.voice_score)
        if self.scream_target_check.isChecked():
            scores.append(self.metrics.scream_score)
        return max(scores, default=0.0)

    def _display_signal_gate(self) -> float:
        return self._ramp(self.metrics.rms_db - self.noise_floor_db, 4.0, 18.0)

    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"[{stamp}] {message}")

    def closeEvent(self, event) -> None:
        self._stop()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = CounterNoiseWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
