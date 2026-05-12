from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path

from PySide6.QtCharts import QChart, QChartView, QDateTimeAxis, QLineSeries, QValueAxis
from PySide6.QtCore import QDate, QDateTime, QMargins, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QPainter, QPen, QTextCursor
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QAbstractScrollArea,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)

from ..adapters.backtest_adapter import backtest_dir, build_powershell_command, project_root, write_runtime_config
from ..config import defaults as d
from ..config import theme
from ..config import texts as t
from ..runner import BacktestRunner
from ..validators import validate_parameters


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(t.APP_TITLE)
        self.resize(1300, 880)
        self.setStyleSheet(theme.APP_STYLESHEET)
        self.runner = BacktestRunner(self)
        self.result_path = ""
        self.factor_checks: dict[str, QCheckBox] = {}
        self.weight_spins: dict[str, QDoubleSpinBox] = {}
        self.regime_path_rows: dict[str, QWidget] = {}
        self.regime_sector_bias_checks: dict[str, QCheckBox] = {}
        self._syncing_sector_cap = False
        self._build_ui()
        self._connect_runner()
        self._update_effective_range()
        self._toggle_short_controls()
        self._toggle_dynamic_weight_controls()
        self._toggle_pool_controls()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel(t.APP_TITLE)
        title.setFont(QFont("SF Pro Display", 18, QFont.Bold))
        subtitle = QLabel(t.APP_SUBTITLE)
        subtitle.setStyleSheet("color: #90a6bf;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._run_setup_tab(), "Run Setup")
        self.tabs.addTab(self._pool_tab(), "Universe Filters")
        self.tabs.addTab(self._long_book_tab(), "Long Book")
        self.tabs.addTab(self._short_hedge_tab(), "Short Hedge")
        self.tabs.addTab(self._factor_tab(), "Factors")
        self.tabs.addTab(self._regimes_tab(), "Regimes")
        self.tabs.addTab(self._advanced_tab(), "Advanced / Risk Gate")
        self.tabs.addTab(self._results_tab(), "Results")
        self.tabs.addTab(self._logs_tab(), "Logs")
        layout.addWidget(self.tabs, stretch=1)
        self.tabs.setTabToolTip(0, t.TAB_RUN_SETUP_TOOLTIP)
        self.tabs.setTabToolTip(1, t.TAB_POOL_TOOLTIP)
        self.tabs.setTabToolTip(2, "Main long book sizing and purchase rules.")
        self.tabs.setTabToolTip(3, "Optional short hedge construction and basket settings.")
        self.tabs.setTabToolTip(4, t.TAB_FACTORS_TOOLTIP)
        self.tabs.setTabToolTip(5, "Regime detection, dynamic weights, and regime-aware post-score switches.")
        self.tabs.setTabToolTip(6, t.TAB_ADVANCED_TOOLTIP)
        self.tabs.setTabToolTip(7, "Latest run performance table and NAV/benchmark chart preview.")
        self.tabs.setTabToolTip(8, t.TAB_LOGS_TOOLTIP)

        bottom = QHBoxLayout()
        self.start_button = QPushButton(t.START_BACKTEST)
        self.stop_button = QPushButton(t.STOP_BACKTEST)
        self.stop_button.setEnabled(False)
        self.open_button = QPushButton(t.OPEN_RESULTS)
        self.open_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        bottom.addWidget(self.start_button)
        bottom.addWidget(self.stop_button)
        bottom.addWidget(self.open_button)
        bottom.addWidget(self.progress, stretch=1)
        layout.addLayout(bottom)
        self.start_button.clicked.connect(self._start_backtest)
        self.stop_button.clicked.connect(self.runner.stop)
        self.open_button.clicked.connect(self._open_results)
        self.setCentralWidget(root)
        self.statusBar().showMessage(t.STATUS_READY)

    def _run_setup_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(14)
        row = QHBoxLayout()
        row.setSpacing(14)
        row.addWidget(self._date_group(), stretch=1)
        row.addWidget(self._run_paths_group(), stretch=2)
        layout.addLayout(row)
        note = QLabel("Use this tab for schedule, J-Quants credential input, and runtime paths. Long-book rules and short-hedge construction now live in their own dedicated tabs.")
        note.setWordWrap(True)
        note.setObjectName("subtleText")
        layout.addWidget(note)
        layout.addWidget(self._config_management_group())
        layout.addStretch(1)
        return tab

    def _config_management_group(self) -> QGroupBox:
        box = QGroupBox("Config Presets")
        layout = QVBoxLayout(box)
        note = QLabel("Save the current GUI parameters to a JSON preset, or load a previously saved preset to rerun the same setup later.")
        note.setWordWrap(True)
        note.setObjectName("subtleText")
        layout.addWidget(note)
        row = QHBoxLayout()
        self.save_config_button = QPushButton("Save Config JSON")
        self.load_config_button = QPushButton("Load Config JSON")
        self.save_config_button.clicked.connect(self._save_config_json)
        self.load_config_button.clicked.connect(self._load_config_json)
        row.addWidget(self.save_config_button)
        row.addWidget(self.load_config_button)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _date_group(self) -> QGroupBox:
        box = QGroupBox(t.GROUP_DATES)
        layout = QVBoxLayout(box)
        note = QLabel("Choose the requested signal window and rebalance frequency. The effective range summary helps you sanity-check the run before launch.")
        note.setWordWrap(True)
        note.setObjectName("subtleText")
        layout.addWidget(note)
        form = QFormLayout()
        self.start_date = QDateEdit(QDate.fromString(d.DEFAULT_START, "yyyy-MM-dd"))
        self.end_date = QDateEdit(QDate.fromString(d.DEFAULT_END, "yyyy-MM-dd"))
        for widget in (self.start_date, self.end_date):
            widget.setCalendarPopup(True)
            widget.setDisplayFormat("yyyy-MM-dd")
            widget.dateChanged.connect(self._update_effective_range)
        self.frequency = QComboBox()
        for key, label in d.FREQUENCIES.items():
            self.frequency.addItem(label, key)
        self.effective_label = QLabel()
        self.effective_label.setWordWrap(True)
        self.effective_label.setObjectName("detailText")
        self.start_date.setToolTip("Requested first date of the backtest window.")
        self.end_date.setToolTip("Requested last date of the backtest window.")
        self.frequency.setToolTip("How often the strategy rebalances during the selected window.")
        form.addRow(t.START_DATE, self.start_date)
        form.addRow(t.END_DATE, self.end_date)
        form.addRow(t.FREQUENCY, self.frequency)
        form.addRow(t.EFFECTIVE_RANGE, self.effective_label)
        layout.addLayout(form)
        release_note = QLabel("This GUI uses a single formula-driven scoring workflow. Configure factor weights and the linear formula on the Factors tab.")
        release_note.setWordWrap(True)
        release_note.setObjectName("subtleText")
        layout.addWidget(release_note)
        return box

    def _pool_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(14)
        box = QGroupBox(t.GROUP_POOL)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.use_market_cap_filter = QCheckBox()
        self.use_market_cap_filter.setChecked(d.DEFAULT_FILTER_ENABLES["market_cap"])
        self.market_cap_mode = QComboBox()
        for key, label in d.MARKET_CAP_MODES.items():
            self.market_cap_mode.addItem(label, key)
        self.rank_min = self._spin(1, 10000, d.DEFAULT_RULES["market_cap"]["rank_min"])
        self.rank_max = self._spin(1, 10000, d.DEFAULT_RULES["market_cap"]["rank_max"])
        self.min_market_cap = self._double_spin(0, 1_000_000, d.DEFAULT_RULES["market_cap"]["min_billion_jpy"], decimals=1)
        self.max_market_cap = self._double_spin(0, 1_000_000, d.DEFAULT_RULES["market_cap"]["max_billion_jpy"], decimals=1)
        self.rank_min.setSuffix(" rank")
        self.rank_max.setSuffix(" rank")
        self.min_market_cap.setSuffix(" bn JPY")
        self.max_market_cap.setSuffix(" bn JPY")
        self.use_liquidity_filter = QCheckBox()
        self.use_liquidity_filter.setChecked(d.DEFAULT_FILTER_ENABLES["liquidity"])
        self.liquidity_lookback = self._spin(1, 500, d.DEFAULT_RULES["liquidity"]["lookback_days"])
        self.min_volume = self._spin(0, 100_000_000, d.DEFAULT_RULES["liquidity"]["min_avg_daily_volume"])
        self.liquidity_lookback.setSuffix(" d")
        self.min_volume.setSuffix(" JPY")
        self.use_volatility_filter = QCheckBox()
        self.use_volatility_filter.setChecked(d.DEFAULT_FILTER_ENABLES["volatility"])
        self.vol_lookback = self._spin(1, 500, d.DEFAULT_RULES["volatility"]["lookback_days"])
        self.max_vol = self._double_spin(0.01, 5.0, d.DEFAULT_RULES["volatility"]["max_annualized_volatility"], decimals=2, step=0.05)
        self.vol_lookback.setSuffix(" d")
        self.max_vol.setSuffix(" x")
        self.selection_top_n = self._spin(1, 5000, d.DEFAULT_RULES["selection"]["top_n_if_over_200"])
        self.selection_top_n.setSuffix(" names")
        self.selection_sort = QComboBox()
        for key, label in d.SORT_OPTIONS.items():
            self.selection_sort.addItem(label, key)
        self.use_max_lot_cost_filter = QCheckBox()
        self.use_max_lot_cost_filter.setChecked(d.DEFAULT_FILTER_ENABLES["max_lot_cost"])
        self.max_lot_cost_jpy = self._double_spin(0.0, 1_000_000_000.0, 0.0, decimals=0, step=10_000.0)
        self.max_lot_cost_jpy.setSuffix(" JPY")
        self.use_sector_constraints = QCheckBox()
        self.use_sector_constraints.setChecked(d.DEFAULT_FILTER_ENABLES["sector_constraints"])
        self.max_names_per_sector = self._spin(1, 1000, d.DEFAULT_BACKTEST["portfolio_constraints"]["max_names_per_sector"])
        self.sector_cap_mode = QComboBox()
        for key, label in d.SECTOR_CAP_MODES.items():
            self.sector_cap_mode.addItem(label, key)
        default_sector_cap_mode = d.DEFAULT_BACKTEST["portfolio_constraints"].get("sector_cap_mode", "hard")
        self.sector_cap_mode.setCurrentIndex(max(0, self.sector_cap_mode.findData(default_sector_cap_mode)))

        for checkbox in (
            self.use_market_cap_filter,
            self.use_liquidity_filter,
            self.use_volatility_filter,
            self.use_max_lot_cost_filter,
            self.use_sector_constraints,
        ):
            checkbox.toggled.connect(self._toggle_pool_controls)

        form.addRow(t.ENABLE_MARKET_CAP, self.use_market_cap_filter)
        form.addRow(t.MARKET_CAP_MODE, self.market_cap_mode)
        form.addRow(t.RANK_MIN, self.rank_min)
        form.addRow(t.RANK_MAX, self.rank_max)
        form.addRow(t.MIN_MARKET_CAP, self.min_market_cap)
        form.addRow(t.MAX_MARKET_CAP, self.max_market_cap)
        form.addRow(t.ENABLE_LIQUIDITY, self.use_liquidity_filter)
        form.addRow(t.LIQUIDITY_LOOKBACK, self.liquidity_lookback)
        form.addRow(t.MIN_VOLUME, self.min_volume)
        form.addRow(t.ENABLE_VOLATILITY, self.use_volatility_filter)
        form.addRow(t.VOL_LOOKBACK, self.vol_lookback)
        form.addRow(t.MAX_VOL, self.max_vol)
        form.addRow(t.SELECTION_TOP_N, self.selection_top_n)
        form.addRow(t.SELECTION_SORT, self.selection_sort)
        form.addRow(t.ENABLE_MAX_LOT_COST, self.use_max_lot_cost_filter)
        form.addRow(t.MAX_LOT_COST, self.max_lot_cost_jpy)
        layout.addWidget(box)
        layout.addStretch(1)
        return tab

    def _factor_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setSpacing(12)
        split = QHBoxLayout()
        split.setSpacing(14)
        factors_box = QGroupBox("Factors")
        factors_layout = QVBoxLayout(factors_box)
        release_banner = QLabel("Linear factor weighting only. No strategy-mode switch or precomputed model preset is included in this build.")
        release_banner.setWordWrap(True)
        release_banner.setStyleSheet("color: #6fd3ff; font-weight: 700;")
        factors_layout.addWidget(release_banner)
        factors_hint = QLabel("Toggle each module here. These switches decide which raw signals are available to the composite score.")
        factors_hint.setWordWrap(True)
        factors_hint.setObjectName("subtleText")
        factors_layout.addWidget(factors_hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        for idx, factor in enumerate(d.FACTORS):
            card = self._factor_card(factor)
            grid.addWidget(card, idx // 2, idx % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        scroll.setWidget(content)
        factors_layout.addWidget(scroll)
        split.addWidget(factors_box, stretch=2)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_content = QWidget()
        right_panel = QVBoxLayout(right_content)
        right_panel.setSpacing(12)

        weights_box = QGroupBox("Factor Weights")
        weights_panel = QVBoxLayout(weights_box)
        weights_note = QLabel("These are direct factor weights for the public linear score. Adjust them yourself instead of relying on any pre-tuned house model.")
        weights_note.setWordWrap(True)
        weights_note.setObjectName("subtleText")
        weights_panel.addWidget(weights_note)
        weights_grid = QGridLayout()
        weights_grid.setHorizontalSpacing(16)
        weights_grid.setVerticalSpacing(10)
        for idx, (name, value) in enumerate(d.DEFAULT_COMPOSITE_WEIGHTS.items()):
            spin = self._double_spin(0.0, 2.0, value, decimals=3, step=0.05)
            spin.setToolTip(d.WEIGHT_DESCRIPTIONS.get(name, ""))
            self.weight_spins[name] = spin
            label = QLabel(d.WEIGHT_DISPLAY_NAMES.get(name, name))
            label.setTextFormat(Qt.RichText)
            label.setToolTip(name)
            desc = QLabel(d.WEIGHT_DESCRIPTIONS.get(name, ""))
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #90a6bf; font-size: 11px;")
            row = idx // 2
            col = (idx % 2) * 3
            weights_grid.addWidget(label, row, col)
            weights_grid.addWidget(spin, row, col + 1)
            weights_grid.addWidget(desc, row, col + 2)
        weights_panel.addLayout(weights_grid)
        right_panel.addWidget(weights_box)

        formula_box = QGroupBox("Linear Formula")
        formula_panel = QVBoxLayout(formula_box)
        formula_display = QLabel(d.COMPOSITE_FORMULA_HTML)
        formula_display.setTextFormat(Qt.RichText)
        formula_display.setWordWrap(True)
        formula_display.setStyleSheet("background: #0d1522; border: 1px solid #24415f; padding: 10px; border-radius: 8px;")
        explanation = QLabel(d.COMPOSITE_EXPLANATION_HTML)
        explanation.setTextFormat(Qt.RichText)
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #90a6bf;")
        formula_panel.addWidget(formula_display)
        formula_panel.addWidget(explanation)
        self.formula = QLineEdit(d.DEFAULT_FORMULA)
        self.formula.setToolTip("Formula is validated before launch. This public build expects you to define the linear factor blend yourself.")
        self.restore_formula_button = QPushButton(t.RESTORE_DEFAULTS)
        self.restore_formula_button.clicked.connect(self._restore_factor_defaults)
        editable_label = QLabel(f"{t.FORMULA} (internal variables)")
        editable_label.setToolTip("Editable formula keeps ASCII variable names for validation and config compatibility.")
        formula_panel.addWidget(editable_label)
        formula_panel.addWidget(self.formula)
        formula_panel.addWidget(self.restore_formula_button)
        right_panel.addWidget(formula_box)
        right_panel.addStretch(1)
        right_scroll.setWidget(right_content)
        split.addWidget(right_scroll, stretch=2)
        outer.addLayout(split)
        return tab

    def _regimes_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        intro = QLabel(
            "Use this page for market-state logic. Regime detection and allocation-gate overrides remain available, "
            "but composite blending is controlled manually on the Factors tab instead of loading private weight models."
        )
        intro.setWordWrap(True)
        intro.setObjectName("subtleText")
        layout.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        panel = QVBoxLayout(content)
        panel.setSpacing(12)

        gate_box = QGroupBox("Allocation Gate")
        gate_form = QFormLayout(gate_box)
        gate_note = QLabel(
            "The unified allocation gate combines regime detection with the market risk score. "
            "High risk can override the weight regime and shrink exposure, while the original active regime still stays available for diagnostics."
        )
        gate_note.setWordWrap(True)
        gate_note.setObjectName("subtleText")
        gate_form.addRow(gate_note)
        self.allocation_gate_enabled = QCheckBox()
        self.allocation_gate_enabled.setChecked(d.DEFAULT_ALLOCATION_GATE["enabled"])
        self.allocation_gate_mode = QComboBox()
        for key, label in d.ALLOCATION_GATE_MODES.items():
            self.allocation_gate_mode.addItem(label, key)
        self.allocation_gate_mode.setCurrentIndex(
            max(0, self.allocation_gate_mode.findData(d.DEFAULT_ALLOCATION_GATE["mode"]))
        )
        self.allocation_gate_risk_override_threshold = self._spin(1, 5, d.DEFAULT_ALLOCATION_GATE["risk_override_threshold"])
        self.allocation_gate_severe_risk_score_threshold = self._spin(1, 5, d.DEFAULT_ALLOCATION_GATE["severe_risk_score_threshold"])
        self.allocation_gate_defensive_regime = QComboBox()
        for regime_name in d.REGIME_NAMES:
            self.allocation_gate_defensive_regime.addItem(regime_name, regime_name)
        self.allocation_gate_defensive_regime.setCurrentIndex(
            max(0, self.allocation_gate_defensive_regime.findData(d.DEFAULT_ALLOCATION_GATE["defensive_regime"]))
        )
        self.allocation_gate_risk_override_threshold.setSuffix(" pts")
        self.allocation_gate_severe_risk_score_threshold.setSuffix(" pts")
        gate_form.addRow("Enable allocation gate", self.allocation_gate_enabled)
        gate_form.addRow("Gate mode", self.allocation_gate_mode)
        gate_form.addRow("Risk override score", self.allocation_gate_risk_override_threshold)
        gate_form.addRow("Severe risk score", self.allocation_gate_severe_risk_score_threshold)
        gate_form.addRow("Defensive regime", self.allocation_gate_defensive_regime)
        panel.addWidget(gate_box)

        regime_box = QGroupBox("Regime Detection & Weight Files")
        regime_form = QFormLayout(regime_box)
        self.use_dynamic_weights = QCheckBox()
        self.use_dynamic_weights.setChecked(False)
        self.use_dynamic_weights.setEnabled(False)
        self.use_dynamic_weights.toggled.connect(self._toggle_dynamic_weight_controls)
        self.weight_source_mode = QComboBox()
        for key, label in d.WEIGHT_SOURCE_MODES.items():
            self.weight_source_mode.addItem(label, key)
        self.weight_source_mode.setCurrentIndex(max(0, self.weight_source_mode.findData("manual")))
        self.weight_source_mode.setEnabled(False)
        regime_form.addRow("Enable regime-aware weights", self.use_dynamic_weights)
        regime_form.addRow("Weight source", self.weight_source_mode)
        self.trend_strong_threshold = self._double_spin(0.0, 5.0, d.DEFAULT_REGIME_CONFIG["trend_strong_threshold"], decimals=2, step=0.05)
        self.trend_weak_threshold = self._double_spin(0.0, 5.0, d.DEFAULT_REGIME_CONFIG["trend_weak_threshold"], decimals=2, step=0.05)
        self.vol_high_enter = self._double_spin(0.0, 1.0, d.DEFAULT_REGIME_CONFIG["vol_high_enter"], decimals=2, step=0.05)
        self.vol_high_exit = self._double_spin(0.0, 1.0, d.DEFAULT_REGIME_CONFIG["vol_high_exit"], decimals=2, step=0.05)
        self.vol_low_enter = self._double_spin(0.0, 1.0, d.DEFAULT_REGIME_CONFIG["vol_low_enter"], decimals=2, step=0.05)
        self.vol_low_exit = self._double_spin(0.0, 1.0, d.DEFAULT_REGIME_CONFIG["vol_low_exit"], decimals=2, step=0.05)
        self.switch_confirm_bars = self._spin(1, 30, d.DEFAULT_REGIME_CONFIG["switch_confirm_bars"])
        self.overlay_confirm_bars = self._spin(1, 30, d.DEFAULT_REGIME_CONFIG["overlay_confirm_bars"])
        self.trend_strong_threshold.setSuffix(" x")
        self.trend_weak_threshold.setSuffix(" x")
        self.vol_high_enter.setSuffix(" q")
        self.vol_high_exit.setSuffix(" q")
        self.vol_low_enter.setSuffix(" q")
        self.vol_low_exit.setSuffix(" q")
        self.switch_confirm_bars.setSuffix(" bars")
        self.overlay_confirm_bars.setSuffix(" bars")
        regime_form.addRow("Trend strong", self.trend_strong_threshold)
        regime_form.addRow("Trend weak", self.trend_weak_threshold)
        regime_form.addRow("Vol high enter", self.vol_high_enter)
        regime_form.addRow("Vol high exit", self.vol_high_exit)
        regime_form.addRow("Vol low enter", self.vol_low_enter)
        regime_form.addRow("Vol low exit", self.vol_low_exit)
        regime_form.addRow("Switch confirm", self.switch_confirm_bars)
        regime_form.addRow("Overlay confirm", self.overlay_confirm_bars)
        for regime_name in d.REGIME_NAMES:
            row = self._path_row(f"weights\\{regime_name}.json", file_mode=True, base_dir=backtest_dir())
            self.regime_path_rows[regime_name] = row
            regime_form.addRow(f"{regime_name} weights", row)
        panel.addWidget(regime_box)

        rotation_box = QGroupBox("Regime Rotation Switches")
        rotation_layout = QVBoxLayout(rotation_box)
        rotation_note = QLabel(
            "These switches control regime-aware post-score behavior such as sector rotation and macro reflation post-ranking filters."
        )
        rotation_note.setWordWrap(True)
        rotation_note.setObjectName("subtleText")
        rotation_layout.addWidget(rotation_note)
        rotation_form = QFormLayout()
        self.regime_sector_bias_regime_source = QComboBox()
        for key, label in d.REGIME_SECTOR_BIAS_SOURCES.items():
            self.regime_sector_bias_regime_source.addItem(label, key)
        self.regime_sector_bias_regime_source.setCurrentIndex(
            max(0, self.regime_sector_bias_regime_source.findData(d.DEFAULT_BACKTEST["post_score_adjustments"]["regime_sector_bias_regime_source"]))
        )
        rotation_form.addRow("Sector bias regime source", self.regime_sector_bias_regime_source)
        for regime_name in d.REGIME_NAMES:
            check = QCheckBox()
            check.setChecked(bool(d.DEFAULT_BACKTEST["post_score_adjustments"].get(f"{regime_name}_sector_bias_enabled", False)))
            self.regime_sector_bias_checks[regime_name] = check
            rotation_form.addRow(f"{regime_name} sector bias", check)
        self.macro_reflation_sector_alignment_enabled = QCheckBox()
        self.macro_reflation_sector_alignment_enabled.setChecked(
            bool(d.DEFAULT_BACKTEST["post_score_adjustments"]["macro_reflation_sector_alignment_enabled"])
        )
        self.macro_reflation_size_tilt_enabled = QCheckBox()
        self.macro_reflation_size_tilt_enabled.setChecked(
            bool(d.DEFAULT_BACKTEST["post_score_adjustments"]["macro_reflation_size_tilt_enabled"])
        )
        self.macro_reflation_leader_bias_enabled = QCheckBox()
        self.macro_reflation_leader_bias_enabled.setChecked(
            bool(d.DEFAULT_BACKTEST["post_score_adjustments"]["macro_reflation_leader_bias_enabled"])
        )
        rotation_form.addRow("macro_reflation sector alignment", self.macro_reflation_sector_alignment_enabled)
        rotation_form.addRow("macro_reflation size tilt", self.macro_reflation_size_tilt_enabled)
        rotation_form.addRow("macro_reflation leader bias", self.macro_reflation_leader_bias_enabled)
        rotation_layout.addLayout(rotation_form)
        panel.addWidget(rotation_box)

        panel.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        return tab

    def _factor_card(self, factor: d.FactorDefinition) -> QWidget:
        card = QFrame()
        card.setObjectName("helpCard")
        card.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        header = QHBoxLayout()
        check = QCheckBox()
        check.setChecked(factor.default_enabled)
        self.factor_checks[factor.key] = check
        title = QLabel(f"{factor.label}")
        title.setStyleSheet("font-weight: 600;")
        badge = QLabel(factor.variable)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet("background: #173852; color: #7fd8ff; border-radius: 10px; padding: 2px 8px; font-weight: 600;")
        header.addWidget(check)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(badge)
        desc = QLabel(factor.description)
        desc.setWordWrap(True)
        desc.setObjectName("subtleText")
        layout.addLayout(header)
        layout.addWidget(desc)
        return card

    def _long_book_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(14)

        intro = QLabel("This tab controls the main long portfolio: account size, number of names, and gross exposure. Use it as the primary place to tune long-book purchase rules.")
        intro.setWordWrap(True)
        intro.setObjectName("subtleText")
        layout.addWidget(intro)
        self.core3_long_note = QLabel()
        self.core3_long_note.setWordWrap(True)
        self.core3_long_note.setObjectName("subtleText")
        layout.addWidget(self.core3_long_note)

        core_box = QGroupBox("Long Book Construction")
        core_layout = QVBoxLayout(core_box)
        core_form = QFormLayout()
        self.initial_capital = self._double_spin(1, 1_000_000_000_000, d.DEFAULT_BACKTEST.get("initial_capital", 100_000_000), decimals=0, step=1_000_000)
        self.initial_capital.setSuffix(" JPY")
        self.target_count = self._spin(1, 1000, d.DEFAULT_BACKTEST["target_count"])
        self.target_count.setSuffix(" names")
        self.gross_exposure = self._double_spin(0.01, 2.0, d.DEFAULT_BACKTEST["max_gross_exposure"], decimals=2, step=0.05)
        self.gross_exposure.setSuffix(" x")
        self.gross_exposure.setToolTip("Gross exposure of the long book before any optional short hedge is added.")
        self.target_count.setToolTip("Number of top-ranked names in the main long portfolio.")
        self.initial_capital.setToolTip("Capital used by the backtest to size positions.")
        core_form.addRow(t.INITIAL_CAPITAL, self.initial_capital)
        core_form.addRow(t.TARGET_COUNT, self.target_count)
        core_form.addRow(t.GROSS_EXPOSURE, self.gross_exposure)
        core_layout.addLayout(core_form)
        layout.addWidget(core_box)

        constraints_box = QGroupBox("Long Book Constraints")
        constraints_form = QFormLayout(constraints_box)
        constraints_note = QLabel("Sector controls live here because they shape the final long book. The current strategy uses equal weight, so single-name concentration is driven mainly by target count.")
        constraints_note.setWordWrap(True)
        constraints_note.setObjectName("subtleText")
        constraints_form.addRow(constraints_note)
        self.max_names_per_sector.setSuffix(" names")
        self.sector_cap_percent = self._double_spin(0.0, 100.0, 0.0, decimals=1, step=1.0)
        self.sector_cap_percent.setSuffix(" %")
        self.sector_cap_percent.setToolTip("Helper input for sector concentration. The GUI converts this percentage into max names per sector using the current target count.")
        constraints_form.addRow(t.ENABLE_SECTOR_CONSTRAINTS, self.use_sector_constraints)
        constraints_form.addRow("Sector cap", self.sector_cap_percent)
        constraints_form.addRow(t.MAX_NAMES_PER_SECTOR, self.max_names_per_sector)
        constraints_form.addRow(t.SECTOR_CAP_MODE, self.sector_cap_mode)
        self.long_single_weight_hint = QLineEdit()
        self.long_single_weight_hint.setReadOnly(True)
        self.long_single_weight_hint.setToolTip("Equal-weight long book implies an approximate per-name target before execution rounding and risk controls.")
        constraints_form.addRow("Equal-weight single-name target", self.long_single_weight_hint)
        layout.addWidget(constraints_box)
        self.target_count.valueChanged.connect(self._update_long_single_weight_hint)
        self.target_count.valueChanged.connect(self._sync_sector_cap_percent_from_names)
        self.max_names_per_sector.valueChanged.connect(self._sync_sector_cap_percent_from_names)
        self.sector_cap_percent.valueChanged.connect(self._sync_sector_names_from_percent)
        self._update_long_single_weight_hint()
        self._sync_sector_cap_percent_from_names()
        layout.addStretch(1)
        return tab

    def _short_hedge_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(14)

        self.enable_short_leg = QCheckBox()
        self.enable_short_leg.setChecked(d.DEFAULT_BACKTEST["enable_short_leg"])
        self.enable_short_leg.toggled.connect(self._toggle_short_controls)
        self.short_gross_exposure = self._double_spin(0.0, 1.5, d.DEFAULT_BACKTEST["short_gross_exposure"], decimals=2, step=0.05)
        self.short_gross_exposure.setSuffix(" x")
        self.short_target_count = self._spin(1, 1000, d.DEFAULT_BACKTEST["short_target_count"])
        self.short_target_count.setSuffix(" names")
        self.short_construction_mode = QComboBox()
        for key, label in d.SHORT_CONSTRUCTION_MODES.items():
            self.short_construction_mode.addItem(label, key)
        self.short_construction_mode.setCurrentIndex(max(0, self.short_construction_mode.findData(d.DEFAULT_BACKTEST["short_construction_mode"])))
        self.short_weighting_mode = QComboBox()
        for key, label in d.SHORT_WEIGHTING_MODES.items():
            self.short_weighting_mode.addItem(label, key)
        self.short_weighting_mode.setCurrentIndex(max(0, self.short_weighting_mode.findData(d.DEFAULT_BACKTEST["short_weighting_mode"])))
        self.short_expansion_mode = QComboBox()
        for key, label in d.SHORT_EXPANSION_MODES.items():
            self.short_expansion_mode.addItem(label, key)
        self.short_expansion_mode.setCurrentIndex(max(0, self.short_expansion_mode.findData(d.DEFAULT_BACKTEST["short_expansion_mode"])))
        self.short_adv_threshold = self._double_spin(0.0, 10_000_000_000.0, float(d.DEFAULT_BACKTEST["short_adv_threshold"] or 0.0), decimals=0, step=10_000_000.0)
        self.short_adv_threshold.setSuffix(" JPY")
        self.short_sector_cap = self._double_spin(0.0, 1.0, float(d.DEFAULT_BACKTEST["short_sector_cap"] or 0.0), decimals=2, step=0.05)
        self.short_sector_cap.setSuffix(" share")
        self.short_min_names = self._spin(1, 1000, d.DEFAULT_BACKTEST["short_min_names"])
        self.short_min_names.setSuffix(" names")
        self.short_max_single_weight = self._double_spin(0.01, 1.0, d.DEFAULT_BACKTEST["short_max_single_weight"], decimals=3, step=0.025)
        self.short_max_single_weight.setSuffix(" share")
        self.short_gross_exposure.setToolTip("Target gross exposure of the short hedge leg. Example: 0.20 means 20% gross short.")
        self.short_target_count.setToolTip("Used by simple short mode. For advanced bottom-decile mode, this acts as a fallback only.")
        self.short_construction_mode.setToolTip("Choose between the simple bottom-N short leg and the research-style advanced bottom-decile basket.")
        self.short_weighting_mode.setToolTip("How the advanced short basket allocates weight across selected names.")
        self.short_expansion_mode.setToolTip("How far the advanced short basket can expand when filling names beyond the core bottom decile.")
        self.short_adv_threshold.setToolTip("Optional ADV60 threshold for advanced short selection. Set 0 to disable.")
        self.short_sector_cap.setToolTip("Optional short-side sector cap as a weight share. Set 0 to disable.")
        self.short_min_names.setToolTip("Minimum number of names required for the advanced short basket.")
        self.short_max_single_weight.setToolTip("Maximum absolute weight allowed for any single short name.")

        intro = QLabel("This tab is dedicated to the optional short hedge. It exposes both the simple bottom-N mode and the research-style advanced bottom-decile basket.")
        intro.setWordWrap(True)
        intro.setObjectName("subtleText")
        layout.addWidget(intro)

        base_box = QGroupBox("Short Hedge Basics")
        base_form = QFormLayout(base_box)
        note = QLabel(t.SHORT_HEDGE_NOTE)
        note.setWordWrap(True)
        note.setObjectName("subtleText")
        base_form.addRow(note)
        base_form.addRow(t.ENABLE_SHORT_HEDGE, self.enable_short_leg)
        base_form.addRow(t.SHORT_GROSS_EXPOSURE, self.short_gross_exposure)
        base_form.addRow(t.SHORT_TARGET_COUNT, self.short_target_count)
        layout.addWidget(base_box)

        advanced_box = QGroupBox("Advanced Short Basket")
        advanced_form = QFormLayout(advanced_box)
        advanced_note = QLabel("Use these settings when you want the research-style bottom-decile short basket rather than a simple bottom-N hedge.")
        advanced_note.setWordWrap(True)
        advanced_note.setObjectName("subtleText")
        advanced_form.addRow(advanced_note)
        advanced_form.addRow(t.SHORT_CONSTRUCTION_MODE, self.short_construction_mode)
        advanced_form.addRow(t.SHORT_WEIGHTING_MODE, self.short_weighting_mode)
        advanced_form.addRow(t.SHORT_EXPANSION_MODE, self.short_expansion_mode)
        advanced_form.addRow(t.SHORT_ADV_THRESHOLD, self.short_adv_threshold)
        advanced_form.addRow(t.SHORT_SECTOR_CAP, self.short_sector_cap)
        advanced_form.addRow(t.SHORT_MIN_NAMES, self.short_min_names)
        advanced_form.addRow(t.SHORT_MAX_SINGLE_WEIGHT, self.short_max_single_weight)
        layout.addWidget(advanced_box)
        layout.addStretch(1)
        return tab

    def _run_paths_group(self) -> QGroupBox:
        box = QGroupBox("Files and Directories")
        layout = QVBoxLayout(box)
        note = QLabel("Enter the J-Quants API key here. It is written only to a temporary runtime file when the backtest starts. This GUI uses the built-in default cache and result directories.")
        note.setWordWrap(True)
        note.setObjectName("subtleText")
        layout.addWidget(note)
        form = QFormLayout()
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.api_key.setToolTip("Paste a J-Quants API key here. This is the simplest credential flow.")
        form.addRow(t.API_KEY, self.api_key)
        layout.addLayout(form)
        return box

    def _advanced_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        intro = QLabel("Execution settings affect trading assumptions. Risk gate settings control when the strategy scales exposure down under stressed market conditions.")
        intro.setWordWrap(True)
        intro.setObjectName("subtleText")
        layout.addWidget(intro)

        self.transaction_cost = self._double_spin(0, 1000, d.DEFAULT_BACKTEST["transaction_cost_bps"], decimals=1, step=1)
        self.slippage = self._double_spin(0, 1000, d.DEFAULT_BACKTEST["slippage_bps"], decimals=1, step=1)
        self.lot_size = self._spin(1, 10000, d.DEFAULT_BACKTEST["lot_size"])
        self.transaction_cost.setSuffix(" bps")
        self.slippage.setSuffix(" bps")
        self.lot_size.setSuffix(" sh")
        self.round_lots = QCheckBox()
        self.round_lots.setChecked(d.DEFAULT_BACKTEST["round_lots"])
        self.use_risk = QCheckBox()
        self.use_risk.setChecked(d.DEFAULT_BACKTEST["use_risk_gating"])
        self.include_sp500 = QCheckBox()
        self.include_sp500.setChecked(d.DEFAULT_BACKTEST["include_sp500_benchmark"])
        self.force_rebuild = QCheckBox()
        self.transaction_cost.setToolTip("Trading cost applied to every fill in basis points.")
        self.slippage.setToolTip("Additional execution slippage in basis points.")
        self.lot_size.setToolTip("Board lot or share increment used by position sizing.")
        self.round_lots.setToolTip("If enabled, orders are rounded down to the configured lot size.")
        self.use_risk.setToolTip("Allow the risk gate to reduce gross exposure when market stress increases.")
        self.include_sp500.setToolTip("Try to include an S&P 500 benchmark series in result comparisons.")
        self.force_rebuild.setToolTip("Force the local cache to rebuild before running.")

        top_split = QHBoxLayout()
        top_split.setSpacing(14)

        execution_box = QGroupBox("Execution")
        execution_form = QFormLayout(execution_box)
        execution_note = QLabel("These parameters define how signals are translated into trades.")
        execution_note.setWordWrap(True)
        execution_note.setObjectName("subtleText")
        execution_form.addRow(execution_note)
        execution_form.addRow(t.TRANSACTION_COST, self.transaction_cost)
        execution_form.addRow(t.SLIPPAGE, self.slippage)
        execution_form.addRow(t.LOT_SIZE, self.lot_size)
        execution_form.addRow(t.ROUND_LOTS, self.round_lots)
        top_split.addWidget(execution_box, stretch=1)

        runtime_box = QGroupBox("Runtime Options")
        runtime_form = QFormLayout(runtime_box)
        runtime_note = QLabel("These switches affect benchmark output, cache behavior, and whether the market risk overlay is active inside the allocation gate.")
        runtime_note.setWordWrap(True)
        runtime_note.setObjectName("subtleText")
        runtime_form.addRow(runtime_note)
        runtime_form.addRow(t.USE_RISK_GATING, self.use_risk)
        runtime_form.addRow(t.INCLUDE_SP500, self.include_sp500)
        runtime_form.addRow(t.FORCE_REBUILD, self.force_rebuild)
        top_split.addWidget(runtime_box, stretch=1)

        layout.addLayout(top_split)

        self.risk_lookback_high = self._spin(20, 1000, d.DEFAULT_RISK["lookback_high"])
        self.risk_rv_window = self._spin(5, 252, d.DEFAULT_RISK["rv_window"])
        self.risk_rv_q_window = self._spin(20, 1000, d.DEFAULT_RISK["rv_q_window"])
        self.risk_off_score = self._spin(1, 5, d.DEFAULT_RISK["risk_off_score"])
        self.risk_trend_short_ma = self._spin(5, 500, d.DEFAULT_RISK["trend_short_ma"])
        self.risk_trend_long_ma = self._spin(20, 1000, d.DEFAULT_RISK["trend_long_ma"])
        self.risk_trend_confirm_days = self._spin(1, 30, d.DEFAULT_RISK["trend_confirm_days"])
        self.risk_upgrade_confirm_days = self._spin(1, 30, d.DEFAULT_RISK["upgrade_confirm_days"])
        self.risk_downgrade_confirm_days = self._spin(1, 30, d.DEFAULT_RISK["downgrade_confirm_days"])
        self.risk_lookback_high.setSuffix(" d")
        self.risk_rv_window.setSuffix(" d")
        self.risk_rv_q_window.setSuffix(" d")
        self.risk_off_score.setSuffix(" pts")
        self.risk_trend_short_ma.setSuffix(" d")
        self.risk_trend_long_ma.setSuffix(" d")
        self.risk_trend_confirm_days.setSuffix(" d")
        self.risk_upgrade_confirm_days.setSuffix(" d")
        self.risk_downgrade_confirm_days.setSuffix(" d")

        risk_box = QGroupBox("Risk Gate Parameters")
        risk_layout = QVBoxLayout(risk_box)
        risk_note = QLabel("The risk gate combines breakout stress, realized volatility, and trend confirmation to decide whether exposure should be reduced.")
        risk_note.setWordWrap(True)
        risk_note.setObjectName("subtleText")
        risk_layout.addWidget(risk_note)

        risk_split = QHBoxLayout()
        risk_split.setSpacing(14)

        stress_box = QGroupBox("Stress Detection")
        stress_form = QFormLayout(stress_box)
        stress_form.addRow("Lookback high", self.risk_lookback_high)
        stress_form.addRow("Realized-vol window", self.risk_rv_window)
        stress_form.addRow("Realized-vol quantile window", self.risk_rv_q_window)
        stress_form.addRow("Risk-off score", self.risk_off_score)
        risk_split.addWidget(stress_box, stretch=1)

        trend_box = QGroupBox("Trend Confirmation")
        trend_form = QFormLayout(trend_box)
        trend_form.addRow("Trend short MA", self.risk_trend_short_ma)
        trend_form.addRow("Trend long MA", self.risk_trend_long_ma)
        trend_form.addRow("Trend confirm days", self.risk_trend_confirm_days)
        trend_form.addRow("Upgrade confirm days", self.risk_upgrade_confirm_days)
        trend_form.addRow("Downgrade confirm days", self.risk_downgrade_confirm_days)
        risk_split.addWidget(trend_box, stretch=1)

        risk_layout.addLayout(risk_split)
        layout.addWidget(risk_box)
        layout.addStretch(1)
        return tab

    def _logs_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        intro = QLabel("Watch live command output, saved paths, progress markers, and diagnostics while the backtest is running.")
        intro.setWordWrap(True)
        intro.setObjectName("subtleText")
        layout.addWidget(intro)
        controls = QHBoxLayout()
        clear = QPushButton("Clear Log")
        copy = QPushButton("Copy Log")
        clear.clicked.connect(lambda: self.log.clear())
        copy.clicked.connect(lambda: self.log.selectAll() or self.log.copy())
        controls.addWidget(clear)
        controls.addWidget(copy)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText(t.LOG_PLACEHOLDER)
        self.log.setMinimumHeight(620)
        layout.addWidget(self.log)
        return tab

    def _results_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)
        intro = QLabel("Review the latest finished run here. The summary table and NAV / benchmark chart update automatically after a backtest completes.")
        intro.setWordWrap(True)
        intro.setObjectName("subtleText")
        layout.addWidget(intro)

        self.result_preview_tag = QLabel("Result preview will appear after a run finishes.")
        self.result_preview_tag.setObjectName("detailText")
        layout.addWidget(self.result_preview_tag)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        summary_panel = QWidget()
        summary_layout = QVBoxLayout(summary_panel)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(8)
        summary_title = QLabel("Latest Run Performance Summary")
        summary_title.setStyleSheet("color: #66d1ff; font-weight: 600;")
        summary_layout.addWidget(summary_title)
        self.performance_table = QTableWidget(0, 7)
        self.performance_table.setHorizontalHeaderLabels(
            ["Series", "AnnRet", "MaxDD", "Vol", "Sharpe", "Sortino", "Excess"]
        )
        self.performance_table.verticalHeader().setVisible(False)
        self.performance_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.performance_table.setSelectionMode(QTableWidget.NoSelection)
        self.performance_table.setAlternatingRowColors(True)
        self.performance_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.performance_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.performance_table.setShowGrid(True)
        self.performance_table.setFocusPolicy(Qt.NoFocus)
        self.performance_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.performance_table.setStyleSheet(
            """
            QTableWidget {
                background: #15263b;
                alternate-background-color: #19304b;
                border: 1px solid #2d4d72;
                gridline-color: #2d4d72;
                color: #e7f0fb;
            }
            QHeaderView::section {
                background: #24415f;
                color: #f2f7fd;
                border: none;
                border-bottom: 1px solid #325980;
                padding: 8px 10px;
                font-weight: 600;
            }
            """
        )
        header = self.performance_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(True)
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setMinimumSectionSize(80)
        self.performance_table.setMaximumHeight(220)
        summary_layout.addWidget(self.performance_table)
        splitter.addWidget(summary_panel)

        chart_panel = QWidget()
        preview_layout = QVBoxLayout(chart_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        chart_title = QLabel("NAV vs Benchmarks")
        chart_title.setStyleSheet("color: #66d1ff; font-weight: 600;")
        preview_layout.addWidget(chart_title)
        self.benchmark_note = QLabel("Strategy is compared against TOPIX and any enabled benchmark series from the finished run.")
        self.benchmark_note.setWordWrap(True)
        self.benchmark_note.setObjectName("subtleText")
        preview_layout.addWidget(self.benchmark_note)
        self.nav_chart = QChart()
        self.nav_chart.setBackgroundVisible(False)
        self.nav_chart.legend().setVisible(True)
        self.nav_chart.legend().setLabelColor(Qt.white)
        self.nav_chart_view = QChartView(self.nav_chart)
        self.nav_chart_view.setRenderHint(QPainter.Antialiasing)
        self.nav_chart_view.setMinimumHeight(460)
        self.nav_chart_view.setStyleSheet("background: #0f1726; border: 1px solid #24415f; border-radius: 10px;")
        preview_layout.addWidget(self.nav_chart_view)
        splitter.addWidget(chart_panel)
        splitter.setSizes([170, 520])
        layout.addWidget(splitter, stretch=1)
        return tab

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(int(value))
        widget.setGroupSeparatorShown(True)
        widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
        return widget

    def _double_spin(self, minimum: float, maximum: float, value: float, *, decimals: int, step: float = 1.0) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(float(value))
        widget.setButtonSymbols(QAbstractSpinBox.NoButtons)
        return widget

    def _path_row(self, default: str, *, file_mode: bool, base_dir: Path | None = None) -> QWidget:
        wrap = QWidget()
        layout = QHBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        line = QLineEdit(str((base_dir or project_root()) / default))
        button = QPushButton("...")
        layout.addWidget(line, stretch=1)
        layout.addWidget(button)
        wrap.line_edit = line  # type: ignore[attr-defined]

        def choose() -> None:
            if file_mode:
                path, _ = QFileDialog.getOpenFileName(self, "Select file", line.text())
            else:
                path = QFileDialog.getExistingDirectory(self, "Select directory", line.text())
            if path:
                line.setText(path)

        button.clicked.connect(choose)
        return wrap

    def _restore_factor_defaults(self) -> None:
        for factor in d.FACTORS:
            self.factor_checks[factor.key].setChecked(factor.default_enabled)
        for name, value in d.DEFAULT_COMPOSITE_WEIGHTS.items():
            self.weight_spins[name].setValue(value)
        self.formula.setText(d.DEFAULT_FORMULA)
        self.allocation_gate_enabled.setChecked(d.DEFAULT_ALLOCATION_GATE["enabled"])
        self._set_combo_by_data(self.allocation_gate_mode, d.DEFAULT_ALLOCATION_GATE["mode"])
        self.allocation_gate_risk_override_threshold.setValue(d.DEFAULT_ALLOCATION_GATE["risk_override_threshold"])
        self.allocation_gate_severe_risk_score_threshold.setValue(d.DEFAULT_ALLOCATION_GATE["severe_risk_score_threshold"])
        self._set_combo_by_data(self.allocation_gate_defensive_regime, d.DEFAULT_ALLOCATION_GATE["defensive_regime"])
        self.use_dynamic_weights.setChecked(False)
        self.weight_source_mode.setCurrentIndex(max(0, self.weight_source_mode.findData("manual")))
        self._set_combo_by_data(
            self.regime_sector_bias_regime_source,
            d.DEFAULT_BACKTEST["post_score_adjustments"]["regime_sector_bias_regime_source"],
        )
        for regime_name, row in self.regime_path_rows.items():
            row.line_edit.setText("")  # type: ignore[attr-defined]
            if regime_name in self.regime_sector_bias_checks:
                self.regime_sector_bias_checks[regime_name].setChecked(
                    bool(d.DEFAULT_BACKTEST["post_score_adjustments"].get(f"{regime_name}_sector_bias_enabled", False))
                )
        self.macro_reflation_sector_alignment_enabled.setChecked(
            bool(d.DEFAULT_BACKTEST["post_score_adjustments"]["macro_reflation_sector_alignment_enabled"])
        )
        self.macro_reflation_size_tilt_enabled.setChecked(
            bool(d.DEFAULT_BACKTEST["post_score_adjustments"]["macro_reflation_size_tilt_enabled"])
        )
        self.macro_reflation_leader_bias_enabled.setChecked(
            bool(d.DEFAULT_BACKTEST["post_score_adjustments"]["macro_reflation_leader_bias_enabled"])
        )

    def _update_effective_range(self) -> None:
        start = self.start_date.date().toPython()
        end = self.end_date.date().toPython()
        days = max(0, (end - start).days + 1)
        self.effective_label.setText(
            f"Requested start: {start}\nRequested end: {end}\nCalendar days: {days}"
        )

    def _update_long_single_weight_hint(self) -> None:
        target_count = max(1, int(self.target_count.value()))
        pct = 100.0 / target_count
        self.long_single_weight_hint.setText(f"~{pct:.2f}% per name ({1.0/target_count:.4f} weight)")

    def _sync_sector_cap_percent_from_names(self) -> None:
        if self._syncing_sector_cap:
            return
        self._syncing_sector_cap = True
        try:
            target_count = max(1, int(self.target_count.value()))
            max_names = max(1, int(self.max_names_per_sector.value()))
            pct = 100.0 * max_names / target_count
            self.sector_cap_percent.setValue(round(pct, 1))
        finally:
            self._syncing_sector_cap = False

    def _sync_sector_names_from_percent(self) -> None:
        if self._syncing_sector_cap:
            return
        self._syncing_sector_cap = True
        try:
            target_count = max(1, int(self.target_count.value()))
            pct = max(0.0, float(self.sector_cap_percent.value()))
            names = max(1, int(round(target_count * pct / 100.0)))
            self.max_names_per_sector.setValue(names)
        finally:
            self._syncing_sector_cap = False

    def _toggle_short_controls(self) -> None:
        enabled = self.enable_short_leg.isChecked()
        self.short_gross_exposure.setEnabled(enabled)
        self.short_target_count.setEnabled(enabled)
        self.short_construction_mode.setEnabled(enabled)
        self.short_weighting_mode.setEnabled(enabled)
        self.short_expansion_mode.setEnabled(enabled)
        self.short_adv_threshold.setEnabled(enabled)
        self.short_sector_cap.setEnabled(enabled)
        self.short_min_names.setEnabled(enabled)
        self.short_max_single_weight.setEnabled(enabled)
        self.short_gross_exposure.setSpecialValueText("Disabled")

    def _toggle_dynamic_weight_controls(self) -> None:
        enabled = False
        self.use_dynamic_weights.setChecked(False)
        self.weight_source_mode.setEnabled(False)
        self._set_combo_by_data(self.weight_source_mode, "manual")
        for row in self.regime_path_rows.values():
            row.setEnabled(enabled)

    def _toggle_pool_controls(self) -> None:
        market_cap_enabled = self.use_market_cap_filter.isChecked()
        for widget in (self.market_cap_mode, self.rank_min, self.rank_max, self.min_market_cap, self.max_market_cap):
            widget.setEnabled(market_cap_enabled)

        liquidity_enabled = self.use_liquidity_filter.isChecked()
        for widget in (self.liquidity_lookback, self.min_volume):
            widget.setEnabled(liquidity_enabled)

        volatility_enabled = self.use_volatility_filter.isChecked()
        for widget in (self.vol_lookback, self.max_vol):
            widget.setEnabled(volatility_enabled)

        lot_cost_enabled = self.use_max_lot_cost_filter.isChecked()
        self.max_lot_cost_jpy.setEnabled(lot_cost_enabled)

        sector_enabled = self.use_sector_constraints.isChecked()
        self.max_names_per_sector.setEnabled(sector_enabled)
        self.sector_cap_mode.setEnabled(sector_enabled)

    def _set_combo_by_data(self, combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _apply_params(self, params: dict) -> None:
        if "start_date" in params:
            self.start_date.setDate(QDate.fromString(str(params["start_date"]), "yyyy-MM-dd"))
        if "end_date" in params:
            self.end_date.setDate(QDate.fromString(str(params["end_date"]), "yyyy-MM-dd"))
        if "frequency" in params:
            self._set_combo_by_data(self.frequency, params["frequency"])
        if "use_market_cap_filter" in params:
            self.use_market_cap_filter.setChecked(bool(params["use_market_cap_filter"]))
        if "market_cap_mode" in params:
            self._set_combo_by_data(self.market_cap_mode, params["market_cap_mode"])
        if "rank_min" in params:
            self.rank_min.setValue(int(params["rank_min"]))
        if "rank_max" in params:
            self.rank_max.setValue(int(params["rank_max"]))
        if "min_market_cap" in params:
            self.min_market_cap.setValue(float(params["min_market_cap"]))
        if "max_market_cap" in params:
            self.max_market_cap.setValue(float(params["max_market_cap"]))

        if "use_liquidity_filter" in params:
            self.use_liquidity_filter.setChecked(bool(params["use_liquidity_filter"]))
        if "liquidity_lookback" in params:
            self.liquidity_lookback.setValue(int(params["liquidity_lookback"]))
        if "min_avg_daily_volume" in params:
            self.min_volume.setValue(int(params["min_avg_daily_volume"]))

        if "use_volatility_filter" in params:
            self.use_volatility_filter.setChecked(bool(params["use_volatility_filter"]))
        if "volatility_lookback" in params:
            self.vol_lookback.setValue(int(params["volatility_lookback"]))
        if "max_annualized_volatility" in params:
            self.max_vol.setValue(float(params["max_annualized_volatility"]))

        if "selection_top_n" in params:
            self.selection_top_n.setValue(int(params["selection_top_n"]))
        if "selection_sort" in params:
            self._set_combo_by_data(self.selection_sort, params["selection_sort"])

        if "use_max_lot_cost_filter" in params:
            self.use_max_lot_cost_filter.setChecked(bool(params["use_max_lot_cost_filter"]))
        if "max_lot_cost_jpy" in params:
            self.max_lot_cost_jpy.setValue(float(params["max_lot_cost_jpy"]))

        if "use_sector_constraints" in params:
            self.use_sector_constraints.setChecked(bool(params["use_sector_constraints"]))
        if "max_names_per_sector" in params:
            self.max_names_per_sector.setValue(int(params["max_names_per_sector"]))
        if "sector_cap_mode" in params:
            self._set_combo_by_data(self.sector_cap_mode, params["sector_cap_mode"])

        enabled_factors = params.get("enabled_factors", {})
        for key, value in enabled_factors.items():
            if key in self.factor_checks:
                self.factor_checks[key].setChecked(bool(value))

        composite_weights = params.get("composite_weights", {})
        for key, value in composite_weights.items():
            if key in self.weight_spins:
                self.weight_spins[key].setValue(float(value))

        if "weight_source_mode" in params:
            self.use_dynamic_weights.setChecked(False)
            self._set_combo_by_data(self.weight_source_mode, "manual")
        regime_weight_map = params.get("regime_weight_map", {})
        for regime_name, path in regime_weight_map.items():
            row = self.regime_path_rows.get(regime_name)
            if row is not None:
                row.line_edit.setText(str(path))  # type: ignore[attr-defined]
        if "allocation_gate_enabled" in params:
            self.allocation_gate_enabled.setChecked(bool(params["allocation_gate_enabled"]))
        if "allocation_gate_mode" in params:
            self._set_combo_by_data(self.allocation_gate_mode, params["allocation_gate_mode"])
        if "allocation_gate_risk_override_threshold" in params:
            self.allocation_gate_risk_override_threshold.setValue(int(params["allocation_gate_risk_override_threshold"]))
        if "allocation_gate_severe_risk_score_threshold" in params:
            self.allocation_gate_severe_risk_score_threshold.setValue(int(params["allocation_gate_severe_risk_score_threshold"]))
        if "allocation_gate_defensive_regime" in params:
            self._set_combo_by_data(self.allocation_gate_defensive_regime, params["allocation_gate_defensive_regime"])
        regime_config = params.get("regime_config", {})
        if "trend_strong_threshold" in regime_config:
            self.trend_strong_threshold.setValue(float(regime_config["trend_strong_threshold"]))
        if "trend_weak_threshold" in regime_config:
            self.trend_weak_threshold.setValue(float(regime_config["trend_weak_threshold"]))
        if "vol_high_enter" in regime_config:
            self.vol_high_enter.setValue(float(regime_config["vol_high_enter"]))
        if "vol_high_exit" in regime_config:
            self.vol_high_exit.setValue(float(regime_config["vol_high_exit"]))
        if "vol_low_enter" in regime_config:
            self.vol_low_enter.setValue(float(regime_config["vol_low_enter"]))
        if "vol_low_exit" in regime_config:
            self.vol_low_exit.setValue(float(regime_config["vol_low_exit"]))
        if "switch_confirm_bars" in regime_config:
            self.switch_confirm_bars.setValue(int(regime_config["switch_confirm_bars"]))
        if "overlay_confirm_bars" in regime_config:
            self.overlay_confirm_bars.setValue(int(regime_config["overlay_confirm_bars"]))
        post_score_adjustments = params.get("post_score_adjustments", {})
        if "regime_sector_bias_regime_source" in post_score_adjustments:
            self._set_combo_by_data(
                self.regime_sector_bias_regime_source,
                post_score_adjustments["regime_sector_bias_regime_source"],
            )
        for regime_name, check in self.regime_sector_bias_checks.items():
            key = f"{regime_name}_sector_bias_enabled"
            if key in post_score_adjustments:
                check.setChecked(bool(post_score_adjustments[key]))
        if "macro_reflation_sector_alignment_enabled" in post_score_adjustments:
            self.macro_reflation_sector_alignment_enabled.setChecked(bool(post_score_adjustments["macro_reflation_sector_alignment_enabled"]))
        if "macro_reflation_size_tilt_enabled" in post_score_adjustments:
            self.macro_reflation_size_tilt_enabled.setChecked(bool(post_score_adjustments["macro_reflation_size_tilt_enabled"]))
        if "macro_reflation_leader_bias_enabled" in post_score_adjustments:
            self.macro_reflation_leader_bias_enabled.setChecked(bool(post_score_adjustments["macro_reflation_leader_bias_enabled"]))

        if "formula" in params:
            self.formula.setText(str(params["formula"]))

        if "initial_capital" in params:
            self.initial_capital.setValue(float(params["initial_capital"]))
        if "target_count" in params:
            self.target_count.setValue(int(params["target_count"]))
        if "max_gross_exposure" in params:
            self.gross_exposure.setValue(float(params["max_gross_exposure"]))

        if "enable_short_leg" in params:
            self.enable_short_leg.setChecked(bool(params["enable_short_leg"]))
        if "short_gross_exposure" in params:
            self.short_gross_exposure.setValue(float(params["short_gross_exposure"]))
        if "short_target_count" in params:
            self.short_target_count.setValue(int(params["short_target_count"]))
        if "short_construction_mode" in params:
            self._set_combo_by_data(self.short_construction_mode, params["short_construction_mode"])
        if "short_weighting_mode" in params:
            self._set_combo_by_data(self.short_weighting_mode, params["short_weighting_mode"])
        if "short_expansion_mode" in params:
            self._set_combo_by_data(self.short_expansion_mode, params["short_expansion_mode"])
        if "short_adv_threshold" in params:
            self.short_adv_threshold.setValue(float(params["short_adv_threshold"] or 0.0))
        if "short_sector_cap" in params:
            self.short_sector_cap.setValue(float(params["short_sector_cap"] or 0.0))
        if "short_min_names" in params:
            self.short_min_names.setValue(int(params["short_min_names"]))
        if "short_max_single_weight" in params:
            self.short_max_single_weight.setValue(float(params["short_max_single_weight"]))

        if "transaction_cost_bps" in params:
            self.transaction_cost.setValue(float(params["transaction_cost_bps"]))
        if "slippage_bps" in params:
            self.slippage.setValue(float(params["slippage_bps"]))
        if "lot_size" in params:
            self.lot_size.setValue(int(params["lot_size"]))
        if "round_lots" in params:
            self.round_lots.setChecked(bool(params["round_lots"]))
        if "use_risk_gating" in params:
            self.use_risk.setChecked(bool(params["use_risk_gating"]))
        if "include_sp500_benchmark" in params:
            self.include_sp500.setChecked(bool(params["include_sp500_benchmark"]))
        if "force_rebuild_cache" in params:
            self.force_rebuild.setChecked(bool(params["force_rebuild_cache"]))

        if "risk_lookback_high" in params:
            self.risk_lookback_high.setValue(int(params["risk_lookback_high"]))
        if "risk_rv_window" in params:
            self.risk_rv_window.setValue(int(params["risk_rv_window"]))
        if "risk_rv_q_window" in params:
            self.risk_rv_q_window.setValue(int(params["risk_rv_q_window"]))
        if "risk_off_score" in params:
            self.risk_off_score.setValue(int(params["risk_off_score"]))
        if "risk_trend_short_ma" in params:
            self.risk_trend_short_ma.setValue(int(params["risk_trend_short_ma"]))
        if "risk_trend_long_ma" in params:
            self.risk_trend_long_ma.setValue(int(params["risk_trend_long_ma"]))
        if "risk_trend_confirm_days" in params:
            self.risk_trend_confirm_days.setValue(int(params["risk_trend_confirm_days"]))
        if "risk_upgrade_confirm_days" in params:
            self.risk_upgrade_confirm_days.setValue(int(params["risk_upgrade_confirm_days"]))
        if "risk_downgrade_confirm_days" in params:
            self.risk_downgrade_confirm_days.setValue(int(params["risk_downgrade_confirm_days"]))

        if "api_key" in params:
            self.api_key.setText(str(params["api_key"]))

        self._update_effective_range()
        self._update_long_single_weight_hint()
        self._toggle_short_controls()
        self._toggle_dynamic_weight_controls()
        self._toggle_pool_controls()

    def _save_config_json(self) -> None:
        params = self._collect_params()
        default_dir = backtest_dir() / "saved_gui_configs"
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save GUI Config",
            os.fspath(default_dir / "backtest_preset.json"),
            "JSON Files (*.json)",
        )
        if not path:
            return
        serializable = dict(params)
        serializable["start_date"] = params["start_date"].toString("yyyy-MM-dd")
        serializable["end_date"] = params["end_date"].toString("yyyy-MM-dd")
        serializable.pop("api_key", None)
        Path(path).write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
        self.statusBar().showMessage(f"Saved config preset: {path}")

    def _load_config_json(self) -> None:
        default_dir = backtest_dir() / "saved_gui_configs"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load GUI Config",
            os.fspath(default_dir),
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            params = json.loads(Path(path).read_text(encoding="utf-8"))
            self._apply_params(params)
        except Exception as exc:
            QMessageBox.critical(self, "Load Config Failed", f"Could not load config JSON.\n\n{exc}")
            return
        self.statusBar().showMessage(f"Loaded config preset: {path}")

    def _collect_params(self) -> dict:
        return {
            "start_date": self.start_date.date(),
            "end_date": self.end_date.date(),
            "frequency": self.frequency.currentData(),
            "use_market_cap_filter": self.use_market_cap_filter.isChecked(),
            "market_cap_mode": self.market_cap_mode.currentData(),
            "rank_min": self.rank_min.value(),
            "rank_max": self.rank_max.value(),
            "min_market_cap": self.min_market_cap.value(),
            "max_market_cap": self.max_market_cap.value(),
            "use_liquidity_filter": self.use_liquidity_filter.isChecked(),
            "liquidity_lookback": self.liquidity_lookback.value(),
            "min_avg_daily_volume": self.min_volume.value(),
            "use_volatility_filter": self.use_volatility_filter.isChecked(),
            "volatility_lookback": self.vol_lookback.value(),
            "max_annualized_volatility": self.max_vol.value(),
            "selection_top_n": self.selection_top_n.value(),
            "selection_sort": self.selection_sort.currentData(),
            "use_max_lot_cost_filter": self.use_max_lot_cost_filter.isChecked(),
            "max_lot_cost_jpy": self.max_lot_cost_jpy.value(),
            "use_sector_constraints": self.use_sector_constraints.isChecked(),
            "max_names_per_sector": self.max_names_per_sector.value(),
            "sector_cap_mode": self.sector_cap_mode.currentData(),
            "enabled_factors": {key: widget.isChecked() for key, widget in self.factor_checks.items()},
            "composite_weights": {key: widget.value() for key, widget in self.weight_spins.items()},
            "allocation_gate_enabled": self.allocation_gate_enabled.isChecked(),
            "allocation_gate_mode": self.allocation_gate_mode.currentData(),
            "allocation_gate_risk_override_threshold": self.allocation_gate_risk_override_threshold.value(),
            "allocation_gate_severe_risk_score_threshold": self.allocation_gate_severe_risk_score_threshold.value(),
            "allocation_gate_defensive_regime": self.allocation_gate_defensive_regime.currentData(),
            "weight_source_mode": "manual",
            "regime_weight_map": {},
            "post_score_adjustments": {
                "regime_sector_bias_regime_source": self.regime_sector_bias_regime_source.currentData(),
                **{
                    f"{regime_name}_sector_bias_enabled": check.isChecked()
                    for regime_name, check in self.regime_sector_bias_checks.items()
                },
                "macro_reflation_sector_alignment_enabled": self.macro_reflation_sector_alignment_enabled.isChecked(),
                "macro_reflation_size_tilt_enabled": self.macro_reflation_size_tilt_enabled.isChecked(),
                "macro_reflation_leader_bias_enabled": self.macro_reflation_leader_bias_enabled.isChecked(),
            },
            "regime_config": {
                "trend_strong_threshold": self.trend_strong_threshold.value(),
                "trend_weak_threshold": self.trend_weak_threshold.value(),
                "vol_high_enter": self.vol_high_enter.value(),
                "vol_high_exit": self.vol_high_exit.value(),
                "vol_low_enter": self.vol_low_enter.value(),
                "vol_low_exit": self.vol_low_exit.value(),
                "switch_confirm_bars": self.switch_confirm_bars.value(),
                "overlay_confirm_bars": self.overlay_confirm_bars.value(),
            },
            "formula": self.formula.text(),
            "initial_capital": self.initial_capital.value(),
            "target_count": self.target_count.value(),
            "max_gross_exposure": self.gross_exposure.value(),
            "enable_short_leg": self.enable_short_leg.isChecked(),
            "short_gross_exposure": self.short_gross_exposure.value(),
            "short_target_count": self.short_target_count.value(),
            "short_construction_mode": self.short_construction_mode.currentData(),
            "short_weighting_mode": self.short_weighting_mode.currentData(),
            "short_expansion_mode": self.short_expansion_mode.currentData(),
            "short_adv_threshold": self.short_adv_threshold.value(),
            "short_sector_cap": self.short_sector_cap.value(),
            "short_min_names": self.short_min_names.value(),
            "short_max_single_weight": self.short_max_single_weight.value(),
            "transaction_cost_bps": self.transaction_cost.value(),
            "slippage_bps": self.slippage.value(),
            "lot_size": self.lot_size.value(),
            "round_lots": self.round_lots.isChecked(),
            "use_risk_gating": self.use_risk.isChecked(),
            "include_sp500_benchmark": self.include_sp500.isChecked(),
            "force_rebuild_cache": self.force_rebuild.isChecked(),
            "risk_lookback_high": self.risk_lookback_high.value(),
            "risk_rv_window": self.risk_rv_window.value(),
            "risk_rv_q_window": self.risk_rv_q_window.value(),
            "risk_off_score": self.risk_off_score.value(),
            "risk_trend_short_ma": self.risk_trend_short_ma.value(),
            "risk_trend_long_ma": self.risk_trend_long_ma.value(),
            "risk_trend_confirm_days": self.risk_trend_confirm_days.value(),
            "risk_upgrade_confirm_days": self.risk_upgrade_confirm_days.value(),
            "risk_downgrade_confirm_days": self.risk_downgrade_confirm_days.value(),
            "api_key": self.api_key.text(),
        }

    def _start_backtest(self) -> None:
        params = self._collect_params()
        validation = validate_parameters(params)
        if not validation.ok:
            QMessageBox.warning(self, t.MSG_VALIDATION_FAILED, "\n".join(validation.errors))
            return
        runtime_dir = backtest_dir() / ".gui_runtime"
        config_path = write_runtime_config(params, runtime_dir)
        credential_lines = []
        if params["api_key"].strip():
            credential_lines.append(f"api_key={params['api_key'].strip()}")
        credential_path = runtime_dir / "gui_api_credentials.txt"
        credential_path.write_text("\n".join(credential_lines) + "\n", encoding="utf-8")
        launch_params = dict(params)
        launch_params["api_file"] = str(credential_path)
        command = build_powershell_command(launch_params, config_path)
        self.log.clear()
        self.performance_table.setRowCount(0)
        self.result_preview_tag.setText("Backtest running... latest result preview will appear here after completion.")
        self.benchmark_note.setText("Strategy is compared against TOPIX and any enabled benchmark series from the finished run.")
        self.nav_chart.removeAllSeries()
        self.nav_chart.setTitle("")
        self.progress.setValue(0)
        self.result_path = ""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.open_button.setEnabled(False)
        self.runner.start(command, backtest_dir())

    def _connect_runner(self) -> None:
        self.runner.log_received.connect(self._append_log)
        self.runner.progress_changed.connect(self.progress.setValue)
        self.runner.status_changed.connect(self.statusBar().showMessage)
        self.runner.result_path_changed.connect(self._set_result_path)
        self.runner.finished_ok.connect(self._on_finished)
        self.runner.failed.connect(self._on_failed)

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log.setTextCursor(cursor)
        for line in text.splitlines():
            escaped = html.escape(line)
            color = self._log_color_for_line(line)
            cursor.insertHtml(
                f'<pre style="margin:0; color:{color}; font-family: Consolas, \'Cascadia Mono\', monospace; white-space: pre-wrap;">{escaped}</pre>'
            )
        if text.endswith("\n"):
            cursor.insertHtml('<pre style="margin:0; color:#7f8fa4; font-family: Consolas, \'Cascadia Mono\', monospace;"> </pre>')
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _log_color_for_line(self, line: str) -> str:
        lower = line.lower()
        if not line.strip():
            return theme.LOG_COLORS["muted"]
        if line.startswith("COMMAND:"):
            return theme.LOG_COLORS["command"]
        if "traceback" in lower or "failed" in lower or "error" in lower or "exception" in lower:
            return theme.LOG_COLORS["error"]
        if "warning" in lower or "miss" in lower or "skip" in lower:
            return theme.LOG_COLORS["warning"]
        if "outputs_saved" in lower or "completed" in lower or "done" in lower or "hit=True" in lower or "cache hit" in lower:
            return theme.LOG_COLORS["success"]
        if "performance summary" in lower or line.startswith("==="):
            return theme.LOG_COLORS["summary"]
        if "[rebalance-debug]" in line or "[composite-debug]" in line or "[regime-debug]" in line or "[allocation-gate]" in line:
            return theme.LOG_COLORS["debug"]
        if "[profile]" in line:
            return "#b5cea8"
        if "[local]" in line or "[cell5]" in line:
            return theme.LOG_COLORS["info"]
        return theme.LOG_COLORS["default"]

    def _set_result_path(self, path: str) -> None:
        self.result_path = path
        self.open_button.setEnabled(bool(path))

    def _on_finished(self, path: str) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.open_button.setEnabled(bool(path))
        self._load_result_preview(path)
        self.tabs.setCurrentIndex(6)
        message = f"{t.MSG_BACKTEST_DONE}\n\nResult path:\n{path or t.MSG_NO_RESULT_PATH}"
        QMessageBox.information(self, t.MSG_BACKTEST_DONE, message)

    def _on_failed(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        QMessageBox.critical(self, t.MSG_BACKTEST_FAILED, message)

    def _open_results(self) -> None:
        if not self.result_path:
            return
        path = Path(self.result_path)
        if not path.is_absolute():
            path = project_root() / path
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.fspath(path)))

    def _open_nav_curve(self, result_path: str) -> None:
        if not result_path:
            return
        root = Path(result_path)
        if not root.is_absolute():
            root = project_root() / root
        for name in ("nav_curve.svg", "nav_curve.png"):
            chart = root / "performance" / name
            if chart.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.fspath(chart)))
                return

    def _load_result_preview(self, result_path: str) -> None:
        if not result_path:
            self.result_preview_tag.setText("Backtest finished, but no result path was detected.")
            return
        root = Path(result_path)
        if not root.is_absolute():
            root = project_root() / root
        perf_csv = root / "performance" / "performance_summary.csv"
        perf_ts_csv = root / "performance" / "performance_timeseries.csv"

        if perf_csv.exists():
            rows: list[dict[str, str]] = []
            with perf_csv.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            headers = ["series", "annualized_return", "max_drawdown", "annualized_volatility", "sharpe_ratio", "sortino_ratio", "excess_return"]

            def _format_perf_value(key: str, value: str) -> str:
                if value is None or value == "":
                    return "-"
                if key == "series":
                    if value == "strategy":
                        return "Strategy"
                    return value
                try:
                    numeric = float(value)
                except ValueError:
                    return value
                if key in {"annualized_return", "annualized_volatility", "max_drawdown", "excess_return"}:
                    return f"{numeric * 100:.2f}%"
                return f"{numeric:.3f}"

            self.performance_table.setRowCount(len(rows))
            for row_idx, row in enumerate(rows):
                for col_idx, key in enumerate(headers):
                    item = QTableWidgetItem(_format_perf_value(key, str(row.get(key, ""))))
                    if key == "series":
                        item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    else:
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if key == "series" and str(row.get(key, "")) == "strategy":
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                    self.performance_table.setItem(row_idx, col_idx, item)
            self._resize_performance_table_to_contents()

            benchmark_labels = [
                _format_perf_value("series", str(row.get("series", "")))
                for row in rows
                if str(row.get("series", "")) != "strategy"
            ]
            if benchmark_labels:
                self.benchmark_note.setText(
                    "Benchmarks in this run: " + ", ".join(benchmark_labels) + ". The chart below compares normalized NAV paths."
                )
            else:
                self.benchmark_note.setText("Only the strategy series was found in the latest run output.")
        else:
            self.performance_table.setRowCount(0)
            self.benchmark_note.setText("performance_summary.csv not found.")

        if perf_ts_csv.exists():
            self._render_nav_chart(perf_ts_csv)
        else:
            self.nav_chart.removeAllSeries()
            self.nav_chart.setTitle("")
        self.result_preview_tag.setText(f"Latest finished run: {root.name}")

    def _render_nav_chart(self, perf_ts_csv: Path) -> None:
        self.nav_chart.removeAllSeries()
        for axis in list(self.nav_chart.axes()):
            self.nav_chart.removeAxis(axis)
        self.nav_chart.setTitle("")
        self.nav_chart.setBackgroundBrush(Qt.transparent)
        self.nav_chart.setPlotAreaBackgroundVisible(True)
        self.nav_chart.setPlotAreaBackgroundBrush(Qt.transparent)
        self.nav_chart.legend().setVisible(True)
        self.nav_chart.legend().setLabelColor(Qt.white)
        self.nav_chart.legend().setAlignment(Qt.AlignTop)
        self.nav_chart.setMargins(QMargins(8, 8, 8, 8))

        rows: dict[str, list[tuple[QDateTime, float]]] = {}
        with perf_ts_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                series_name = str(row.get("series", "") or "")
                date_text = str(row.get("date", "") or "")
                nav_text = str(row.get("nav", "") or "")
                if not series_name or not date_text or not nav_text:
                    continue
                try:
                    nav_value = float(nav_text)
                except ValueError:
                    continue
                dt = QDateTime.fromString(date_text, "yyyy-MM-dd")
                if not dt.isValid():
                    continue
                rows.setdefault(series_name, []).append((dt, nav_value))

        color_map = {
            "strategy": "#32cd32",
            "TOPIX": "#ff4040",
            "SP500": "#2ea8ff",
        }
        min_nav: float | None = None
        max_nav: float | None = None
        min_dt: QDateTime | None = None
        max_dt: QDateTime | None = None

        for series_name, points in rows.items():
            line = QLineSeries()
            display_name = "Strategy" if series_name == "strategy" else series_name
            line.setName(display_name)
            pen = QPen()
            pen.setWidth(2 if series_name == "strategy" else 1)
            pen.setColor(color_map.get(series_name, "#d7e7fa"))
            line.setPen(pen)
            for dt, nav_value in points:
                line.append(float(dt.toMSecsSinceEpoch()), nav_value)
                min_nav = nav_value if min_nav is None else min(min_nav, nav_value)
                max_nav = nav_value if max_nav is None else max(max_nav, nav_value)
                min_dt = dt if min_dt is None or dt < min_dt else min_dt
                max_dt = dt if max_dt is None or dt > max_dt else max_dt
            self.nav_chart.addSeries(line)

        axis_x = QDateTimeAxis()
        axis_x.setFormat("yyyy-MM")
        axis_x.setLabelsColor(Qt.white)
        axis_x.setGridLineColor(Qt.darkGray)
        axis_x.setTickCount(6)

        axis_y = QValueAxis()
        axis_y.setLabelsColor(Qt.white)
        axis_y.setGridLineColor(Qt.darkGray)
        axis_y.setLabelFormat("%.2f")

        if min_dt is not None and max_dt is not None:
            axis_x.setRange(min_dt, max_dt)
        if min_nav is not None and max_nav is not None:
            pad = max((max_nav - min_nav) * 0.08, 0.01)
            axis_y.setRange(min_nav - pad, max_nav + pad)

        self.nav_chart.addAxis(axis_x, Qt.AlignBottom)
        self.nav_chart.addAxis(axis_y, Qt.AlignLeft)
        for series in self.nav_chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)

    def _resize_performance_table_to_contents(self) -> None:
        self.performance_table.resizeRowsToContents()
        header_height = self.performance_table.horizontalHeader().height()
        row_height = sum(self.performance_table.rowHeight(i) for i in range(self.performance_table.rowCount()))
        frame = self.performance_table.frameWidth() * 2
        padding = 8
        target_height = max(70, min(260, header_height + row_height + frame + padding))
        self.performance_table.setMinimumHeight(target_height)
        self.performance_table.setMaximumHeight(target_height)
