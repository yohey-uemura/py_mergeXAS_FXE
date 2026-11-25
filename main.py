#!/usr/bin/env python3
import os
import re
import sys
import traceback
import natsort
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

from PySide6 import QtCore

from silx.gui import qt
from silx.gui.plot import Plot1D
from mw import Ui_MainWindow


def show_warning(parent, text):
    qt.QMessageBox.warning(parent, "Warning", text)


class XASMergeWindow(qt.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Initialize Silx Plot1D widgets inside the UI placeholders
        self.raw_plot = self._embed_plot(self.plot_raw_lower)
        self.hist_plot = self._embed_plot(self.plot_raw_upper)
        self.merged_plot = self._embed_plot(self.plot_merged)

        # Configure scale table cleanly (1 column: Scale, rows match loaded runs)
        self.table_scales.setColumnCount(1)
        self.table_scales.setHorizontalHeaderLabels(["Scale"])
        self.table_scales.setRowCount(0)

        # Initial UI state
        self.on_facility_changed(self.tab_facility.currentIndex())
        self._connect_signals()

    @staticmethod
    def _embed_plot(container):
        plot = Plot1D()
        layout = qt.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(plot)
        return plot

    def _connect_signals(self):
        # Directory & file selection
        self.btn_open_dir.clicked.connect(self.select_directory)
        self.btn_reload_dir.clicked.connect(self.reload_directory)
        self.btn_unselect_all.clicked.connect(self.list_dir_runs.clearSelection)
        self.btn_load_runs.clicked.connect(self.load_selected_runs)

        # Facility / scan type / detector changes
        self.tab_facility.currentChanged.connect(self.on_facility_changed)
        for rb in (self.rb_fxe_ch2a, self.rb_fxe_ch2b, self.rb_sacla_pd, self.rb_sacla_mpccd):
            rb.toggled.connect(self._auto_reload_if_valid)

        # Raw data view & scaling
        self.rb_raw_diff.clicked.connect(self.plot_multi_runs)
        self.rb_raw_on.clicked.connect(self.plot_multi_runs)
        self.rb_raw_off.clicked.connect(self.plot_multi_runs)
        self.combo_raw_channel.currentTextChanged.connect(self.plot_single_run)
        self.table_scales.itemChanged.connect(self.plot_multi_runs)

        # Selected runs & merging
        self.list_selected_runs.itemSelectionChanged.connect(self.on_selected_runs_changed)
        self.btn_merge.clicked.connect(self.merge)
        self.btn_save.clicked.connect(self.save_data)

    # ------------------------------------------------------------------
    # Configuration & Header Helpers
    # ------------------------------------------------------------------
    @property
    def is_fxe(self):
        return self.tab_facility.currentIndex() == 0

    def _header_text(self, col):
        item = self.table_headers.item(0, col)
        return item.text().strip() if item else ""

    def get_scan_config(self):
        """Centralizes column mapping, file extensions, and axis labels for FXE vs SACLA."""
        if self.is_fxe:
            ch = "2_A" if self.rb_fxe_ch2a.isChecked() else "2_B"
            is_energy = self.rb_fxe_energy.isChecked()
            return {
                "ext": f"I0_2_C_If_{ch}",
                "xlabel": self._header_text(0) if is_energy else self._header_text(1),
                "ylabel_on": f"{self._header_text(2)}:{ch}",
                "ylabel_off": f"{self._header_text(3)}:{ch}",
                "ylabel_diff": f"{self._header_text(4)}:{ch}",
                "elabel_on": f"{self._header_text(5)}:{ch}",
                "elabel_off": f"{self._header_text(6)}:{ch}",
                "graph_xlabel": "Energy (eV)" if is_energy else "PPODL (ps)",
                "save_xlabel": "#Energy/eV" if is_energy else "#Motor/pls",
            }
        else:
            ch = "" if self.rb_sacla_pd.isChecked() else "_mpccd"
            is_energy = self.rb_sacla_energy.isChecked()
            scan_prefix = "escan" if is_energy else "mscan"
            return {
                "ext": f"{scan_prefix}{ch}",
                "xlabel": self._header_text(0) if is_energy else self._header_text(1),
                "ylabel_on": self._header_text(2),
                "ylabel_off": self._header_text(3),
                "ylabel_diff": self._header_text(4),
                "elabel_on": self._header_text(5),
                "elabel_off": self._header_text(6),
                "graph_xlabel": "Energy (eV)" if is_energy else "PPODL (ps)",
                "save_xlabel": "#Energy/eV" if is_energy else "#Motor/pls",
            }

    def _run_csv_path(self, run_name, ext):
        base_dir = self.edit_dir_path.text().strip()
        run_dir = base_dir if self.is_fxe else os.path.join(base_dir, run_name)
        return os.path.join(run_dir, f"{run_name}_{ext}.csv")

    # ------------------------------------------------------------------
    # Directory & Tab Handlers
    # ------------------------------------------------------------------
    def on_facility_changed(self, idx):
        self.table_headers.blockSignals(True)
        if idx == 0:  # FXE
            self.table_headers.item(0, 0).setText("#Energy [eV]")
            self.table_headers.item(0, 1).setText("#PPODL [ps]")
            self.table_headers.item(0, 7).setText("__no_entry__")
            self.table_headers.item(0, 8).setText("__no_entry__")
            self.table_headers.item(0, 9).setText("__no_entry__")
            self.chk_use_raw.setChecked(False)
            self.chk_use_raw.setEnabled(False)
            self.spin_lower_limit.setEnabled(False)
            self.spin_upper_limit.setEnabled(False)
        else:  # SACLA
            self.table_headers.item(0, 0).setText("Energy")
            self.table_headers.item(0, 1).setText("#motor")
            self.table_headers.item(0, 7).setText("pd_8")
            self.table_headers.item(0, 8).setText("pd_9")
            self.table_headers.item(0, 9).setText("mpccd")
            self.chk_use_raw.setEnabled(True)
            self.spin_lower_limit.setEnabled(True)
            self.spin_upper_limit.setEnabled(True)
        self.table_headers.blockSignals(False)
        self._auto_reload_if_valid()

    def _auto_reload_if_valid(self):
        if os.path.isdir(self.edit_dir_path.text().strip()):
            self.reload_directory()

    def select_directory(self):
        current = self.edit_dir_path.text().strip()
        start_dir = current if os.path.isdir(current) else os.path.expanduser("~")
        chosen = qt.QFileDialog.getExistingDirectory(self, "Select a directory", start_dir)
        if chosen and os.path.isdir(chosen):
            self.edit_dir_path.setText(chosen)
            self.reload_directory()
        else:
            show_warning(self, "You did not select a valid directory.")

    def reload_directory(self):
        dat_dir = self.edit_dir_path.text().strip()
        if not dat_dir or not os.path.isdir(dat_dir):
            show_warning(self, "Please select a directory first.")
            return

        self.list_dir_runs.clear()
        if self.is_fxe:
            ch = "2_A" if self.rb_fxe_ch2a.isChecked() else "2_B"
            entries = [
                f.split("_")[0]
                for f in os.listdir(dat_dir)
                if f"{ch}.csv" in f
            ]
        else:
            entries = [
                d
                for d in os.listdir(dat_dir)
                if os.path.isdir(os.path.join(dat_dir, d)) and re.match(r"^r\d+", d)
            ]

        if entries:
            self.list_dir_runs.addItems(natsort.natsorted(set(entries)))
        else:
            show_warning(self, "No matching runs found in the selected directory.")

    # ------------------------------------------------------------------
    # Run Loading & Scale Table Management
    # ------------------------------------------------------------------
    def load_selected_runs(self):
        items = [x.text() for x in self.list_dir_runs.selectedItems()]
        if not items:
            return

        self.combo_raw_channel.clear()
        self.combo_raw_channel.addItems(items)

        # Populate scale table dynamically to match loaded runs
        self.table_scales.blockSignals(True)
        self.table_scales.setRowCount(len(items))
        self.table_scales.setVerticalHeaderLabels(items)
        for row in range(len(items)):
            self.table_scales.setItem(row, 0, qt.QTableWidgetItem("1.0"))
        self.table_scales.blockSignals(False)

        # Populate selected runs list and select all by default
        self.list_selected_runs.blockSignals(True)
        self.list_selected_runs.clear()
        self.list_selected_runs.addItems(items)
        for i in range(self.list_selected_runs.count()):
            self.list_selected_runs.item(i).setSelected(True)
        self.list_selected_runs.blockSignals(False)

        self.on_selected_runs_changed()

    def on_selected_runs_changed(self):
        selected = [x.text() for x in self.list_selected_runs.selectedItems()]
        self.combo_merge_method.clear()
        if selected:
            self.combo_merge_method.addItems(selected)
        self.plot_multi_runs()

    def _get_scale_for_run(self, run_name):
        for row in range(self.table_scales.rowCount()):
            header = self.table_scales.verticalHeaderItem(row)
            if header and header.text() == run_name:
                try:
                    return float(self.table_scales.item(row, 0).text())
                except (ValueError, AttributeError):
                    return 1.0
        return 1.0

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_multi_runs(self):
        self.hist_plot.clear()
        cfg = self.get_scan_config()
        selected = [x.text() for x in self.list_selected_runs.selectedItems()]
        if not selected:
            return

        mode = (
            "diff" if self.rb_raw_diff.isChecked()
            else "on" if self.rb_raw_on.isChecked()
            else "off"
        )

        for run in selected:
            csv_path = self._run_csv_path(run, cfg["ext"])
            alpha = self._get_scale_for_run(run)
            try:
                df = pd.read_csv(csv_path, sep=r"\s+")
                x = df[cfg["xlabel"]]
                if mode == "diff":
                    y = (df[cfg["ylabel_on"]] - df[cfg["ylabel_off"]]) * alpha
                    yerr = np.hypot(df[cfg["elabel_on"]] * alpha, df[cfg["elabel_off"]] * alpha)
                elif mode == "on":
                    y = df[cfg["ylabel_on"]] * alpha
                    yerr = np.abs(df[cfg["elabel_on"]] * alpha)
                else:
                    y = df[cfg["ylabel_off"]] * alpha
                    yerr = np.abs(df[cfg["elabel_off"]] * alpha)

                self.hist_plot.addCurve(x, y, yerror=yerr, linewidth=1.5, symbol=".", legend=run)
            except Exception as e:
                print(f"Error plotting {run}: {e}")

        self.hist_plot.setGraphXLabel(cfg["graph_xlabel"])
        self.hist_plot.setGraphYLabel("diff" if mode == "diff" else "XAS")

    def plot_single_run(self, run_name):
        if not run_name:
            return

        self.raw_plot.clear()
        cfg = self.get_scan_config()
        csv_path = self._run_csv_path(run_name, cfg["ext"])

        if not os.path.isfile(csv_path):
            show_warning(
                self,
                f"File <i><font color='red'>{os.path.basename(csv_path)}</font></i> not found. "
                "Check scan type and detector selection.",
            )
            return

        try:
            df = pd.read_csv(csv_path, sep=r"\s+")
            x = df[cfg["xlabel"]]
            y_on, y_off = df[cfg["ylabel_on"]], df[cfg["ylabel_off"]]
            e_on, e_off = df[cfg["elabel_on"]], df[cfg["elabel_off"]]

            self.raw_plot.addCurve(x, y_on, yerror=e_on, color="red", linewidth=1.5, symbol=".", legend="On")
            self.raw_plot.addCurve(x, y_off, yerror=e_off, color="blue", linewidth=1.5, symbol=".", legend="Off")
            self.raw_plot.addCurve(
                x,
                y_on - y_off,
                yerror=np.hypot(e_on, e_off),
                yaxis="right",
                color="green",
                linewidth=1.5,
                symbol=".",
                legend="diff",
            )
            self.raw_plot.setGraphXLabel(cfg["graph_xlabel"])
            self.raw_plot.setGraphYLabel("XAS")
            self.raw_plot.setGraphYLabel(r"$\Delta$XAS", axis="right")
            self.raw_plot.setGraphTitle(run_name)
        except Exception as e:
            show_warning(self, f"Error loading {run_name}:\n{e}")

        self.plot_multi_runs()

    # ------------------------------------------------------------------
    # Merging & Saving
    # ------------------------------------------------------------------
    @staticmethod
    def cc_to_energy(mono_pulse):
        theta = mono_pulse * 4e-6
        return np.round(12.3984 / (6.270832 * np.sin(np.deg2rad(theta))) * 1000, 2)

    def merge(self):
        if self.chk_use_raw.isChecked():
            self.merge_raw_data()
        else:
            self.merge_spectra()

    def merge_raw_data(self):
        try:
            dat_dir = self.edit_dir_path.text().strip()
            items = [x.text() for x in self.list_selected_runs.selectedItems()]
            if not items:
                return

            df_on = pd.concat(
                [pd.read_csv(os.path.join(dat_dir, d, f"laseron_{d}.csv"), index_col="#Tag") for d in items]
            )
            df_off = pd.concat(
                [pd.read_csv(os.path.join(dat_dir, d, f"laseroff_{d}.csv"), index_col="#Tag") for d in items]
            )

            motor_cols = ["mono"] + [k for k in df_on.columns if "motor" in k]
            for col in motor_cols:
                df_on[col] = df_on[col].astype(str).str.replace("pulse", "", regex=False).astype(int)
                df_off[col] = df_off[col].astype(str).str.replace("pulse", "", regex=False).astype(int)

            df_on["Energy"] = self.cc_to_energy(df_on["mono"])
            df_off["Energy"] = self.cc_to_energy(df_off["mono"])

            pd_cols = [k for k in df_on.columns if "pd" in k]
            for col in pd_cols:
                for df in (df_on, df_off):
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace("V", "", regex=False),
                        errors="coerce",
                    )

            cfg = self.get_scan_config()
            if self.rb_sacla_energy.isChecked():
                xlabel = cfg["xlabel"]
                setpoints = np.round(np.unique(df_on["Energy"].values), 2)
            else:
                xlabel = "motor_1"
                setpoints = np.unique(df_on["motor_1"].values)

            delta = 0.1
            thr_low, thr_high = self.spin_lower_limit.value(), self.spin_upper_limit.value()
            l_i0_1, l_i0_2, l_if = self._header_text(7), self._header_text(8), self._header_text(9)

            xas_on, xas_off, err_on, err_off = [], [], [], []
            for e in tqdm(setpoints):
                for j, df in enumerate((df_on, df_off)):
                    subset = df[np.abs(df[xlabel] - e) < delta]
                    valid = (
                        subset[l_i0_1].notna()
                        & subset[l_i0_2].notna()
                        & (subset[l_i0_1] > thr_low)
                        & (subset[l_i0_1] < thr_high)
                        & (subset[l_i0_2] > thr_low)
                        & (subset[l_i0_2] < thr_high)
                    )
                    i0 = 0.5 * (subset.loc[valid, l_i0_1] + subset.loc[valid, l_i0_2])
                    iff = subset.loc[valid, l_if]
                    n_shots = int(valid.sum())

                    val = (iff.sum() / i0.sum()) if n_shots > 0 else np.nan
                    err = (np.std(iff / i0) / np.sqrt(n_shots)) if n_shots > 0 else np.nan
                    if j == 0:
                        xas_on.append(val)
                        err_on.append(err)
                    else:
                        xas_off.append(val)
                        err_off.append(err)

            xas_on, xas_off = np.array(xas_on), np.array(xas_off)
            err_on, err_off = np.array(err_on), np.array(err_off)

            self.merged_plot.clear()
            self.merged_plot.addCurve(setpoints, xas_on, yerror=err_on, color="red", linewidth=1.5, symbol=".", legend="On")
            self.merged_plot.addCurve(setpoints, xas_off, yerror=err_off, color="blue", linewidth=1.5, symbol=".", legend="Off")
            self.merged_plot.addCurve(
                setpoints,
                xas_on - xas_off,
                yerror=np.hypot(err_on, err_off),
                color="green",
                yaxis="right",
                linewidth=1.5,
                symbol=".",
                legend="diff",
            )
            self.merged_plot.setGraphXLabel(cfg["graph_xlabel"])
            self.merged_plot.setGraphYLabel("XAS")
            self.merged_plot.setGraphYLabel(r"$\Delta$XAS", axis="right")
        except Exception:
            show_warning(self, traceback.format_exc())

    def merge_spectra(self):
        ref_run = self.combo_merge_method.currentText()
        selected = [x.text() for x in self.list_selected_runs.selectedItems()]
        if not ref_run or not selected:
            return

        cfg = self.get_scan_config()
        try:
            df_model = pd.read_csv(self._run_csv_path(ref_run, cfg["ext"]), sep=r"\s+")
            x_grid = df_model[cfg["xlabel"]].values

            xas_on_list, xas_off_list, var_on_list, var_off_list = [], [], [], []
            for run in selected:
                df = pd.read_csv(self._run_csv_path(run, cfg["ext"]), sep=r"\s+")
                x_run = df[cfg["xlabel"]].values

                xas_on_list.append(interp1d(x_run, df[cfg["ylabel_on"]].values, bounds_error=False)(x_grid))
                xas_off_list.append(interp1d(x_run, df[cfg["ylabel_off"]].values, bounds_error=False)(x_grid))
                var_on_list.append(interp1d(x_run, df[cfg["elabel_on"]].values ** 2, bounds_error=False)(x_grid))
                var_off_list.append(interp1d(x_run, df[cfg["elabel_off"]].values ** 2, bounds_error=False)(x_grid))

            xas_on_arr = np.array(xas_on_list)
            xas_off_arr = np.array(xas_off_list)
            n_valid = np.sum(~np.isnan(xas_on_arr), axis=0)

            mean_on = np.nanmean(xas_on_arr, axis=0)
            mean_off = np.nanmean(xas_off_arr, axis=0)
            err_on = np.sqrt(np.nanmean(var_on_list, axis=0) / np.maximum(n_valid, 1))
            err_off = np.sqrt(np.nanmean(var_off_list, axis=0) / np.maximum(n_valid, 1))

            self.merged_plot.clear()
            self.merged_plot.addCurve(x_grid, mean_on, yerror=err_on, color="red", linewidth=1.5, symbol=".", legend="On")
            self.merged_plot.addCurve(x_grid, mean_off, yerror=err_off, color="blue", linewidth=1.5, symbol=".", legend="Off")
            self.merged_plot.addCurve(
                x_grid,
                mean_on - mean_off,
                yerror=np.hypot(err_on, err_off),
                color="green",
                yaxis="right",
                linewidth=1.5,
                symbol=".",
                legend="diff",
            )
            self.merged_plot.setGraphXLabel(cfg["graph_xlabel"])
            self.merged_plot.setGraphYLabel("XAS")
            self.merged_plot.setGraphYLabel(r"$\Delta$XAS", axis="right")
        except Exception:
            show_warning(self, traceback.format_exc())

    def save_data(self):
        dat_dir = self.edit_dir_path.text().strip()
        if not os.path.isdir(dat_dir):
            return

        selected = [x.text() for x in self.list_selected_runs.selectedItems()]
        default_name = os.path.join(dat_dir, f"avg_{'_'.join(selected)}.csv")
        out_path, _ = qt.QFileDialog.getSaveFileName(self, "Set the output file name", default_name)
        if not out_path:
            return

        try:
            x, y_on, _, yerr_on = self.merged_plot.getCurve("On").getData()
            _, y_off, _, yerr_off = self.merged_plot.getCurve("Off").getData()
            _, y_diff, _, yerr_diff = self.merged_plot.getCurve("diff").getData()

            cfg = self.get_scan_config()
            pd.DataFrame(
                {
                    cfg["save_xlabel"]: x,
                    "On": y_on,
                    "Off": y_off,
                    "diff": y_diff,
                    "err_on": yerr_on,
                    "err_off": yerr_off,
                    "err_diff": yerr_diff,
                }
            ).to_csv(out_path, index=False, sep=" ", float_format="%.6f")
        except Exception:
            show_warning(self, traceback.format_exc())


if __name__ == "__main__":
    app = qt.QApplication(sys.argv)
    window = XASMergeWindow()
    window.setWindowTitle("Merge XAS Data")
    window.show()
    sys.exit(app.exec_())
