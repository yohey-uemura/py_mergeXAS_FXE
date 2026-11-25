# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mw.ui'
##
## Created by: Qt User Interface Compiler version 6.12.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox,
    QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMenuBar, QPushButton, QRadioButton, QSizePolicy,
    QSpacerItem, QSplitter, QStatusBar, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1600, 960)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.layout_central = QVBoxLayout(self.centralwidget)
        self.layout_central.setObjectName(u"layout_central")
        self.splitter_main = QSplitter(self.centralwidget)
        self.splitter_main.setObjectName(u"splitter_main")
        self.splitter_main.setOrientation(Qt.Horizontal)
        self.splitter_main.setChildrenCollapsible(False)
        self.panel_left = QFrame(self.splitter_main)
        self.panel_left.setObjectName(u"panel_left")
        self.panel_left.setFrameShape(QFrame.StyledPanel)
        self.panel_left.setFrameShadow(QFrame.Sunken)
        self.layout_left = QVBoxLayout(self.panel_left)
        self.layout_left.setObjectName(u"layout_left")
        self.layout_dir_bar = QHBoxLayout()
        self.layout_dir_bar.setObjectName(u"layout_dir_bar")
        self.btn_open_dir = QPushButton(self.panel_left)
        self.btn_open_dir.setObjectName(u"btn_open_dir")

        self.layout_dir_bar.addWidget(self.btn_open_dir)

        self.edit_dir_path = QLineEdit(self.panel_left)
        self.edit_dir_path.setObjectName(u"edit_dir_path")
        self.edit_dir_path.setReadOnly(True)

        self.layout_dir_bar.addWidget(self.edit_dir_path)


        self.layout_left.addLayout(self.layout_dir_bar)

        self.tab_facility = QTabWidget(self.panel_left)
        self.tab_facility.setObjectName(u"tab_facility")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab_facility.sizePolicy().hasHeightForWidth())
        self.tab_facility.setSizePolicy(sizePolicy)
        self.tab_fxe = QWidget()
        self.tab_fxe.setObjectName(u"tab_fxe")
        self.layout_tab_fxe = QHBoxLayout(self.tab_fxe)
        self.layout_tab_fxe.setObjectName(u"layout_tab_fxe")
        self.group_fxe_scan_type = QGroupBox(self.tab_fxe)
        self.group_fxe_scan_type.setObjectName(u"group_fxe_scan_type")
        self.layout_fxe_scan = QHBoxLayout(self.group_fxe_scan_type)
        self.layout_fxe_scan.setObjectName(u"layout_fxe_scan")
        self.rb_fxe_energy = QRadioButton(self.group_fxe_scan_type)
        self.bg_fxe_scan_type = QButtonGroup(MainWindow)
        self.bg_fxe_scan_type.setObjectName(u"bg_fxe_scan_type")
        self.bg_fxe_scan_type.addButton(self.rb_fxe_energy)
        self.rb_fxe_energy.setObjectName(u"rb_fxe_energy")
        self.rb_fxe_energy.setChecked(True)

        self.layout_fxe_scan.addWidget(self.rb_fxe_energy)

        self.rb_fxe_ppodl = QRadioButton(self.group_fxe_scan_type)
        self.bg_fxe_scan_type.addButton(self.rb_fxe_ppodl)
        self.rb_fxe_ppodl.setObjectName(u"rb_fxe_ppodl")

        self.layout_fxe_scan.addWidget(self.rb_fxe_ppodl)


        self.layout_tab_fxe.addWidget(self.group_fxe_scan_type)

        self.group_fxe_detector = QGroupBox(self.tab_fxe)
        self.group_fxe_detector.setObjectName(u"group_fxe_detector")
        self.layout_fxe_det = QHBoxLayout(self.group_fxe_detector)
        self.layout_fxe_det.setObjectName(u"layout_fxe_det")
        self.rb_fxe_ch2a = QRadioButton(self.group_fxe_detector)
        self.bg_fxe_detector = QButtonGroup(MainWindow)
        self.bg_fxe_detector.setObjectName(u"bg_fxe_detector")
        self.bg_fxe_detector.addButton(self.rb_fxe_ch2a)
        self.rb_fxe_ch2a.setObjectName(u"rb_fxe_ch2a")
        self.rb_fxe_ch2a.setChecked(True)

        self.layout_fxe_det.addWidget(self.rb_fxe_ch2a)

        self.rb_fxe_ch2b = QRadioButton(self.group_fxe_detector)
        self.bg_fxe_detector.addButton(self.rb_fxe_ch2b)
        self.rb_fxe_ch2b.setObjectName(u"rb_fxe_ch2b")

        self.layout_fxe_det.addWidget(self.rb_fxe_ch2b)


        self.layout_tab_fxe.addWidget(self.group_fxe_detector)

        self.tab_facility.addTab(self.tab_fxe, "")
        self.tab_sacla = QWidget()
        self.tab_sacla.setObjectName(u"tab_sacla")
        self.layout_tab_sacla = QHBoxLayout(self.tab_sacla)
        self.layout_tab_sacla.setObjectName(u"layout_tab_sacla")
        self.group_sacla_scan_type = QGroupBox(self.tab_sacla)
        self.group_sacla_scan_type.setObjectName(u"group_sacla_scan_type")
        self.layout_sacla_scan = QHBoxLayout(self.group_sacla_scan_type)
        self.layout_sacla_scan.setObjectName(u"layout_sacla_scan")
        self.rb_sacla_energy = QRadioButton(self.group_sacla_scan_type)
        self.bg_sacla_scan_type = QButtonGroup(MainWindow)
        self.bg_sacla_scan_type.setObjectName(u"bg_sacla_scan_type")
        self.bg_sacla_scan_type.addButton(self.rb_sacla_energy)
        self.rb_sacla_energy.setObjectName(u"rb_sacla_energy")
        self.rb_sacla_energy.setChecked(True)

        self.layout_sacla_scan.addWidget(self.rb_sacla_energy)

        self.rb_sacla_delay = QRadioButton(self.group_sacla_scan_type)
        self.bg_sacla_scan_type.addButton(self.rb_sacla_delay)
        self.rb_sacla_delay.setObjectName(u"rb_sacla_delay")

        self.layout_sacla_scan.addWidget(self.rb_sacla_delay)


        self.layout_tab_sacla.addWidget(self.group_sacla_scan_type)

        self.group_sacla_detector = QGroupBox(self.tab_sacla)
        self.group_sacla_detector.setObjectName(u"group_sacla_detector")
        self.layout_sacla_det = QHBoxLayout(self.group_sacla_detector)
        self.layout_sacla_det.setObjectName(u"layout_sacla_det")
        self.rb_sacla_pd = QRadioButton(self.group_sacla_detector)
        self.bg_sacla_detector = QButtonGroup(MainWindow)
        self.bg_sacla_detector.setObjectName(u"bg_sacla_detector")
        self.bg_sacla_detector.addButton(self.rb_sacla_pd)
        self.rb_sacla_pd.setObjectName(u"rb_sacla_pd")
        self.rb_sacla_pd.setChecked(True)

        self.layout_sacla_det.addWidget(self.rb_sacla_pd)

        self.rb_sacla_mpccd = QRadioButton(self.group_sacla_detector)
        self.bg_sacla_detector.addButton(self.rb_sacla_mpccd)
        self.rb_sacla_mpccd.setObjectName(u"rb_sacla_mpccd")

        self.layout_sacla_det.addWidget(self.rb_sacla_mpccd)


        self.layout_tab_sacla.addWidget(self.group_sacla_detector)

        self.tab_facility.addTab(self.tab_sacla, "")

        self.layout_left.addWidget(self.tab_facility)

        self.layout_dir_controls = QHBoxLayout()
        self.layout_dir_controls.setObjectName(u"layout_dir_controls")
        self.lbl_dir_data = QLabel(self.panel_left)
        self.lbl_dir_data.setObjectName(u"lbl_dir_data")

        self.layout_dir_controls.addWidget(self.lbl_dir_data)

        self.spacer_dir_controls = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.layout_dir_controls.addItem(self.spacer_dir_controls)

        self.btn_unselect_all = QPushButton(self.panel_left)
        self.btn_unselect_all.setObjectName(u"btn_unselect_all")

        self.layout_dir_controls.addWidget(self.btn_unselect_all)

        self.btn_reload_dir = QPushButton(self.panel_left)
        self.btn_reload_dir.setObjectName(u"btn_reload_dir")
        icon = QIcon()
        icon.addFile(u"reload.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.btn_reload_dir.setIcon(icon)
        self.btn_reload_dir.setIconSize(QSize(20, 20))

        self.layout_dir_controls.addWidget(self.btn_reload_dir)


        self.layout_left.addLayout(self.layout_dir_controls)

        self.list_dir_runs = QListWidget(self.panel_left)
        self.list_dir_runs.setObjectName(u"list_dir_runs")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(3)
        sizePolicy1.setHeightForWidth(self.list_dir_runs.sizePolicy().hasHeightForWidth())
        self.list_dir_runs.setSizePolicy(sizePolicy1)
        self.list_dir_runs.setSelectionMode(QAbstractItemView.MultiSelection)

        self.layout_left.addWidget(self.list_dir_runs)

        self.lbl_headers = QLabel(self.panel_left)
        self.lbl_headers.setObjectName(u"lbl_headers")

        self.layout_left.addWidget(self.lbl_headers)

        self.table_headers = QTableWidget(self.panel_left)
        if (self.table_headers.columnCount() < 15):
            self.table_headers.setColumnCount(15)
        __qtablewidgetitem = QTableWidgetItem()
        self.table_headers.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.table_headers.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.table_headers.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.table_headers.setHorizontalHeaderItem(3, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.table_headers.setHorizontalHeaderItem(4, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.table_headers.setHorizontalHeaderItem(5, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.table_headers.setHorizontalHeaderItem(6, __qtablewidgetitem6)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.table_headers.setHorizontalHeaderItem(7, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        self.table_headers.setHorizontalHeaderItem(8, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        self.table_headers.setHorizontalHeaderItem(9, __qtablewidgetitem9)
        if (self.table_headers.rowCount() < 1):
            self.table_headers.setRowCount(1)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.table_headers.setItem(0, 0, __qtablewidgetitem10)
        __qtablewidgetitem11 = QTableWidgetItem()
        self.table_headers.setItem(0, 1, __qtablewidgetitem11)
        __qtablewidgetitem12 = QTableWidgetItem()
        self.table_headers.setItem(0, 2, __qtablewidgetitem12)
        __qtablewidgetitem13 = QTableWidgetItem()
        self.table_headers.setItem(0, 3, __qtablewidgetitem13)
        __qtablewidgetitem14 = QTableWidgetItem()
        self.table_headers.setItem(0, 4, __qtablewidgetitem14)
        __qtablewidgetitem15 = QTableWidgetItem()
        self.table_headers.setItem(0, 5, __qtablewidgetitem15)
        __qtablewidgetitem16 = QTableWidgetItem()
        self.table_headers.setItem(0, 6, __qtablewidgetitem16)
        __qtablewidgetitem17 = QTableWidgetItem()
        self.table_headers.setItem(0, 7, __qtablewidgetitem17)
        __qtablewidgetitem18 = QTableWidgetItem()
        self.table_headers.setItem(0, 8, __qtablewidgetitem18)
        __qtablewidgetitem19 = QTableWidgetItem()
        self.table_headers.setItem(0, 9, __qtablewidgetitem19)
        self.table_headers.setObjectName(u"table_headers")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.table_headers.sizePolicy().hasHeightForWidth())
        self.table_headers.setSizePolicy(sizePolicy2)
        self.table_headers.setMaximumSize(QSize(16777215, 75))
        self.table_headers.setRowCount(1)
        self.table_headers.setColumnCount(15)

        self.layout_left.addWidget(self.table_headers)

        self.layout_load_bar = QHBoxLayout()
        self.layout_load_bar.setObjectName(u"layout_load_bar")
        self.spacer_load_bar = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.layout_load_bar.addItem(self.spacer_load_bar)

        self.btn_load_runs = QPushButton(self.panel_left)
        self.btn_load_runs.setObjectName(u"btn_load_runs")

        self.layout_load_bar.addWidget(self.btn_load_runs)


        self.layout_left.addLayout(self.layout_load_bar)

        self.splitter_main.addWidget(self.panel_left)
        self.splitter_right = QSplitter(self.splitter_main)
        self.splitter_right.setObjectName(u"splitter_right")
        self.splitter_right.setOrientation(Qt.Horizontal)
        self.splitter_right.setChildrenCollapsible(False)
        self.panel_raw_data = QFrame(self.splitter_right)
        self.panel_raw_data.setObjectName(u"panel_raw_data")
        self.panel_raw_data.setFrameShape(QFrame.StyledPanel)
        self.panel_raw_data.setFrameShadow(QFrame.Sunken)
        self.layout_raw_data = QVBoxLayout(self.panel_raw_data)
        self.layout_raw_data.setObjectName(u"layout_raw_data")
        self.group_raw_mode = QGroupBox(self.panel_raw_data)
        self.group_raw_mode.setObjectName(u"group_raw_mode")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.group_raw_mode.sizePolicy().hasHeightForWidth())
        self.group_raw_mode.setSizePolicy(sizePolicy3)
        self.layout_raw_mode = QHBoxLayout(self.group_raw_mode)
        self.layout_raw_mode.setObjectName(u"layout_raw_mode")
        self.rb_raw_diff = QRadioButton(self.group_raw_mode)
        self.bg_raw_mode = QButtonGroup(MainWindow)
        self.bg_raw_mode.setObjectName(u"bg_raw_mode")
        self.bg_raw_mode.addButton(self.rb_raw_diff)
        self.rb_raw_diff.setObjectName(u"rb_raw_diff")
        self.rb_raw_diff.setChecked(True)

        self.layout_raw_mode.addWidget(self.rb_raw_diff)

        self.rb_raw_on = QRadioButton(self.group_raw_mode)
        self.bg_raw_mode.addButton(self.rb_raw_on)
        self.rb_raw_on.setObjectName(u"rb_raw_on")

        self.layout_raw_mode.addWidget(self.rb_raw_on)

        self.rb_raw_off = QRadioButton(self.group_raw_mode)
        self.bg_raw_mode.addButton(self.rb_raw_off)
        self.rb_raw_off.setObjectName(u"rb_raw_off")

        self.layout_raw_mode.addWidget(self.rb_raw_off)


        self.layout_raw_data.addWidget(self.group_raw_mode)

        self.splitter_raw_plots = QSplitter(self.panel_raw_data)
        self.splitter_raw_plots.setObjectName(u"splitter_raw_plots")
        self.splitter_raw_plots.setOrientation(Qt.Vertical)
        self.splitter_raw_plots.setChildrenCollapsible(False)
        self.container_raw_top = QWidget(self.splitter_raw_plots)
        self.container_raw_top.setObjectName(u"container_raw_top")
        self.layout_raw_top = QHBoxLayout(self.container_raw_top)
        self.layout_raw_top.setObjectName(u"layout_raw_top")
        self.layout_raw_top.setContentsMargins(0, 0, 0, 0)
        self.plot_raw_upper = QWidget(self.container_raw_top)
        self.plot_raw_upper.setObjectName(u"plot_raw_upper")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy4.setHorizontalStretch(3)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.plot_raw_upper.sizePolicy().hasHeightForWidth())
        self.plot_raw_upper.setSizePolicy(sizePolicy4)

        self.layout_raw_top.addWidget(self.plot_raw_upper)

        self.table_scales = QTableWidget(self.container_raw_top)
        if (self.table_scales.columnCount() < 2):
            self.table_scales.setColumnCount(2)
        __qtablewidgetitem20 = QTableWidgetItem()
        self.table_scales.setHorizontalHeaderItem(0, __qtablewidgetitem20)
        if (self.table_scales.rowCount() < 10):
            self.table_scales.setRowCount(10)
        __qtablewidgetitem21 = QTableWidgetItem()
        self.table_scales.setItem(0, 0, __qtablewidgetitem21)
        __qtablewidgetitem22 = QTableWidgetItem()
        self.table_scales.setItem(1, 0, __qtablewidgetitem22)
        __qtablewidgetitem23 = QTableWidgetItem()
        self.table_scales.setItem(2, 0, __qtablewidgetitem23)
        __qtablewidgetitem24 = QTableWidgetItem()
        self.table_scales.setItem(3, 0, __qtablewidgetitem24)
        __qtablewidgetitem25 = QTableWidgetItem()
        self.table_scales.setItem(4, 0, __qtablewidgetitem25)
        __qtablewidgetitem26 = QTableWidgetItem()
        self.table_scales.setItem(5, 0, __qtablewidgetitem26)
        __qtablewidgetitem27 = QTableWidgetItem()
        self.table_scales.setItem(6, 0, __qtablewidgetitem27)
        __qtablewidgetitem28 = QTableWidgetItem()
        self.table_scales.setItem(7, 0, __qtablewidgetitem28)
        __qtablewidgetitem29 = QTableWidgetItem()
        self.table_scales.setItem(8, 0, __qtablewidgetitem29)
        __qtablewidgetitem30 = QTableWidgetItem()
        self.table_scales.setItem(9, 0, __qtablewidgetitem30)
        self.table_scales.setObjectName(u"table_scales")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy5.setHorizontalStretch(1)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.table_scales.sizePolicy().hasHeightForWidth())
        self.table_scales.setSizePolicy(sizePolicy5)
        self.table_scales.setMaximumSize(QSize(180, 16777215))
        self.table_scales.setRowCount(10)
        self.table_scales.setColumnCount(2)

        self.layout_raw_top.addWidget(self.table_scales)

        self.splitter_raw_plots.addWidget(self.container_raw_top)
        self.plot_raw_lower = QWidget(self.splitter_raw_plots)
        self.plot_raw_lower.setObjectName(u"plot_raw_lower")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.plot_raw_lower.sizePolicy().hasHeightForWidth())
        self.plot_raw_lower.setSizePolicy(sizePolicy6)
        self.splitter_raw_plots.addWidget(self.plot_raw_lower)

        self.layout_raw_data.addWidget(self.splitter_raw_plots)

        self.combo_raw_channel = QComboBox(self.panel_raw_data)
        self.combo_raw_channel.setObjectName(u"combo_raw_channel")

        self.layout_raw_data.addWidget(self.combo_raw_channel)

        self.splitter_right.addWidget(self.panel_raw_data)
        self.panel_merged_data = QFrame(self.splitter_right)
        self.panel_merged_data.setObjectName(u"panel_merged_data")
        self.panel_merged_data.setFrameShape(QFrame.StyledPanel)
        self.panel_merged_data.setFrameShadow(QFrame.Sunken)
        self.layout_merged_data = QVBoxLayout(self.panel_merged_data)
        self.layout_merged_data.setObjectName(u"layout_merged_data")
        self.group_selected_data = QGroupBox(self.panel_merged_data)
        self.group_selected_data.setObjectName(u"group_selected_data")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(2)
        sizePolicy7.setHeightForWidth(self.group_selected_data.sizePolicy().hasHeightForWidth())
        self.group_selected_data.setSizePolicy(sizePolicy7)
        self.layout_selected_data = QVBoxLayout(self.group_selected_data)
        self.layout_selected_data.setObjectName(u"layout_selected_data")
        self.list_selected_runs = QListWidget(self.group_selected_data)
        self.list_selected_runs.setObjectName(u"list_selected_runs")
        self.list_selected_runs.setSelectionMode(QAbstractItemView.MultiSelection)

        self.layout_selected_data.addWidget(self.list_selected_runs)

        self.layout_merge_controls = QHBoxLayout()
        self.layout_merge_controls.setObjectName(u"layout_merge_controls")
        self.combo_merge_method = QComboBox(self.group_selected_data)
        self.combo_merge_method.setObjectName(u"combo_merge_method")
        sizePolicy2.setHeightForWidth(self.combo_merge_method.sizePolicy().hasHeightForWidth())
        self.combo_merge_method.setSizePolicy(sizePolicy2)

        self.layout_merge_controls.addWidget(self.combo_merge_method)

        self.btn_merge = QPushButton(self.group_selected_data)
        self.btn_merge.setObjectName(u"btn_merge")

        self.layout_merge_controls.addWidget(self.btn_merge)


        self.layout_selected_data.addLayout(self.layout_merge_controls)

        self.layout_thresholds = QGridLayout()
        self.layout_thresholds.setObjectName(u"layout_thresholds")
        self.lbl_lower_limit = QLabel(self.group_selected_data)
        self.lbl_lower_limit.setObjectName(u"lbl_lower_limit")

        self.layout_thresholds.addWidget(self.lbl_lower_limit, 0, 1, 1, 1)

        self.lbl_upper_limit = QLabel(self.group_selected_data)
        self.lbl_upper_limit.setObjectName(u"lbl_upper_limit")

        self.layout_thresholds.addWidget(self.lbl_upper_limit, 0, 2, 1, 1)

        self.chk_use_raw = QCheckBox(self.group_selected_data)
        self.chk_use_raw.setObjectName(u"chk_use_raw")

        self.layout_thresholds.addWidget(self.chk_use_raw, 1, 0, 1, 1)

        self.spin_lower_limit = QDoubleSpinBox(self.group_selected_data)
        self.spin_lower_limit.setObjectName(u"spin_lower_limit")
        self.spin_lower_limit.setDecimals(4)
        self.spin_lower_limit.setMaximum(1.000000000000000)
        self.spin_lower_limit.setSingleStep(0.001000000000000)
        self.spin_lower_limit.setValue(0.001000000000000)

        self.layout_thresholds.addWidget(self.spin_lower_limit, 1, 1, 1, 1)

        self.spin_upper_limit = QDoubleSpinBox(self.group_selected_data)
        self.spin_upper_limit.setObjectName(u"spin_upper_limit")
        self.spin_upper_limit.setDecimals(4)
        self.spin_upper_limit.setMaximum(1.000000000000000)
        self.spin_upper_limit.setSingleStep(0.001000000000000)
        self.spin_upper_limit.setValue(0.999000000000000)

        self.layout_thresholds.addWidget(self.spin_upper_limit, 1, 2, 1, 1)


        self.layout_selected_data.addLayout(self.layout_thresholds)


        self.layout_merged_data.addWidget(self.group_selected_data)

        self.plot_merged = QWidget(self.panel_merged_data)
        self.plot_merged.setObjectName(u"plot_merged")
        sizePolicy1.setHeightForWidth(self.plot_merged.sizePolicy().hasHeightForWidth())
        self.plot_merged.setSizePolicy(sizePolicy1)

        self.layout_merged_data.addWidget(self.plot_merged)

        self.layout_save_bar = QHBoxLayout()
        self.layout_save_bar.setObjectName(u"layout_save_bar")
        self.spacer_save_bar = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.layout_save_bar.addItem(self.spacer_save_bar)

        self.btn_save = QPushButton(self.panel_merged_data)
        self.btn_save.setObjectName(u"btn_save")

        self.layout_save_bar.addWidget(self.btn_save)


        self.layout_merged_data.addLayout(self.layout_save_bar)

        self.splitter_right.addWidget(self.panel_merged_data)
        self.splitter_main.addWidget(self.splitter_right)

        self.layout_central.addWidget(self.splitter_main)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1600, 26))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        QWidget.setTabOrder(self.btn_open_dir, self.edit_dir_path)
        QWidget.setTabOrder(self.edit_dir_path, self.tab_facility)
        QWidget.setTabOrder(self.tab_facility, self.rb_fxe_energy)
        QWidget.setTabOrder(self.rb_fxe_energy, self.rb_fxe_ppodl)
        QWidget.setTabOrder(self.rb_fxe_ppodl, self.rb_fxe_ch2a)
        QWidget.setTabOrder(self.rb_fxe_ch2a, self.rb_fxe_ch2b)
        QWidget.setTabOrder(self.rb_fxe_ch2b, self.btn_unselect_all)
        QWidget.setTabOrder(self.btn_unselect_all, self.btn_reload_dir)
        QWidget.setTabOrder(self.btn_reload_dir, self.list_dir_runs)
        QWidget.setTabOrder(self.list_dir_runs, self.table_headers)
        QWidget.setTabOrder(self.table_headers, self.btn_load_runs)
        QWidget.setTabOrder(self.btn_load_runs, self.rb_raw_diff)
        QWidget.setTabOrder(self.rb_raw_diff, self.rb_raw_on)
        QWidget.setTabOrder(self.rb_raw_on, self.rb_raw_off)
        QWidget.setTabOrder(self.rb_raw_off, self.combo_raw_channel)
        QWidget.setTabOrder(self.combo_raw_channel, self.list_selected_runs)
        QWidget.setTabOrder(self.list_selected_runs, self.combo_merge_method)
        QWidget.setTabOrder(self.combo_merge_method, self.btn_merge)
        QWidget.setTabOrder(self.btn_merge, self.chk_use_raw)
        QWidget.setTabOrder(self.chk_use_raw, self.spin_lower_limit)
        QWidget.setTabOrder(self.spin_lower_limit, self.spin_upper_limit)
        QWidget.setTabOrder(self.spin_upper_limit, self.btn_save)

        self.retranslateUi(MainWindow)

        self.tab_facility.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Merge XAS Data", None))
        self.btn_open_dir.setText(QCoreApplication.translate("MainWindow", u"Open", None))
        self.edit_dir_path.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Select data directory...", None))
        self.group_fxe_scan_type.setTitle(QCoreApplication.translate("MainWindow", u"Scan type", None))
        self.rb_fxe_energy.setText(QCoreApplication.translate("MainWindow", u"Energy", None))
        self.rb_fxe_ppodl.setText(QCoreApplication.translate("MainWindow", u"PPODL", None))
        self.group_fxe_detector.setTitle(QCoreApplication.translate("MainWindow", u"Detector", None))
        self.rb_fxe_ch2a.setText(QCoreApplication.translate("MainWindow", u"2_A", None))
        self.rb_fxe_ch2b.setText(QCoreApplication.translate("MainWindow", u"2_B", None))
        self.tab_facility.setTabText(self.tab_facility.indexOf(self.tab_fxe), QCoreApplication.translate("MainWindow", u"FXE", None))
        self.group_sacla_scan_type.setTitle(QCoreApplication.translate("MainWindow", u"Scan type", None))
        self.rb_sacla_energy.setText(QCoreApplication.translate("MainWindow", u"Energy", None))
        self.rb_sacla_delay.setText(QCoreApplication.translate("MainWindow", u"Delay", None))
        self.group_sacla_detector.setTitle(QCoreApplication.translate("MainWindow", u"Detector", None))
        self.rb_sacla_pd.setText(QCoreApplication.translate("MainWindow", u"PD", None))
        self.rb_sacla_mpccd.setText(QCoreApplication.translate("MainWindow", u"mpccd", None))
        self.tab_facility.setTabText(self.tab_facility.indexOf(self.tab_sacla), QCoreApplication.translate("MainWindow", u"SACLA", None))
        self.lbl_dir_data.setText(QCoreApplication.translate("MainWindow", u"Data in the directory", None))
        self.btn_unselect_all.setText(QCoreApplication.translate("MainWindow", u"Unselect all", None))
#if QT_CONFIG(tooltip)
        self.btn_reload_dir.setToolTip(QCoreApplication.translate("MainWindow", u"Reload directory", None))
#endif // QT_CONFIG(tooltip)
        self.btn_reload_dir.setText("")
        self.lbl_headers.setText(QCoreApplication.translate("MainWindow", u"Headers", None))
        ___qtablewidgetitem = self.table_headers.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"Energy", None))
        ___qtablewidgetitem1 = self.table_headers.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Delay", None))
        ___qtablewidgetitem2 = self.table_headers.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"xas_on", None))
        ___qtablewidgetitem3 = self.table_headers.horizontalHeaderItem(3)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"xas_off", None))
        ___qtablewidgetitem4 = self.table_headers.horizontalHeaderItem(4)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("MainWindow", u"diff", None))
        ___qtablewidgetitem5 = self.table_headers.horizontalHeaderItem(5)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("MainWindow", u"err_on", None))
        ___qtablewidgetitem6 = self.table_headers.horizontalHeaderItem(6)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("MainWindow", u"err_off", None))
        ___qtablewidgetitem7 = self.table_headers.horizontalHeaderItem(7)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("MainWindow", u"I0_1", None))
        ___qtablewidgetitem8 = self.table_headers.horizontalHeaderItem(8)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("MainWindow", u"I0_2", None))
        ___qtablewidgetitem9 = self.table_headers.horizontalHeaderItem(9)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("MainWindow", u"If", None))

        __sortingEnabled = self.table_headers.isSortingEnabled()
        self.table_headers.setSortingEnabled(False)
        ___qtablewidgetitem10 = self.table_headers.item(0, 0)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("MainWindow", u"#Energy [eV]", None))
        ___qtablewidgetitem11 = self.table_headers.item(0, 1)
        ___qtablewidgetitem11.setText(QCoreApplication.translate("MainWindow", u"#PPODL [ps]", None))
        ___qtablewidgetitem12 = self.table_headers.item(0, 2)
        ___qtablewidgetitem12.setText(QCoreApplication.translate("MainWindow", u"xas_on", None))
        ___qtablewidgetitem13 = self.table_headers.item(0, 3)
        ___qtablewidgetitem13.setText(QCoreApplication.translate("MainWindow", u"xas_off", None))
        ___qtablewidgetitem14 = self.table_headers.item(0, 4)
        ___qtablewidgetitem14.setText(QCoreApplication.translate("MainWindow", u"diff", None))
        ___qtablewidgetitem15 = self.table_headers.item(0, 5)
        ___qtablewidgetitem15.setText(QCoreApplication.translate("MainWindow", u"err_on", None))
        ___qtablewidgetitem16 = self.table_headers.item(0, 6)
        ___qtablewidgetitem16.setText(QCoreApplication.translate("MainWindow", u"err_off", None))
        ___qtablewidgetitem17 = self.table_headers.item(0, 7)
        ___qtablewidgetitem17.setText(QCoreApplication.translate("MainWindow", u"__no_entry__", None))
        ___qtablewidgetitem18 = self.table_headers.item(0, 8)
        ___qtablewidgetitem18.setText(QCoreApplication.translate("MainWindow", u"__no_entry__", None))
        ___qtablewidgetitem19 = self.table_headers.item(0, 9)
        ___qtablewidgetitem19.setText(QCoreApplication.translate("MainWindow", u"__no_entry__", None))
        self.table_headers.setSortingEnabled(__sortingEnabled)

        self.btn_load_runs.setText(QCoreApplication.translate("MainWindow", u"Load runs", None))
        self.group_raw_mode.setTitle(QCoreApplication.translate("MainWindow", u"Raw data", None))
        self.rb_raw_diff.setText(QCoreApplication.translate("MainWindow", u"diff", None))
        self.rb_raw_on.setText(QCoreApplication.translate("MainWindow", u"On", None))
        self.rb_raw_off.setText(QCoreApplication.translate("MainWindow", u"Off", None))
        ___qtablewidgetitem20 = self.table_scales.horizontalHeaderItem(0)
        ___qtablewidgetitem20.setText(QCoreApplication.translate("MainWindow", u"scale", None))

        __sortingEnabled1 = self.table_scales.isSortingEnabled()
        self.table_scales.setSortingEnabled(False)
        ___qtablewidgetitem21 = self.table_scales.item(0, 0)
        ___qtablewidgetitem21.setText(QCoreApplication.translate("MainWindow", u"1", None))
        ___qtablewidgetitem22 = self.table_scales.item(1, 0)
        ___qtablewidgetitem22.setText(QCoreApplication.translate("MainWindow", u"1", None))
        ___qtablewidgetitem23 = self.table_scales.item(2, 0)
        ___qtablewidgetitem23.setText(QCoreApplication.translate("MainWindow", u"1", None))
        ___qtablewidgetitem24 = self.table_scales.item(3, 0)
        ___qtablewidgetitem24.setText(QCoreApplication.translate("MainWindow", u"1", None))
        ___qtablewidgetitem25 = self.table_scales.item(4, 0)
        ___qtablewidgetitem25.setText(QCoreApplication.translate("MainWindow", u"1", None))
        ___qtablewidgetitem26 = self.table_scales.item(5, 0)
        ___qtablewidgetitem26.setText(QCoreApplication.translate("MainWindow", u"1", None))
        ___qtablewidgetitem27 = self.table_scales.item(6, 0)
        ___qtablewidgetitem27.setText(QCoreApplication.translate("MainWindow", u"1", None))
        ___qtablewidgetitem28 = self.table_scales.item(7, 0)
        ___qtablewidgetitem28.setText(QCoreApplication.translate("MainWindow", u"1", None))
        ___qtablewidgetitem29 = self.table_scales.item(8, 0)
        ___qtablewidgetitem29.setText(QCoreApplication.translate("MainWindow", u"1", None))
        ___qtablewidgetitem30 = self.table_scales.item(9, 0)
        ___qtablewidgetitem30.setText(QCoreApplication.translate("MainWindow", u"1", None))
        self.table_scales.setSortingEnabled(__sortingEnabled1)

        self.group_selected_data.setTitle(QCoreApplication.translate("MainWindow", u"Selected data", None))
        self.btn_merge.setText(QCoreApplication.translate("MainWindow", u"Merge", None))
        self.lbl_lower_limit.setText(QCoreApplication.translate("MainWindow", u"lower", None))
        self.lbl_upper_limit.setText(QCoreApplication.translate("MainWindow", u"upper", None))
        self.chk_use_raw.setText(QCoreApplication.translate("MainWindow", u"use raw", None))
        self.btn_save.setText(QCoreApplication.translate("MainWindow", u"Save", None))
    # retranslateUi

