#!/usr/bin/env /home/uemura/Apps/anaconda3_2021Aprl/bin/python
import sys, os, string, io, glob, re, yaml, math, time, shutil, natsort
import numpy as np
import pandas as pd

import silx
from silx.gui import qt
app = qt.QApplication([])
import time
import silx.gui.colors as silxcolors
from silx.gui.plot import PlotWindow, Plot1D, Plot2D, PlotWidget,items
import silx.gui.colors as silxcolors
import silx.io as silxIO
import tifffile as tif
from scipy.interpolate import interp1d

from mw import Ui_MainWindow

def msg(txt):
    _msg = qt.QMessageBox()
    _msg.setIcon(qt.QMessageBox.Warning)
    _msg.setText(txt)
    _msg.setStandardButtons(qt.QMessageBox.Ok)
    return _msg

default_cmap = silxcolors.Colormap(name='jet')

class Ui(qt.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        self.u = Ui_MainWindow()
        self.u.setupUi(self)

        self.raw_plot = Plot1D()
        layout = qt.QVBoxLayout()
        self.u.widget.setLayout(layout)
        layout.addWidget(self.raw_plot)

        self.conv_plot = Plot1D()
        layout = qt.QVBoxLayout()
        self.u.widget_2.setLayout(layout)
        layout.addWidget(self.conv_plot)

        self.hist = Plot1D()
        layout = qt.QVBoxLayout()
        self.u.widget_3.setLayout(layout)
        layout.addWidget(self.hist)


        def changeHeaders():
            if self.u.tabWidget.currentIndex() == 0:
                self.u.tableWidget.item(0, 0).setText('#Energy [eV]')
                self.u.tableWidget.item(0, 1).setText('#PPODL [ps]')
                print (self.u.tableWidget.itemAt(0, 0).text())
            elif self.u.tabWidget.currentIndex() == 1:
                self.u.tableWidget.item(0, 0).setText('Energy')
                self.u.tableWidget.item(0, 1).setText('#motor')

        self.u.tabWidget.currentChanged.connect(changeHeaders)

        def reloadDir():
            _dir = self.u.textBrowser.toPlainText()
            if self.u.tabWidget.currentIndex() == 0:
                ch = self.u.ch2A.isChecked() * '2_A' + self.u.ch2B.isChecked() * '2_B'
                if _dir and os.path.isdir(_dir):
                    self.u.textBrowser.clear()
                    self.u.textBrowser.append(_dir)
                    print (f'{ch}.csv')
                    files = [x.split('_')[0] for x in os.listdir(_dir) if f'{ch}.csv' in x]
                    self.u.listWidget.clear()
                    if files:
                        self.u.listWidget.addItems(natsort.natsorted(files))
                    else:
                        msg("Please select a directory which includes tiff files ;_;").exec_()
                else:
                    msg("!! You did not select a directory :(").exec_()
            elif self.u.tabWidget.currentIndex() == 1:
                ch = self.u.rb_pd.isChecked() * '' + self.u.rb_mpccd.isChecked() * 'mpccd'
                if _dir and os.path.isdir(_dir):
                    self.u.textBrowser.clear()
                    self.u.textBrowser.append(_dir)
                    dirs = [x for x in os.listdir(_dir) if os.path.isdir(_dir+'/'+x) and re.match('r\d+',x)]
                    self.u.listWidget.clear()
                    if dirs:
                        self.u.listWidget.addItems(natsort.natsorted(dirs))
                    else:
                        msg("Please select a directory which includes tiff files ;_;").exec_()
                else:
                    msg("!! You did not select a directory :(").exec_()


        def selectDir():
            if os.path.isdir(self.u.textBrowser.toPlainText()):
                dat_dir = self.u.textBrowser.toPlainText()
            else:
                dat_dir = os.environ['HOME']
            self.u.listWidget.clear()
            FO_dialog = qt.QFileDialog(self)
            f = FO_dialog.getExistingDirectory(self, "Select a directory", dat_dir,)
            if f and os.path.isdir(f):
                self.u.textBrowser.clear()
                self.u.textBrowser.append(f)
                reloadDir()

            else:
                msg("!! You did not select a directory :(").exec_()

        def plot_multi_runs():
            self.hist.clear()
            if self.u.tabWidget.currentIndex() == 0:
                ch = self.u.ch2A.isChecked() * '2_A' + self.u.ch2B.isChecked() * '2_B'
                ext = f'I0_2_C_If_{ch}'
                xlabel = self.u.tableWidget.item(0, 0).text() * (self.u.radioButton.isChecked())+ \
                         self.u.tableWidget.item(0, 1).text() * (self.u.radioButton_2.isChecked())
                ylabel_diff = (f'{self.u.tableWidget.item(0, 4).text()}:{ch}')
                ylabel_on = (f'{self.u.tableWidget.item(0, 2).text()}:{ch}')
                ylabel_off = (f'{self.u.tableWidget.item(0, 3).text()}:{ch}')
                GraphXLabel = 'Energy (eV)'*self.u.radioButton.isChecked() + 'PPODL (ps)'*self.u.radioButton_2.isChecked()
                GraphYLabel = self.u.rb_diff.isChecked() * 'diff'+ \
                         self.u.rb_on.isChecked() * 'XAS'+ \
                         self.u.rb_off.isChecked() * 'XAS'

            else:
                ch = self.u.rb_pd.isChecked() * '' + self.u.rb_mpccd.isChecked() * '_mpccd'
                ext = self.u.rb_seng.isChecked()*'escan'+self.u.rb_sdelay.isChecked()*'mscan'+f'{ch}'
                xlabel = self.u.tableWidget.item(0, 0).text() * (self.u.rb_seng.isChecked()) + \
                         self.u.tableWidget.item(0, 1).text() * (self.u.rb_sdelay.isChecked())
                ylabel_diff = (f'{self.u.tableWidget.item(0, 4).text()}')
                ylabel_on = (f'{self.u.tableWidget.item(0, 2).text()}')
                ylabel_off = (f'{self.u.tableWidget.item(0, 3).text()}')
                GraphXLabel = 'Energy (eV)' * self.u.rb_seng.isChecked() + 'PPODL (ps)' * self.u.rb_sdelay.isChecked()
                GraphYLabel = self.u.rb_diff.isChecked() * 'diff' + \
                              self.u.rb_on.isChecked() * 'XAS' + \
                              self.u.rb_off.isChecked() * 'XAS'
            try:
                datdir = self.u.textBrowser.toPlainText()
                items = [x.text() for x in self.u.listWidget_2.selectedItems()]

                for k, _f in enumerate(items):
                    _datdir = datdir if self.u.tabWidget.currentIndex() == 0 else f'{datdir}/{_f}'
                    alpha = float(self.u.tableWidget_2.item(k, 0).text()) if k < 10 else 1
                    try:
                        df = pd.read_csv(f'{_datdir}/{_f}_{ext}.csv', delim_whitespace=True)
                        if self.u.rb_diff.isChecked():
                            self.hist.addCurve(df[xlabel],df[ylabel_on]-df[ylabel_off],linewidth=1.5,symbol='.',legend=_f)
                        elif self.u.rb_on.isChecked():
                            self.hist.addCurve(df[xlabel],df[ylabel_on],linewidth=1.5,symbol='.',legend=_f)
                        elif self.u.rb_off.isChecked():
                            self.hist.addCurve(df[xlabel],df[ylabel_off],linewidth=1.5,symbol='.',legend=_f)
                    except Exception as e:
                        print (f"Error for '_f': str{e}")
                self.hist.setGraphXLabel(GraphXLabel)
                self.hist.setGraphYLabel(GraphYLabel)

            except Exception as e:
                print(f"Error: str{e}")

        self.u.rb_diff.clicked.connect(plot_multi_runs)
        self.u.rb_on.clicked.connect(plot_multi_runs)
        self.u.rb_off.clicked.connect(plot_multi_runs)

        def setItems_combobox2():
            if self.u.listWidget_2.selectedItems():
                items = [x.text() for x in self.u.listWidget_2.selectedItems()]
                self.u.comboBox_2.clear()
                self.u.comboBox_2.addItems(items)

                plot_multi_runs()

        def load_data():
            self.u.comboBox.clear()
            if self.u.listWidget.selectedItems():
                items = [x.text() for x in self.u.listWidget.selectedItems()]
                self.u.comboBox.addItems(items)

                self.u.listWidget_2.clear()
                self.u.listWidget_2.addItems(items)
                for j in range(self.u.listWidget_2.count()):
                    self.u.listWidget_2.item(j).setSelected(True)

                setItems_combobox2()


        def unselectFiles():
            if self.u.listWidget.selectedItems():
                for x in self.u.listWidget.selectedItems():
                    x.setSelected(False)


        def plot_single_run(rnum):
            if rnum:
                if self.u.tabWidget.currentIndex() == 0:
                    ch = self.u.ch2A.isChecked() * '2_A' + self.u.ch2B.isChecked() * '2_B'
                    datdir = self.u.textBrowser.toPlainText()
                    ext = f'I0_2_C_If_{ch}'
                    xlabel = self.u.tableWidget.item(0, 0).text() * (self.u.radioButton.isChecked()) + \
                             self.u.tableWidget.item(0, 1).text() * (self.u.radioButton_2.isChecked())
                    ylabel_diff = (f'{self.u.tableWidget.item(0, 4).text()}:{ch}')
                    ylabel_on = (f'{self.u.tableWidget.item(0, 2).text()}:{ch}')
                    ylabel_off = (f'{self.u.tableWidget.item(0, 3).text()}:{ch}')
                    GraphXLabel = 'Energy (eV)' * self.u.radioButton.isChecked() + 'PPODL (ps)' * self.u.radioButton_2.isChecked()
                    # GraphYLabel = self.u.rb_diff.isChecked() * 'diff' + \
                    #               self.u.rb_on.isChecked() * 'XAS' + \
                    #               self.u.rb_off.isChecked() * 'XAS'

                else:
                    ch = self.u.rb_pd.isChecked() * '' + self.u.rb_mpccd.isChecked() * '_mpccd'
                    ext = self.u.rb_seng.isChecked() * 'escan' + self.u.rb_sdelay.isChecked() * 'mscan' + f'{ch}'
                    xlabel = self.u.tableWidget.item(0, 0).text() * (self.u.rb_seng.isChecked()) + \
                             self.u.tableWidget.item(0, 1).text() * (self.u.rb_sdelay.isChecked())
                    ylabel_diff = (f'{self.u.tableWidget.item(0, 4).text()}')
                    ylabel_on = (f'{self.u.tableWidget.item(0, 2).text()}')
                    ylabel_off = (f'{self.u.tableWidget.item(0, 3).text()}')
                    GraphXLabel = 'Energy (eV)' * self.u.rb_seng.isChecked() + 'PPODL (ps)' * self.u.rb_sdelay.isChecked()
                    # GraphYLabel = self.u.rb_diff.isChecked() * 'diff' + \
                    #               self.u.rb_on.isChecked() * 'XAS' + \
                    #               self.u.rb_off.isChecked() * 'XAS'

                datdir = self.u.textBrowser.toPlainText()
                _datdir = datdir if self.u.tabWidget.currentIndex() == 0 else f'{datdir}/{rnum}'
                file = f'{rnum}_{ext}.csv'
                if not os.path.isfile(_datdir+'/'+file):
                    msg(f"!! <i><font color='red'>{file}</font> </i> files are not found. You do not select the scan type properly... :(").exec_()
                    return
                try:
                    df = pd.read_csv(_datdir+'/'+file,delim_whitespace=True)

                    self.raw_plot.addCurve(df[xlabel], df[ylabel_on], color='red', linewidth=1.5,symbol='.', legend='On')
                    self.raw_plot.addCurve(df[xlabel], df[ylabel_off], color='blue', linewidth=1.5,symbol='.', legend='Off')
                    self.raw_plot.addCurve(df[xlabel], df[ylabel_on] - df[ylabel_off], yaxis='right',color='green', linewidth=1.5, symbol='.', legend='diff')
                    self.raw_plot.setGraphXLabel(GraphXLabel)
                    self.raw_plot.setGraphYLabel('XAS')
                    self.raw_plot.setGraphYLabel('$\Delta$XAS', axis='right')
                    self.raw_plot.setGraphTitle(f'{rnum}')

                except Exception as e:
                    msg(f"!! {str(e)}").exec_()

                plot_multi_runs()



        def merge():
            if self.u.comboBox_2.currentText():
                rnum = self.u.comboBox_2.currentText()
                ch = self.u.ch2A.isChecked() * '2_A' + self.u.ch2B.isChecked() * '2_B'
                datdir = self.u.textBrowser.toPlainText()
                _datdir = datdir if self.u.tabWidget.currentIndex() == 0 else f'{datdir}/{rnum}'

                if self.u.tabWidget.currentIndex() == 0:
                    ch = self.u.ch2A.isChecked() * '2_A' + self.u.ch2B.isChecked() * '2_B'
                    ext = f'I0_2_C_If_{ch}'
                    xlabel = self.u.tableWidget.item(0, 0).text() * (self.u.radioButton.isChecked())+ \
                             self.u.tableWidget.item(0, 1).text() * (self.u.radioButton_2.isChecked())
                    ylabel_diff = (f'{self.u.tableWidget.item(0, 4).text()}:{ch}')
                    ylabel_on = (f'{self.u.tableWidget.item(0, 2).text()}:{ch}')
                    ylabel_off = (f'{self.u.tableWidget.item(0, 3).text()}:{ch}')
                    GraphXLabel = 'Energy (eV)'*self.u.radioButton.isChecked() + 'PPODL (ps)'*self.u.radioButton_2.isChecked()
                    GraphYLabel = self.u.rb_diff.isChecked() * 'diff'+ \
                             self.u.rb_on.isChecked() * 'XAS'+ \
                             self.u.rb_off.isChecked() * 'XAS'

                else:
                    ch = self.u.rb_pd.isChecked() * '' + self.u.rb_mpccd.isChecked() * '_mpccd'
                    ext = self.u.rb_seng.isChecked()*'escan'+self.u.rb_sdelay.isChecked()*'mscan'+f'{ch}'
                    xlabel = self.u.tableWidget.item(0, 0).text() * (self.u.rb_seng.isChecked()) + \
                             self.u.tableWidget.item(0, 1).text() * (self.u.rb_sdelay.isChecked())
                    ylabel_diff = (f'{self.u.tableWidget.item(0, 4).text()}')
                    ylabel_on = (f'{self.u.tableWidget.item(0, 2).text()}')
                    ylabel_off = (f'{self.u.tableWidget.item(0, 3).text()}')
                    GraphXLabel = 'Energy (eV)' * self.u.rb_seng.isChecked() + 'PPODL (ps)' * self.u.rb_sdelay.isChecked()
                    GraphYLabel = self.u.rb_diff.isChecked() * 'diff' + \
                                  self.u.rb_on.isChecked() * 'XAS' + \
                                  self.u.rb_off.isChecked() * 'XAS'

                xas_on, xas_off = [], []
                df_model = pd.read_csv(_datdir + '/' + f'{rnum}_{ext}.csv', delim_whitespace=True)

                try:
                    datdir = self.u.textBrowser.toPlainText()
                    items = [x.text() for x in self.u.listWidget_2.selectedItems()]
                    x = df_model[xlabel].values
                    for _f in items:
                        _datdir = datdir if self.u.tabWidget.currentIndex() == 0 else f'{datdir}/{_f}'
                        df = pd.read_csv(f'{_datdir}/{_f}_{ext}.csv', delim_whitespace=True)
                        func = interp1d(df[xlabel].values, df[ylabel_on].values, bounds_error=False)
                        xas_on.append(func(x))
                        func = interp1d(df[xlabel].values, df[ylabel_off].values, bounds_error=False)
                        xas_off.append(func(x))

                    xas_on = np.array(xas_on)
                    xas_off = np.array(xas_off)
                    self.conv_plot.addCurve(x, np.nanmean(xas_on, axis=0), color='red', linewidth=1.5, symbol='.',legend='On')
                    self.conv_plot.addCurve(x, np.nanmean(xas_off, axis=0), color='blue', linewidth=1.5, symbol='.',legend='Off')
                    self.conv_plot.addCurve(x, np.nanmean(xas_on, axis=0) - np.nanmean(xas_off, axis=0), color='green',yaxis='right', linewidth=1.5, symbol='.', legend='diff')

                except Exception as e:
                    msg("!! {str(e)}").exec_()

        def savedata():
            if os.path.isdir(self.u.textBrowser.toPlainText()):
                FO_dialog = qt.QFileDialog(self)
                f = FO_dialog.getSaveFileName(self, "Set the output file name", self.u.textBrowser.toPlainText(), )
                if f[0]:
                    try:
                        x,On,_,_ = self.conv_plot.getCurve('On').getData()
                        x, Off, _, _ = self.conv_plot.getCurve('Off').getData()
                        x, diff, _, _ = self.conv_plot.getCurve('diff').getData()
                        label = '#Energy/eV'*(self.u.radioButton.isChecked()) + '#Motor/pls'*(self.u.radioButton_2.isChecked())
                        pd.DataFrame({
                            label: x,
                            'On': On,
                            'Off': Off,
                            'diff': diff
                        }).to_csv(f[0],index=False,sep=' ',float_format='%.6f')
                    except Exception as e:
                        msg(f"!! {str(e)}").exec_()

        self.u.pushButton.clicked.connect(selectDir)
        self.u.pushButton_2.clicked.connect(load_data)
        self.u.comboBox.currentTextChanged.connect(plot_single_run)
        self.u.pushButton_3.clicked.connect(merge)
        self.u.pushButton_4.clicked.connect(reloadDir)
        self.u.pushButton_5.clicked.connect(savedata)
        self.u.pushButton_6.clicked.connect(unselectFiles)
        self.u.listWidget_2.itemClicked.connect(setItems_combobox2)
        self.show()

if __name__ == '__main__':
    mw = Ui()
    mw.setWindowTitle('merge XAS at FXE')
    app.exec_()
