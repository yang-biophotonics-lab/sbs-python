import numpy as np
from pandas import read_csv
import time
import matplotlib.pyplot as plt
import pipeline

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication,QMainWindow,QGridLayout, QTableWidgetItem
from PyQt5.QtCore import QTimer,pyqtSlot,QThread
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import sys,random, time,os,re
from Ui_dou3 import Ui_MainWindow
import pyarrow
from joblib import Parallel, delayed
##read file

t1 = time.perf_counter()
filename = 'D:\\workspace\\01\\20240809\\test2\\test2.txt'
data = read_csv(filename,skiprows=1,sep=';', header=None, usecols=(0,3), engine='pyarrow')#if throw error"excepted byte,but int",update pandas
file_append = 'D:\\workspace\\01\\20240809\\test2\\test2_append.txt'
data_append = read_csv(file_append,skiprows=5,sep=';', header=None, usecols=(1,2), engine='pyarrow')
t2 = time.perf_counter()

##initialize variables
class experiment_settings:
    samples_ignored = (10,10)
    shift=(4,3)
    x_pixel_to_um = 0.1
    y_pixel_to_um = 0.2
    freq_num = 141
    range_X = 40
    range_Y = 60
    n_planes = 6
    pixel_X = int(range_X/x_pixel_to_um +1)
    pixel_Y = int(range_Y/y_pixel_to_um +1)
    pixel_num = pixel_Y * pixel_X

ifdouble =True

data = data.to_numpy(dtype=np.float32)
data_FC= data[:,1]
data_LIA = data[:,0] + 1
data_append = data_append.to_numpy(dtype=np.float32)
del data

reffreq = data_append[-experiment_settings.freq_num:,1][experiment_settings.samples_ignored[0]:experiment_settings.freq_num-experiment_settings.samples_ignored[1]]
refvolt = data_append[-experiment_settings.freq_num:,0][experiment_settings.samples_ignored[0]:experiment_settings.freq_num-experiment_settings.samples_ignored[1]]
#test = np.array([1.3, 2.3, 3.3])
del data_append

interpolator = interp1d(refvolt, reffreq, kind='linear', fill_value="extrapolate")
data_FC= interpolator(data_FC)



data_X,data_Y=pipeline.preprocess(data_LIA,data_FC,experiment_settings.freq_num,experiment_settings.pixel_num,experiment_settings.n_planes)
data_X = data_X + 0.09
t3 = time.perf_counter()
##single peak fit
initial_parameters = np.zeros((experiment_settings.pixel_num,4),dtype=np.float32)
initial_parameters[:,0] = 1
initial_parameters[:,1] = 5.3
initial_parameters[:,2] = 0.3
initial_parameters[:,3] = 1

constraints = np.zeros((experiment_settings.pixel_num,8),dtype=np.float32)
constraints[:,0] = 0.2
constraints[:,1] = 2
constraints[:,2] = 4.8
constraints[:,3] = 5.65
constraints[:,4] = 0.1
constraints[:,5] = 0.6
constraints[:,6] = 0
constraints[:,7] = 2



volume, single_fit = pipeline.singlepeak(data_X,data_Y,experiment_settings,initial_parameters,constraints)
#volume, single_fit = pipeline.cpufit(data_X,data_Y,experiment_settings,initial_parameters,constraints)

if ifdouble:
    volumesingle_peak, volume_double_peak, single_peak, double_peak = pipeline.double_peak(data_X,data_Y,experiment_settings,single_fit,constraints)
t4 = time.perf_counter()


# class Myplot for plotting with matplotlib
class Myplot(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        # new figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # activate figure window
        # super(Plot_dynamic,self).__init__(self.fig)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        #self.fig.canvas.mpl_connect('button_press_event', self)
        # sub plot by self.axes
        self.axes= self.fig.add_subplot(111)
        # initial figure
        self.compute_initial_figure()
        FigureCanvas.updateGeometry(self)
    def compute_initial_figure(self):
        pass



# class for plotting a specific figure static or dynamic
class static_fig(Myplot):
    def __init__(self,*args,**kwargs):
        Myplot.__init__(self,*args,**kwargs)

    def compute_initial_figure(self):
        z=0
        self.minfreq = 5
        self.maxfreq = 5.6
        pass

        

class d_fig(Myplot):
    def __init__(self,*args,**kwargs):
        Myplot.__init__(self,*args,**kwargs)

    def compute_initial_figure(self):
        self.axes.set_title("brillouin")
        self.axes.set_xlabel("freq")
        self.axes.set_ylabel("gain (a.u.)")

class histo_fig(Myplot):
    def __init__(self,*args,**kwargs):
        Myplot.__init__(self,*args,**kwargs)

    def compute_initial_figure(self):
        self.axes.set_title("")
        self.axes.set_xlabel("")
        self.axes.set_ylabel("")

# class for the application window
class AppWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(AppWindow,self).__init__(parent)
        self.setupUi(self)
        # ^O^ static_fig can changed to any other function
        #self.fig1=static_fig(width=5, height=4, dpi=100)
        self.f1 = static_fig(width=5, height=3, dpi=72)
        self.f2 = static_fig(width=5, height=3, dpi=72)
        self.zplane.setMaximum(experiment_settings.n_planes)
        self.zplane.setMinimum(1)
        self.zplane.setValue(1)
        self.zplane.valueChanged.connect(self.update_fig)
        self.zplane.valueChanged.connect(self.update_fig2)
        self.zplane.valueChanged.connect(self.update_freq)
        self.f1.minfreq = 5.1
        self.f1.maxfreq = 5.45
        self.f2.minfreq = 0.2
        self.f2.maxfreq = 0.4
        self.minfreq.setProperty("value", self.f1.minfreq)
        self.minfreq.valueChanged.connect(self.update_fig)
        self.maxfreq.setProperty("value", self.f1.maxfreq)
        self.maxfreq.valueChanged.connect(self.update_fig)
        self.minfreq_2.setProperty("value", self.f2.minfreq)
        self.minfreq_2.valueChanged.connect(self.update_fig2)
        self.maxfreq_2.setProperty("value", self.f2.maxfreq)
        self.maxfreq_2.valueChanged.connect(self.update_fig2)
        self.fq = d_fig(width=5, height=3, dpi=72)
        self.hist1 = histo_fig(width=5, height=3, dpi=72)
        self.hist2 = histo_fig(width=5, height=3, dpi=72)
        self.f1.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.f1.setFocus()
        self.f2.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.f2.setFocus()
        self.f1.mpl_connect('button_press_event', self.OnClick)
        self.f2.mpl_connect('button_press_event', self.OnClick)
        self.f1.mpl_connect('key_press_event', self.OnMove)
        self.f2.mpl_connect('key_press_event', self.OnMove)
        self.comboBox.currentIndexChanged.connect(self.update_fig)
        self.comboBox.currentIndexChanged.connect(self.update_freq)
        self.comboBox_2.setCurrentIndex(2)
        self.comboBox_2.currentIndexChanged.connect(self.update_fig2)
        self.comboBox_2.currentIndexChanged.connect(self.update_freq)
        # add NavigationToolbar in the figure (widgets)
        self.f_ntb1 = NavigationToolbar(self.f1, self)
        self.f_ntb2 = NavigationToolbar(self.f2, self)
        self.f_ntb3 = NavigationToolbar(self.fq, self)
        self.f_ntb4 = NavigationToolbar(self.hist1, self)
        self.f_ntb5 = NavigationToolbar(self.hist2, self)
        #self.Start_plot.clicked.connect(self.plot_cos)
        # add the static_fig in the Plot box
        self.gridlayout1 = QGridLayout(self.fig1)
        self.gridlayout1.addWidget(self.f1)
        self.gridlayout1.addWidget(self.f_ntb1)
        self.gridlayout2 = QGridLayout(self.fig2)
        self.gridlayout2.addWidget(self.f2)
        self.gridlayout2.addWidget(self.f_ntb2)
        # add the dynamic_fig in the Plot box
        self.gridlayout3 = QGridLayout(self.freq)
        self.gridlayout3.addWidget(self.fq)
        self.gridlayout3.addWidget(self.f_ntb3)

        self.gridlayout4 = QGridLayout(self.histogram)
        self.gridlayout4.addWidget(self.hist1)
        self.gridlayout4.addWidget(self.f_ntb4)

        self.gridlayout5 = QGridLayout(self.histogram_2)
        self.gridlayout5.addWidget(self.hist2)
        self.gridlayout5.addWidget(self.f_ntb5)
        #global ifdouble
        if ifdouble:
            self.double_peak.setChecked(True)
            self.update_comboBox()
        else:
            self.double_peak.setChecked(False)
        self.double_peak.setChecked(False)
        self.double_peak.toggled.connect(self.update_comboBox)
        
        self.x1 = experiment_settings.pixel_Y//2
        self.y1 = experiment_settings.pixel_X//2
        self.update_fig()
        self.update_fig2()
        self.update_freq()



    def OnClick(self,event=None):
        self.y1=int(event.xdata/experiment_settings.x_pixel_to_um-2*max(experiment_settings.shift))%(experiment_settings.pixel_X-2*max(experiment_settings.shift))+2*max(experiment_settings.shift)
        self.x1=int(event.ydata/experiment_settings.y_pixel_to_um)%(experiment_settings.pixel_Y)
        self.update_freq()
        self.update_mark()
        self.update_mark2()


    def OnMove(self,event=None):
        if event.key == 'left':
            self.y1=(self.y1-1-2*max(experiment_settings.shift))%(experiment_settings.pixel_X-2*max(experiment_settings.shift))+2*max(experiment_settings.shift)
        elif event.key == 'right':
            self.y1=(self.y1+1-2*max(experiment_settings.shift))%(experiment_settings.pixel_X-2*max(experiment_settings.shift))+2*max(experiment_settings.shift)
        elif event.key == 'up':
            self.x1=(self.x1-1)%(experiment_settings.pixel_Y)
        elif event.key == 'down':
            self.x1=(self.x1+1)%(experiment_settings.pixel_Y)

        self.update_freq()    
        self.update_mark()
        self.update_mark2()    
    def update_comboBox(self):

        if self.double_peak.isChecked():
            self.comboBox.addItem("")
            self.comboBox.setItemText(5, QtCore.QCoreApplication.translate("MainWindow", "l1_shift"))
            self.comboBox.addItem("")
            self.comboBox.setItemText(6, QtCore.QCoreApplication.translate("MainWindow", "l1_amplitude"))
            self.comboBox.addItem("")
            self.comboBox.setItemText(7, QtCore.QCoreApplication.translate("MainWindow", "l1_width"))
            self.comboBox.addItem("")
            self.comboBox.setItemText(8, QtCore.QCoreApplication.translate("MainWindow", "l2_shift"))
            self.comboBox.addItem("")
            self.comboBox.setItemText(9, QtCore.QCoreApplication.translate("MainWindow", "l2_amplitude"))
            self.comboBox.addItem("")
            self.comboBox.setItemText(10, QtCore.QCoreApplication.translate("MainWindow", "l2_width"))
            self.comboBox.addItem("")
            self.comboBox.setItemText(11, QtCore.QCoreApplication.translate("MainWindow", "l12_offset"))
            self.comboBox.addItem("")
            self.comboBox.setItemText(12, QtCore.QCoreApplication.translate("MainWindow", "l12_error"))
            self.comboBox_2.addItem("")
            self.comboBox_2.setItemText(5, QtCore.QCoreApplication.translate("MainWindow", "l1_shift"))
            self.comboBox_2.addItem("")
            self.comboBox_2.setItemText(6, QtCore.QCoreApplication.translate("MainWindow", "l1_amplitude"))
            self.comboBox_2.addItem("")
            self.comboBox_2.setItemText(7, QtCore.QCoreApplication.translate("MainWindow", "l1_width"))
            self.comboBox_2.addItem("")
            self.comboBox_2.setItemText(8, QtCore.QCoreApplication.translate("MainWindow", "l2_shift"))
            self.comboBox_2.addItem("")
            self.comboBox_2.setItemText(9, QtCore.QCoreApplication.translate("MainWindow", "l2_amplitude"))
            self.comboBox_2.addItem("")
            self.comboBox_2.setItemText(10, QtCore.QCoreApplication.translate("MainWindow", "l2_width"))
            self.comboBox_2.addItem("")
            self.comboBox_2.setItemText(11, QtCore.QCoreApplication.translate("MainWindow", "l12_offset"))
            self.comboBox_2.addItem("")
            self.comboBox_2.setItemText(12, QtCore.QCoreApplication.translate("MainWindow", "l12_error"))
   
    def update_fig(self):
        
        #update f1 hist1
        z=int(self.zplane.value()-1)
        para = self.comboBox.currentText()   
        minfreq = self.minfreq.value()
        maxfreq = self.maxfreq.value()
        imageshift = 2*max(experiment_settings.shift)
        self.f1.fig.clear() 
        self.f1.axes= self.f1.fig.add_subplot(111)
        self.hist1.axes.cla()      
        if para == 'shift':
            x_0 = volume.shift[z]
            x_1 = single_fit.parameters[z,:,1]
        elif para == 'amplitude':
            x_0 = volume.amplitude[z]
            x_1 = single_fit.parameters[z,:,0]
        elif para == 'width':
            x_0 = volume.width[z]
            x_1 = single_fit.parameters[z,:,2]
        elif para == 'offset':
            x_0 = volume.offset[z]
            x_1 = single_fit.parameters[z,:,3]
        elif para == 'error':
            x_0 = volume.error[z]
            x_1 = single_fit.chi_squares[z,:]
        elif para == 'l1_shift':
            x_0 = volume_double_peak.l1_shift[z]
            x_1 = double_peak.parameters[z,:,1]
        elif para == 'l1_amplitude':
            x_0 = volume_double_peak.l1_amplitude[z]
            x_1 = double_peak.parameters[z,:,0]
        elif para == 'l1_width':
            x_0 = volume_double_peak.l1_width[z]
            x_1 = double_peak.parameters[z,:,2]
        elif para == 'l2_shift':
            x_0 = volume_double_peak.l2_shift[z]
            x_1 = double_peak.parameters[z,:,4]
        elif para == 'l2_amplitude':
            x_0 = volume_double_peak.l2_amplitude[z]
            x_1 = double_peak.parameters[z,:,3]
        elif para == 'l2_width':
            x_0 = volume_double_peak.l2_width[z]
            x_1 = double_peak.parameters[z,:,5]
        elif para == 'l12_offset':
            x_0 = volume_double_peak.l12_offset[z]
            x_1 = double_peak.parameters[z,:,6]
        elif para == 'l12_error':
            x_0 = volume_double_peak.error[z]
            x_1 = double_peak.chi_squares[z,:]
                  
        x=np.linspace(0,np.size(x_0,0)*experiment_settings.y_pixel_to_um,np.size(x_0,0)+1,endpoint=True)
        y=np.linspace(0,np.size(x_0,1)*experiment_settings.x_pixel_to_um,np.size(x_0,1)+1,endpoint=True)
        self.f1.axes.invert_yaxis()
        self.f1.c=self.f1.axes.pcolormesh(y[imageshift:np.size(y)-imageshift],x,x_0[:,imageshift:np.size(x_0,1)-imageshift],cmap='jet',shading="auto",vmin=minfreq,vmax=maxfreq)
        self.f1.plotmark = self.f1.axes.plot(self.y1*experiment_settings.x_pixel_to_um, self.x1*experiment_settings.y_pixel_to_um, '+k')
        self.f1.axes.set_title(f"{para}")
        self.f1.axes.set_xlabel("X in um")
        self.f1.axes.set_ylabel("Y in um")
        self.f1.axes.set_aspect('equal', adjustable='box')
        self.f1.fig.colorbar(self.f1.c)
        self.hist1.axes.hist(x_1,50)  
        self.f1.draw() 
        self.hist1.draw()  
    def update_mark(self):
        # 更新 plot 点
        self.f1.plotmark[0].set_data(self.y1 * experiment_settings.x_pixel_to_um,
                                      self.x1 * experiment_settings.y_pixel_to_um)
        # 重新绘制图形
        self.f1.draw() 
    def update_mark2(self):
        self.f2.plotmark[0].set_data(self.y1 * experiment_settings.x_pixel_to_um,
                                      self.x1 * experiment_settings.y_pixel_to_um)
        
        self.f2.draw()
    def update_fig2(self):
        #update f2 hist2
        z=int(self.zplane.value()-1)
        para = self.comboBox_2.currentText()   
        minfreq = self.minfreq_2.value()
        maxfreq = self.maxfreq_2.value()
        imageshift = 2*max(experiment_settings.shift)
        self.f2.fig.clear() 
        self.f2.axes= self.f2.fig.add_subplot(111)
        self.hist2.axes.cla()      
        if para == 'shift':
            x_0 = volume.shift[z]
            x_1 = single_fit.parameters[z,:,1]
        elif para == 'amplitude':
            x_0 = volume.amplitude[z]
            x_1 = single_fit.parameters[z,:,0]
        elif para == 'width':
            x_0 = volume.width[z]
            x_1 = single_fit.parameters[z,:,2]
        elif para == 'offset':
            x_0 = volume.offset[z]
            x_1 = single_fit.parameters[z,:,3]
        elif para == 'error':
            x_0 = volume.error[z]
            x_1 = single_fit.chi_squares[z,:]
        elif para == 'l1_shift':
            x_0 = volume_double_peak.l1_shift[z]
            x_1 = double_peak.parameters[z,:,1]
        elif para == 'l1_amplitude':
            x_0 = volume_double_peak.l1_amplitude[z]
            x_1 = double_peak.parameters[z,:,0]
        elif para == 'l1_width':
            x_0 = volume_double_peak.l1_width[z]
            x_1 = double_peak.parameters[z,:,2]
        elif para == 'l2_shift':
            x_0 = volume_double_peak.l2_shift[z]
            x_1 = double_peak.parameters[z,:,4]
        elif para == 'l2_amplitude':
            x_0 = volume_double_peak.l2_amplitude[z]
            x_1 = double_peak.parameters[z,:,3]
        elif para == 'l2_width':
            x_0 = volume_double_peak.l2_width[z]
            x_1 = double_peak.parameters[z,:,5]
        elif para == 'l12_offset':
            x_0 = volume_double_peak.l12_offset[z]
            x_1 = double_peak.parameters[z,:,6]
        elif para == 'l12_error':
            x_0 = volume_double_peak.error[z]
            x_1 = double_peak.chi_squares[z,:]
                  
        x=np.linspace(0,np.size(x_0,0)*experiment_settings.y_pixel_to_um,np.size(x_0,0)+1,endpoint=True)
        y=np.linspace(0,np.size(x_0,1)*experiment_settings.x_pixel_to_um,np.size(x_0,1)+1,endpoint=True)
        self.f2.axes.invert_yaxis()
        self.f2.c=self.f2.axes.pcolormesh(y[imageshift:np.size(y)-imageshift],x,x_0[:,imageshift:np.size(x_0,1)-imageshift],cmap='jet',shading="auto",vmin=minfreq,vmax=maxfreq)
        self.f2.plotmark = self.f2.axes.plot(self.y1*experiment_settings.x_pixel_to_um, self.x1*experiment_settings.y_pixel_to_um, '+k')
        self.f2.axes.set_title(f"{para}")
        self.f2.axes.set_xlabel("X in um")
        self.f2.axes.set_ylabel("Y in um")
        self.f2.axes.set_aspect('equal', adjustable='box')
        self.f2.fig.colorbar(self.f2.c)
        self.hist2.axes.hist(x_1,50)#,(0,0.5))  
        self.f2.draw() 
        self.hist2.draw()  

    def update_freq(self):
        z=int(self.zplane.value()-1)
        data_xplot = data_X[z,volume.index[z,self.x1,self.y1]]
        data_yplot = data_Y[z,volume.index[z,self.x1,self.y1]]
        ampl = volume.amplitude[z,self.x1,self.y1]
        x_00 = volume.shift[z,self.x1,self.y1]
        gama = volume.width[z,self.x1,self.y1]
        ofst = volume.offset[z,self.x1,self.y1]
        eror = volume.error[z,self.x1,self.y1]
        fitted = ampl/(1 + ((data_xplot - x_00)/gama)**2) + ofst

        used = volume.weights[z,volume.index[z,self.x1,self.y1]].astype(np.bool_)

        self.fq.axes.cla()
        self.fq.axes.plot(data_xplot[used],fitted[used],'--',label = 'single_fit')
        
        self.text.setText(f'File:'+filename)
                          
        # 设置表格的行数和列数
        self.table.setRowCount(15)  # 5 行
        self.table.setColumnCount(3)  # 3 列
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        
        # 设置表格标题
        headers = ["Variable", "Value", "Unit"]
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        # 填充表格数据
        single_peak_data = {
            "shift": (x_00, "GHz"),
            "amplitude": (ampl, "ua"),
            "gamma": (gama, "ua"),
            "offset": (ofst, "Volt"),
            "error": (eror, "ua")
        }
        self.fill_table_data(single_peak_data)

        if self.double_peak.isChecked():
            ampl_1 = volume_double_peak.l1_amplitude[z,self.x1,self.y1]
            x_00_1 = volume_double_peak.l1_shift[z,self.x1,self.y1]
            gama_1 = volume_double_peak.l1_width[z,self.x1,self.y1]
            ampl_2 = volume_double_peak.l2_amplitude[z,self.x1,self.y1]
            x_00_2 = volume_double_peak.l2_shift[z,self.x1,self.y1]
            gama_2 = volume_double_peak.l2_width[z,self.x1,self.y1]
            offset = volume_double_peak.l12_offset[z,self.x1,self.y1]
            errorr = volume_double_peak.error[z,self.x1,self.y1]
            double_fitted = ampl_1/(1 + ((data_xplot - x_00_1)/gama_1)**2) + ampl_2/(1 + ((data_xplot - x_00_2)/gama_2)**2) + offset
            peak1 = ampl_1/(1 + ((data_xplot - x_00_1)/gama_1)**2) + offset
            peak2 = ampl_2/(1 + ((data_xplot - x_00_2)/gama_2)**2) + offset
            used = volume.weights[z,volume.index[z,self.x1,self.y1]].astype(np.bool_)
            
            self.fq.axes.plot(data_xplot[used],peak1[used],label = 'L1')
            self.fq.axes.plot(data_xplot[used],peak2[used],label = 'L2')
            self.fq.axes.plot(data_xplot[used],double_fitted[used],label = 'double_fit')
            # 填充双峰数据
            double_peak_data = {
                "": ("", ""),
                "l1_shift": (x_00_1, "GHz"),
                "l1_amplitude": (ampl_1, "ua"),
                "l1_gamma": (gama_1, "ua"),
                " ": ("", ""),
                "l2_shift": (x_00_2, "GHz"),
                "l2_amplitude": (ampl_2, "ua"),
                "l2_gamma": (gama_2, "ua"),
                "offset": (offset, "Volt"),
                "error": (errorr, "ua")
            }
            self.fill_table_data(double_peak_data, start_row=5)
        
        self.fq.axes.plot(data_xplot[used],data_yplot[used],'.', label = 'raw data')
        #self.fq.axes.plot(data_xplot[np.logical_not(used)],data_yplot[np.logical_not(used)],'x', label = 'raw data ignored')
       
        self.fq.axes.set_title(f"Signal for pixel({self.y1:d}, {self.x1:d}, {z+1:d})\n({self.y1*experiment_settings.x_pixel_to_um:.1f}um, {self.x1*experiment_settings.y_pixel_to_um:.1f}um)")
        self.fq.axes.set_xlabel("freq (GHz)")
        self.fq.axes.set_ylabel("gain (a.u.)")
        self.fq.axes.legend()
        self.fq.draw()
    def fill_table_data(self, data_dict, start_row=0):
        row = start_row
        for variable, (value, unit) in data_dict.items():
            self.table.setItem(row, 0, QTableWidgetItem(variable))
            if isinstance(value, str):
                self.table.setItem(row, 1, QTableWidgetItem(value))
            else:self.table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))
            self.table.setItem(row, 2, QTableWidgetItem(unit))
            row += 1


app = QApplication(sys.argv)
win = AppWindow()
t5 = time.perf_counter()
current_dir = os.getcwd()
subfolder_path = os.path.join(current_dir, 'tiff_export')
if not os.path.exists(subfolder_path):
    os.makedirs(subfolder_path)
image_path = os.path.join(subfolder_path, "shift")
pipeline.save_as_tiff(volume.shift, image_path, "jet")
print('''读文件：%.4fs\n预处理：%.4fs\n拟合：%.4fs\n总计：%.4fs\n'''%(t2-t1,t3-t2,t4-t3,t5-t1))
win.time.setText(f'{t5-t1:.4f}')
win.show()
sys.exit(app.exec_())
