import numpy as np
import pygpufit.gpufit as gf

def preprocess(raw_data,raw_x,samples_per_pixel,n_pixels,n_planes):
    data_Y = raw_data.reshape(n_planes,n_pixels,samples_per_pixel)
    data_X = raw_x.reshape(n_planes,n_pixels,samples_per_pixel)
    return (data_X,data_Y)

def convert_to_2d(array_1d, pixel_Y, pixel_X, shift):
    array_2d = np.zeros((pixel_Y,pixel_X + max(shift)*2),dtype=np.float32)
    array_2d[0:pixel_Y,0:pixel_X] = np.reshape(array_1d,(pixel_Y,pixel_X))
    #np.roll(array_2d[:,1::2],shift[0])
    array_2d[0::2,:]=np.flip(array_2d[0::2,:],axis=1)
    return array_2d

import pylab
import numpy as np
import sys, os
from scipy.optimize import curve_fit
#from tqdm import tqdm

def func(x, a, x0, g, b):
  return a * g**2 / (g**2 + (x - x0)**2 ) + b; 

def cpufit(data_X,data_Y,experiment_settings,initial_parameters,constraints):
    
    
    class single_fit:
        parameters      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num,4),dtype=np.float32)
        states          = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        chi_squares     = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        number_iterations = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        execution_time  = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.int32)
        index =np.arange(experiment_settings.pixel_X*experiment_settings.pixel_Y)
    class volume:
        amplitude   = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        shift       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        width       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        offset      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        error       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        states      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        time        = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        index       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.int32)
        weights     = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num, experiment_settings.freq_num),dtype=np.int32)

    #pbar = tqdm(total=experiment_settings.n_planes*experiment_settings.pixel_num)
    #for z in [0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2]:
    for z in range(experiment_settings.n_planes):
        weights = np.zeros((experiment_settings.pixel_num, experiment_settings.freq_num))
        weights[:,experiment_settings.samples_ignored[0]: experiment_settings.freq_num-experiment_settings.samples_ignored[1]] = 1
        #single_fit.parameters[z], single_fit.states[z], single_fit.chi_squares[z], single_fit.number_iterations[z], single_fit.execution_time[z] = gpufit_lorentzian_constrainted(data_X[z], data_Y[z], experiment_settings.freq_begin, experiment_settings.freq_end, weights, initial_parameters, constraints) 
        for k in range(experiment_settings.pixel_num):
            single_fit.parameters[z][k], pcov = curve_fit(func, data_X[z,k,:], data_Y[z,k,:], p0=initial_parameters[k,:], bounds=(constraints[k,0::2],constraints[k,1::2]))                # 曲线拟合，popt为函数的参数list
            #y_pred = [func(i, popt[0], popt[1], popt[2], popt[3]) for i in x]    # 直接用函数和函数参数list来进行y值的计算
            #if k%10000 ==0 :print(k)
            #pbar.update(1)
        volume.amplitude[z] = convert_to_2d(single_fit.parameters[z,:,0], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.shift[z]     = convert_to_2d(single_fit.parameters[z,:,1], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.width[z]     = convert_to_2d(single_fit.parameters[z,:,2], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.offset[z]    = convert_to_2d(single_fit.parameters[z,:,3], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.error[z]     = convert_to_2d(single_fit.chi_squares[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.states[z]    = convert_to_2d(single_fit.states[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.time[z]      = convert_to_2d(single_fit.execution_time[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.index[z]     = convert_to_2d(single_fit.index, experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.weights[z]   = weights
    #pbar.close()
    if z>1:
        volume.amplitude[1::2]   = np.rot90(volume.amplitude[1::2], 2, (1,2))
        volume.shift[1::2]       = np.rot90(volume.shift[1::2], 2, (1,2))
        volume.width[1::2]       = np.rot90(volume.width[1::2], 2, (1,2))
        volume.offset[1::2]      = np.rot90(volume.offset[1::2], 2, (1,2))
        volume.error[1::2]       = np.rot90(volume.error[1::2], 2, (1,2))
        volume.states[1::2]      = np.rot90(volume.states[1::2], 2, (1,2))
        volume.index[1::2]       = np.rot90(volume.index[1::2], 2, (1,2))

    return volume, single_fit


def singlepeak(data_X,data_Y,experiment_settings,initial_parameters,constraints):
    
    class single_fit:
        parameters      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num,4),dtype=np.float32)
        states          = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        chi_squares     = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        number_iterations = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        execution_time  = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.int32)
        index =np.arange(experiment_settings.pixel_X*experiment_settings.pixel_Y)
    class volume:
        amplitude   = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        shift       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        width       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        offset      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        error       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        states      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        time        = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        index       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.int32)
        weights     = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num, experiment_settings.freq_num),dtype=np.int32)
    #pbar = tqdm(total=experiment_settings.n_planes*experiment_settings.pixel_num)
    #for z in [0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2, 0,1,2]:
    for z in range(experiment_settings.n_planes):
        #print('z=%d'%z)
        weights = np.zeros((experiment_settings.pixel_num, experiment_settings.freq_num))
        weights[:,experiment_settings.samples_ignored[0]: experiment_settings.freq_num-experiment_settings.samples_ignored[1]] = 1
        single_fit.parameters[z], single_fit.states[z], single_fit.chi_squares[z], single_fit.number_iterations[z], single_fit.execution_time[z] = gpufit_lorentzian_constrainted(data_X[z], data_Y[z], weights, initial_parameters, constraints) 
        #pbar.update(experiment_settings.pixel_num)
        volume.amplitude[z] = convert_to_2d(single_fit.parameters[z,:,0], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.shift[z]     = convert_to_2d(single_fit.parameters[z,:,1], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.width[z]     = convert_to_2d(single_fit.parameters[z,:,2], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.offset[z]    = convert_to_2d(single_fit.parameters[z,:,3], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.error[z]     = convert_to_2d(single_fit.chi_squares[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.states[z]    = convert_to_2d(single_fit.states[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.time[z]      = convert_to_2d(single_fit.execution_time[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.index[z]     = convert_to_2d(single_fit.index, experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume.weights[z]   = weights
    #pbar.close()
    if z>1:
        volume.amplitude[1::2]   = np.rot90(volume.amplitude[1::2], 2, (1,2))
        volume.shift[1::2]       = np.rot90(volume.shift[1::2], 2, (1,2))
        volume.width[1::2]       = np.rot90(volume.width[1::2], 2, (1,2))
        volume.offset[1::2]      = np.rot90(volume.offset[1::2], 2, (1,2))
        volume.error[1::2]       = np.rot90(volume.error[1::2], 2, (1,2))
        volume.states[1::2]      = np.rot90(volume.states[1::2], 2, (1,2))
        volume.index[1::2]       = np.rot90(volume.index[1::2], 2, (1,2))

    return volume, single_fit


def double_peak(data_X,data_Y,experiment_settings,previous_peak_fit,constraints):
    class single_peak:
        parameters      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num,4),dtype=np.float32)
        states          = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        chi_squares     = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        number_iterations = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        execution_time  = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.int32)
        index =np.arange(experiment_settings.pixel_X*experiment_settings.pixel_Y)
    class double_peak:
        parameters      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num,7),dtype=np.float32)
        states          = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        chi_squares     = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        number_iterations = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.float32)
        execution_time  = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num),dtype=np.int32)
        index =np.arange(experiment_settings.pixel_X*experiment_settings.pixel_Y)
    class volume_single_peak:
        amplitude   = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        shift       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        width       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        offset      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        error       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        states      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        time        = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        index       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.int32)
        weights     = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num, experiment_settings.freq_num),dtype=np.int32)
    class volume_double_peak:
        l1_amplitude   = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        l1_shift       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        l1_width       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        l2_amplitude   = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        l2_shift       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        l2_width       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        l12_offset      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        error       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        states      = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        time        = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.float32)
        index       = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_Y,experiment_settings.pixel_X + max(experiment_settings.shift)*2),dtype=np.int32)
        double_weights     = np.zeros((experiment_settings.n_planes,experiment_settings.pixel_num, experiment_settings.freq_num),dtype=np.int32)
    double_constraints = np.zeros((experiment_settings.pixel_num,14),dtype=np.float32)
    double_constraints[:,0] = 0.1
    double_constraints[:,1] = 2
    double_constraints[:,2] = 4.8
    double_constraints[:,3] = 5.4
    double_constraints[:,4] = 0.05
    double_constraints[:,5] = 0.9

    double_constraints[:,6] = 0.1
    double_constraints[:,7] = 2
    double_constraints[:,8] = 5.1
    double_constraints[:,9] = 5.6
    double_constraints[:,10] = 0.05
    double_constraints[:,11] = 0.5

    double_constraints[:,12] = 0.1
    double_constraints[:,13] = 2.0

    


    for z in range(experiment_settings.n_planes):
        weights = np.zeros((experiment_settings.pixel_num, experiment_settings.freq_num))
        weights[:,experiment_settings.samples_ignored[0]: experiment_settings.freq_num-experiment_settings.samples_ignored[1]] = 1
        #double_weights = np.zeros((experiment_settings.pixel_num, experiment_settings.freq_num))
        #double_weights[data_X[z,:,:]>(previous_peak_fit.parameters[z,:,1]-1.5*previous_peak_fit.parameters[z,:,2])[:,None]]=1        
        
        double_initial_paras = np.zeros((experiment_settings.pixel_num,7),dtype=np.float32)
        double_initial_paras[:,0] = previous_peak_fit.parameters[z,:,0]/2
        double_initial_paras[:,1] = 5
        double_initial_paras[:,2] = previous_peak_fit.parameters[z,:,2]/2
        double_initial_paras[:,3] = previous_peak_fit.parameters[z,:,0]/2
        double_initial_paras[:,4] = 5.2
        double_initial_paras[:,5] = previous_peak_fit.parameters[z,:,2]/2
        double_initial_paras[:,6] = previous_peak_fit.parameters[z,:,3]/2

        single_peak.parameters[z], single_peak.states[z], single_peak.chi_squares[z], single_peak.number_iterations[z], single_peak.execution_time[z] = gpufit_lorentzian_constrainted(data_X[z], data_Y[z], weights, previous_peak_fit.parameters[z], constraints)      
        double_peak.parameters[z], double_peak.states[z], double_peak.chi_squares[z], double_peak.number_iterations[z], double_peak.execution_time[z] = gpufit_double_lorentzian_constrainted(data_X[z], data_Y[z], weights, double_initial_paras, double_constraints) 

        volume_single_peak.amplitude[z] = convert_to_2d(single_peak.parameters[z,:,0], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_single_peak.shift[z]     = convert_to_2d(single_peak.parameters[z,:,1], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_single_peak.width[z]     = convert_to_2d(single_peak.parameters[z,:,2], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_single_peak.offset[z]    = convert_to_2d(single_peak.parameters[z,:,3], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_single_peak.error[z]     = convert_to_2d(single_peak.chi_squares[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_single_peak.states[z]    = convert_to_2d(single_peak.states[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_single_peak.time[z]      = convert_to_2d(single_peak.execution_time[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_single_peak.index[z]     = convert_to_2d(single_peak.index, experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_single_peak.weights[z]   = weights

        volume_double_peak.l1_amplitude[z] = convert_to_2d(double_peak.parameters[z,:,0], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.l1_shift[z]     = convert_to_2d(double_peak.parameters[z,:,1], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.l1_width[z]     = convert_to_2d(double_peak.parameters[z,:,2], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.l2_amplitude[z] = convert_to_2d(double_peak.parameters[z,:,3], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.l2_shift[z]     = convert_to_2d(double_peak.parameters[z,:,4], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.l2_width[z]     = convert_to_2d(double_peak.parameters[z,:,5], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.l12_offset[z]    = convert_to_2d(double_peak.parameters[z,:,6], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.error[z]     = convert_to_2d(double_peak.chi_squares[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.states[z]    = convert_to_2d(double_peak.states[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.time[z]      = convert_to_2d(double_peak.execution_time[z,:], experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.index[z]     = convert_to_2d(double_peak.index, experiment_settings.pixel_Y,experiment_settings.pixel_X, experiment_settings.shift)
        volume_double_peak.double_weights[z]   = weights
    if z>1:
        volume_single_peak.amplitude[1::2]   = np.rot90(volume_single_peak.amplitude[1::2], 2, (1,2))
        volume_single_peak.shift[1::2]       = np.rot90(volume_single_peak.shift[1::2], 2, (1,2))
        volume_single_peak.width[1::2]       = np.rot90(volume_single_peak.width[1::2], 2, (1,2))
        volume_single_peak.offset[1::2]      = np.rot90(volume_single_peak.offset[1::2], 2, (1,2))
        volume_single_peak.error[1::2]       = np.rot90(volume_single_peak.error[1::2], 2, (1,2))
        volume_single_peak.states[1::2]      = np.rot90(volume_single_peak.states[1::2], 2, (1,2))
        volume_single_peak.index[1::2]       = np.rot90(volume_single_peak.index[1::2], 2, (1,2))

        volume_double_peak.l1_amplitude[1::2]   = np.rot90(volume_double_peak.l1_amplitude[1::2], 2, (1,2))
        volume_double_peak.l1_shift[1::2]       = np.rot90(volume_double_peak.l1_shift[1::2], 2, (1,2))
        volume_double_peak.l1_width[1::2]       = np.rot90(volume_double_peak.l1_width[1::2], 2, (1,2))
        volume_double_peak.l2_amplitude[1::2]   = np.rot90(volume_double_peak.l2_amplitude[1::2], 2, (1,2))
        volume_double_peak.l2_shift[1::2]       = np.rot90(volume_double_peak.l2_shift[1::2], 2, (1,2))
        volume_double_peak.l2_width[1::2]       = np.rot90(volume_double_peak.l2_width[1::2], 2, (1,2))
        volume_double_peak.l12_offset[1::2]      = np.rot90(volume_double_peak.l12_offset[1::2], 2, (1,2))
        volume_double_peak.error[1::2]       = np.rot90(volume_double_peak.error[1::2], 2, (1,2))
        volume_double_peak.states[1::2]      = np.rot90(volume_double_peak.states[1::2], 2, (1,2))
        volume_double_peak.index[1::2]       = np.rot90(volume_double_peak.index[1::2], 2, (1,2))


    return volume_single_peak, volume_double_peak, single_peak, double_peak



def gpufit_lorentzian_constrainted(data_X, data_Y, weights, initial_parameters, constraints):
    data_X_single = data_X.astype(np.float32)
    data_Y_single = data_Y.astype(np.float32)
    weights_single = weights.astype(np.float32)
    number_fits = np.size(data_X,0)
    number_samples = np.size(data_Y,1)
    estimator_id =gf.EstimatorID.LSE
    model_id = gf.ModelID.CAUCHY_LORENTZ_1D
    constraint_types = np.array([ gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER], dtype=np.int32)
    number_parameters = 4
    max_number_iterations = 300
    tolerance = 1e-5
    #print('z')
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data_Y_single, weights_single, model_id, initial_parameters, constraints, constraint_types, tolerance, max_number_iterations, None, estimator_id, data_X_single)
    #print(execution_time)
    return parameters, states, chi_squares, number_iterations, execution_time

def gpufit_double_lorentzian_constrainted(data_X, data_Y, weights, initial_parameters, constraints):
    data_X_single = data_X.astype(np.float32)
    data_Y_single = data_Y.astype(np.float32)
    weights_single = weights.astype(np.float32)
    number_fits = np.size(data_X,0)
    number_samples = np.size(data_Y,1)
    estimator_id =gf.EstimatorID.LSE
    model_id = gf.ModelID.DOUBLE_LORENTZIAN_1D
    constraint_types = np.array([ gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER_UPPER], dtype=np.int32)
    number_parameters = 7
    max_number_iterations = 800
    tolerance = 1e-5
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data_Y_single, weights_single, model_id, initial_parameters, constraints, constraint_types, tolerance, max_number_iterations, None, estimator_id, data_X_single)
    return parameters, states, chi_squares, number_iterations, execution_time

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tifffile import imsave  # 使用 tifffile 库来保存 TIFF 文件

def find_nonzero_bounds(data):
    """找到数据中除了0以外的最小值和最大值。"""
    nonzero_data = data[data != 0]
    if nonzero_data.size > 0:
        min_val = np.min(nonzero_data)
        max_val = np.max(nonzero_data)
    else:
        min_val = np.min(data)
        max_val = np.max(data)
    return min_val, max_val

def normalize_data(data, min_val, max_val):
    """根据给定的上下限归一化数据到0-1范围。"""
    return (data - min_val) / (max_val - min_val)

def generate_colorbar(data, path, index, cmap_option="jet"):
    """
    生成并保存 Colorbar 图片，默认使用指定的色图。
    
    参数:
    data (numpy.ndarray): 数据用于生成 Colorbar 的范围。
    path (str): 文件保存的基本路径。
    index (int): 图像索引。
    cmap_option (str): 色图选项，默认为 'jet'。
    """
    # 找到数据中除了0以外的最小值和最大值
    min_val, max_val = find_nonzero_bounds(data)

    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(4, 0.5))

    # 生成 Colorbar
    norm = plt.Normalize(vmin=min_val, vmax=max_val)
    cmap = plt.get_cmap(cmap_option)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb.ax.set_xlabel('Intensity')

    # 保存 Colorbar 图片
    plt.savefig(f"{path}_{index}_colorbar.png", bbox_inches='tight')
    plt.close(fig)

def process_channel(channel_data, option, path, index):
    """
    处理单个通道的数据并保存为 TIFF 文件，并生成对应的 Colorbar。
    
    参数:
    channel_data (numpy.ndarray): 单个通道的数据。
    option (str): 'gray' 表示保存为灰度图，'jet' 表示转换为 jet 色彩图。
    path (str): 文件保存的基本路径。
    index (int): 图像索引。
    """
    # 保存原始数据的 TIFF 文件
    # 使用 tifffile 库保存为 single 格式
    imsave(f"{path}_{index}_original.tiff", channel_data.astype('float32'))

    # 找到数据中除了0以外的最小值和最大值
    min_val, max_val = find_nonzero_bounds(channel_data)

    if option == 'gray':
        # 归一化数据
        normalized_data = normalize_data(channel_data, min_val, max_val)
        gray_img = Image.fromarray((normalized_data * 255).astype('uint8'), 'L')
        gray_img.save(f"{path}_{index}_gray.tiff")
    elif option == 'jet':
        # 归一化数据
        normalized_data = normalize_data(channel_data, min_val, max_val)
        jet_map = plt.get_cmap('jet')
        jet_rgba = jet_map(normalized_data)
        # 移除 alpha 通道
        jet_img = Image.fromarray((jet_rgba[:, :, :3] * 255).astype('uint8'))
        jet_img.save(f"{path}_{index}_jet.tiff")

        # 生成并保存 Colorbar
        generate_colorbar(channel_data, path, index, cmap_option='jet')
    else:
        raise ValueError("option 必须为 'gray' 或 'jet'")

def save_as_tiff(data, path, option):
    """
    保存数据为 TIFF 文件，并生成对应的 Colorbar。
    
    参数:
    data (numpy.ndarray): 形状为 (300, 400) 或 (300, 400, n) 的数组。
    path (str): 文件保存的基本路径。
    option (str): 'gray' 表示保存为灰度图，'jet' 表示转换为 jet 色彩图。
    """
    # 检查数据维度
    if len(data.shape) == 2:
        # 处理二维数据
        process_channel(data, option, path, 0)
    elif len(data.shape) == 3:
        # 处理三维数据

        for i in range(data.shape[0]):
            channel_data = data[i, :, :]
            process_channel(channel_data, option, path, i)
    else:
        raise ValueError("输入数据的维度必须为 2 或 3")

