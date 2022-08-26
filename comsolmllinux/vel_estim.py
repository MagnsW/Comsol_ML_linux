# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:35:58 2021

@author: Magnus
"""

import numpy as np
import scipy.interpolate
#import scipy
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from .lamb import Lamb
#import pandas as pd

def define_windows(df, window1_start, window2_start, window_length=130):
    df_s1 = df.iloc[window1_start:window1_start+window_length]
    df_s2 = df.iloc[window2_start:window2_start+window_length]
    return df_s1, df_s2

def compute_delt(window1_start, window2_start, fs):
    delt = (window2_start - window1_start)/fs
    return delt

def return_center_freqs(df):
    fc = np.array([])
    for item in list(df.columns):
        fc = np.append(fc, [int(item[:3])*1000])
        #fc = np.append(fc, [int(item[:3])])
    return fc

def make_lamb_curves(E=205e9, p=7850, v=0.28, d=6.8):
    c_L = np.sqrt(E*(1-v) / (p*(1+v)*(1-2*v)))
    c_S = np.sqrt(E / (2*p*(1+v)))
    c_R = c_S * ((0.862+1.14*v) / (1+v))

    steel = Lamb(thickness=d, 
            nmodes_sym=5, 
            nmodes_antisym=5, 
            fd_max=10000, 
            vp_max=15000, 
            c_L=c_L, 
            c_S=c_S, 
            c_R=c_R, 
            material='Steel')
    return steel

def phase_velocity_estimation(df_s1, df_s2, delt, delz, fs, fc, wm_interpolator, d):
    #Frequency scan case
    L = 1000
    Sw1 = fft(df_s1, n=L, axis=0)
    Sw2 = fft(df_s2, n=L, axis=0)
    f = fs*np.arange(0, L, 1)/L
    
    infc = np.array([], dtype='int64')
    for fci in fc:
        infc = np.append(infc, np.where(f==fci)[0][0])
        
    w = 2*np.pi*f
    
    ang = np.unwrap(np.angle(Sw2/Sw1), axis=0)
    vphi = delz/(delt-(ang.T/w).T)
    
    plt.figure(figsize=(12, 8))
    plt.plot(f/1000, vphi)
    plt.title('Uncorrected phase velocity spectrums')
    plt.xlabel('kHz')
    plt.ylabel('Velocity')
    plt.grid(which='both')
    plt.show()
    
    vphi_fc = np.array([])
    for i, infc_item in enumerate(infc):
        vphi_fc = np.append(vphi_fc, [vphi[infc_item, i]])
        
    
    vfap_fc = wm_interpolator(fc/1000*d)
    
    plt.figure(figsize=(12, 8))
    plt.plot(fc/1000, vphi_fc)
    plt.plot(fc/1000, vfap_fc)
    plt.grid(which='both')
    plt.legend(['Uncorrected phase velocity', 'Theoretical dispersion curve'])
    plt.xlabel('kHz')
    plt.ylabel('Velocity')
    plt.show()
    
    #Pi skip correction
    ang_apri = np.array([])
    #ang_corrected = np.array([])
    for k in range(len(fc)):
        ang_apri = np.append(ang_apri, 2*np.pi*fc[k]*(delt - delz/vfap_fc[k]))
        if np.abs(ang[infc[k], k] - ang_apri[k]) > np.pi:
            #ang_correction = np.round((ang_apri[k] - ang[infc[k], k])/2/np.pi)*2*np.pi
            ang[:,k] = np.round((ang_apri[k] - ang[infc[k], k])/2/np.pi)*2*np.pi + ang[:,k]
            #ang_corrected = np.append(ang_corrected, np.round((ang_apri[k] - ang[infc[k], k])/2/np.pi) + ang[k])
            #print(k, infc[k], ang[60,k], ang_uncorrected[60,k], ang_correction)
    vphi = delz/(delt-(ang.T/w).T)
    
    vphi_fc = np.array([])
    for i, infc_item in enumerate(infc):
        vphi_fc = np.append(vphi_fc, [vphi[infc_item, i]])
        
    plt.figure(figsize=(12, 8))
    plt.plot(f/1000, vphi)
    plt.title('Corrected phase velocity spectrums')
    plt.xlabel('kHz')
    plt.ylabel('Velocity')
    plt.grid(which='both')
    plt.show()
        
    plt.figure(figsize=(12, 8))
    plt.plot(fc/1000, vphi_fc)
    plt.plot(fc/1000, vfap_fc)
    plt.grid(which='both')
    plt.legend(['Corrected phase velocity', 'Theoretical dispersion curve'])
    plt.xlabel('kHz')
    plt.ylabel('Velocity')
    plt.show()
    return vphi_fc, fc/1000
            
def do_aliasing(curve, kNyq):
    aliased = True
    while aliased:
        curve = np.where(curve < kNyq, curve, curve - 2*kNyq)
        curve = np.where(curve > -kNyq, curve, curve + 2*kNyq)
        if (np.max(curve) < kNyq) & (np.min(curve) > -kNyq):
            aliased = False
    return curve

def FK_thickness_estimation(fk, k_array, f_array, dx, disp_curve_interpolator, ds, minfreq, maxfreq, plot=True, d_true=None):
    '''

    :param fk: FK domain data in absolute amplitude, 2D array. Linear or dB scale.
    :param k_array: Wavenumber array associated with FK
    :param f_array: Frequency array associated with FK
    :param dx: Receiver sampling
    :param disp_curve_interpolator: Dispersion curve interpolator
    :param ds: Array of investigated thicknesses
    :param minfreq: Minimum frequency for analysis
    :param maxfreq: Maximum frequency for analysis
    :return:
    '''
    kNyq = 1/dx/2 #Nyquist wavenumber calculated from dx

    f_array_dispersion = f_array[f_array >= minfreq]
    f_array_dispersion = f_array_dispersion[f_array_dispersion <= maxfreq]

    disp_curves = {}
    for d in ds:
        disp_curves[round(d, 3)] = 1/np.divide(disp_curve_interpolator(f_array_dispersion/1000*d), f_array_dispersion)


    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        plt.pcolormesh(k_array, f_array/1000, fk, cmap='jet')
        plt.ylim(0, maxfreq/1000)
        plt.title('FK plot')
        plt.xlabel('Wavenumber')
        plt.ylabel('Frequency (kHz)')
        plt.subplot(1,2,2)
        plt.pcolormesh(k_array, f_array/1000, fk, cmap='jet')
        for curve in disp_curves:
            plt.plot(do_aliasing(disp_curves[curve], kNyq), f_array_dispersion / 1000, ".", alpha=0.5)
            plt.plot(do_aliasing(-disp_curves[curve], kNyq), f_array_dispersion / 1000, ".", alpha=0.5)
        plt.ylim(0, maxfreq/1000)
        plt.xlabel('Wavenumber')
        #plt.ylabel('Frequency (kHz)')
        plt.title('FK plot with evaluated dispersion curves')
        # plt.colorbar()
        #plt.tight_layout()

    interpolator2d = scipy.interpolate.RectBivariateSpline(f_array, k_array, fk)

    curves_sum = {}
    for curve in disp_curves:
        curves_sum[curve] = 0
        for i in range(len(f_array_dispersion)):
            curves_sum[curve] += interpolator2d(f_array_dispersion[i], do_aliasing(disp_curves[curve], kNyq)[i])[0][0]
            curves_sum[curve] += interpolator2d(f_array_dispersion[i], do_aliasing(-disp_curves[curve], kNyq)[i])[0][0]

    d_estimate = max(curves_sum, key=curves_sum.get)

    if plot:
        plt.figure(figsize=(12, 6))
        # sns.lineplot(data=df_curves_sum)
        plt.plot(curves_sum.keys(), curves_sum.values(), 'o-', label='Dispersion curve "Stacking power"')
        plt.axvline(x=d_estimate, color='g', label='Estimated thickness: ' + str(d_estimate) + ' mm')
        if d_true:
            plt.axvline(x=d_true, color='r', label='True thickness: ' + str(d_true) + ' mm')
        plt.grid(visible=True, which='major')
        plt.grid(visible=True, which='minor', alpha=0.5)
        plt.minorticks_on()
        plt.xticks(ticks=ds[::5], rotation=45)
        plt.xlabel('Dispersion curve plate thickness (mm)')
        plt.ylabel('Stacking amplitude')

        plt.legend()
        plt.show()

    return d_estimate