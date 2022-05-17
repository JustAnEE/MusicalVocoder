import numpy as np
from scipy.signal import butter, lfilter, sosfilt, hilbert
from scipy.io import wavfile


# -- HPF for optional output
def hpf_params(f_c, f_s, order = 3):

    b, a = butter(order, f_c, btype='high', analog=False, fs = f_s)

    return b, a


# -- Optional Function: Pink Noise
def pink_noise(sig, num):

    sig_pink = []

    return sig_pink


# -- Optional Function: White Noise
def white_noise(sig, num):

    noise = np.random.normal(0.2, 0.7, size=num)

    sig_white = sig + 0.15*noise

    return sig_white


# -- Optional Function: Sawtooth Carrier
def sawtooth():

    saw_carrier = []

    return saw_carrier


# -- Get second order cascade of butterworth bandpass filter for each 1/3rd octave band.
def get_sos(fl, fr, N=10):
    
    sos = butter(N, [fl, fr], btype='bandpass', analog=False, output='sos')
    
    return sos 


# -- Get the band edges for each 1/3rd octave band 
def get_band_edges(fc0, M):
    
    f_l = np.zeros(M+1)
    f_r = np.zeros(M+1)
    
    for k in range(0, M):
        
        f_l[k] = fc0*(2**((2*k-1)/6))
        f_r[k] = fc0*(2**((2*k+1)/6))
    
    return f_l, f_r 


# -- Obtain the filter bank coefficients for all 1/3 octave bands.
def octave_bank(fc0, M, fs):
    
    f_N = 0.5*fs
    
    # -- Get band edges for each third octave band 
    f_l, f_r = get_band_edges(fc0, M)
    
    # -- Normalize wrt to Nyquist frequency
    f_l = f_l/f_N
    f_r = f_r/f_N
    
    H = np.zeros((M, 60))
    
    for k in range(0, M):
        
        sos = get_sos(f_l[k], f_r[k])
        
        H[k, :] = sos.flatten()
        
    return H


# -- Signal Subbands
def get_subbands(sig, H, shape, L, M):
    
    sub_bands = np.zeros((M, L))
    
    for k in range(0, M):
        
        h = H[k, :].reshape(shape)
        
        sub_bands[k, :] = sosfilt(h, sig)

    return sub_bands


# -- Hilbert Envelopes 
def hilbert_envelopes(m_s, M, L):
    
    h_env = np.zeros((M, L))
    
    for k in range(0, M):
        
        # -- Note: hilbert returns the analytic signal itself
        h_env[k, :] = np.abs(hilbert(m_s[k,:]))

    return h_env


# -- Vocoded output 
def vocode(envelopes, m_c, M, L):
    
    vocoded_sig = np.zeros((1, L))
    
    for k in range(0, M):
        
        vocoded_sig += envelopes[k, :]*m_c[k, :]
    
    return 150*vocoded_sig


# -- Main Script 

M = 32  # Number of filters in the bank
N = 10  # Filter bank BPF order
fc0 = 12.5
fs = 44100

sos_shape = (N, 6)

H_bank = octave_bank(fc0, M, fs)

f_sc, carrier = wavfile.read('synth.wav')
f_sm, modulator = wavfile.read('counting.wav')

# -- Use smallest signal length
if len(modulator) < len(carrier):
    
    L = len(modulator)
    
if len(carrier) < len(modulator):
    
    L = len(carrier)

# -- Configure Noise Carrier Options
carrier_shape = np.shape(carrier)
noised_carrier = input('Enter: No Noise = 0, White Noise = 1, Pink Noise = 2 \n')
print('----------------------- \n')

if noised_carrier == '1':
    carrier = white_noise(carrier, carrier_shape)

if noised_carrier == '2':
    carrier = pink_noise(carrier, carrier_shape)

mod_subs = get_subbands(modulator[:L], H_bank, sos_shape, L, M)
carrier_subs = get_subbands(carrier[:L], H_bank, sos_shape, L, M)

env = hilbert_envelopes(mod_subs, M, L)

vocoded = vocode(env, carrier_subs, M, L)

HPF = input('High Pass Output? No = 0, Yes = 1 \n')
print('----------------------- \n')

if HPF == '1':
    
    b, a = hpf_params(250, f_sc)
    vocoded = lfilter(b, a, vocoded)

non_zero_vocoded = vocoded[np.where(vocoded != 0)]
non_zero_vocoded = non_zero_vocoded.astype('float32')

wavfile.write('vocoded.wav', f_sc, non_zero_vocoded)
