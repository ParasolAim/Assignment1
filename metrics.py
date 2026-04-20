import numpy as np 
 
def snr_db(x, y, eps=1e-12): 
    """ 
    Signal-to-noise ratio in dB. 
    x: original 
    y: reconstructed 
    """ 
    x = np.asarray(x) 
    y = np.asarray(y) 
    noise = x - y 
    p_signal = np.mean(x ** 2) 
    p_noise = np.mean(noise ** 2) + eps 
    return 10 * np.log10((p_signal + eps) / p_noise) 
 
def mse(x, y): 
    x = np.asarray(x) 
    y = np.asarray(y) 
    return np.mean((x - y) ** 2) 
 
def compression_ratio(original_num_bits, compressed_num_bits): 
    return original_num_bits / compressed_num_bits