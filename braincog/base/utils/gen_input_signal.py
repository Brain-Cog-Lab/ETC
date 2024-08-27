import numpy as np
import random
import copy

dt = 1.0                  # ms
lambda_max = 0.25*dt # maximum spike rate (spikes per time step)
eps_ = 1e-6

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def recp_signal(raw_sig, rows, cols, padding, stride=1):
    sigs = []
    for r in range(padding[0], raw_sig.shape[0] - padding[0], stride):
        for c in range(padding[1], raw_sig.shape[1] - padding[1], stride):
            x_l = r - padding[0]
            x_r = r + padding[0] +1
            y_l = c - padding[1] 
            y_r = c + padding[1] +1
            sigs.append(raw_sig[x_l:x_r, y_l:y_r].flatten())
    sigs = np.array(sigs).T
    return sigs

def form_input_dog(image, recp, steps, stride=1):
    padding  = recp
    width = int(np.sqrt(image.shape[0]))
    image_reshape = np.reshape(image, (width, width))
    img_pad = np.pad(image_reshape, padding, pad_with, padder=0)
    rows = int((recp[0]*2+1) **2)
    cols = image.shape[0] // stride**2
    signal = np.zeros((steps, rows, cols))
    for i in range(steps):
        signal[i] = recp_signal(img_pad, rows, cols, padding)        
    return signal

def form_input_gabor(input_signal, recp, stride=1):
    padding = recp
    width = int(np.sqrt(input_signal.shape[0]))
    input_signal_reshape =  np.reshape(input_signal, (width, width))
    signal_padding = np.pad(input_signal_reshape, padding, pad_with, padder=0)
    rows = int((recp[0]*2+1) **2)
    cols = input_signal.shape[0] // stride**2
    signal = recp_signal(signal_padding, rows, cols, padding, stride)
    return signal

def regul_image(image, reverse=False):
    img = image.copy()
    if reverse:
        img[img>0] = 1
        img[img==0] = 255
    else:
        img[img>0] = 255
    return img

    
def img2spikes(image, 
               image_delta, 
               image_ori, 
               image_ori_delta,
               steps, 
               sig_len, 
               shift=None, 
               noise=None,
               noise_rate=None):

    signal = np.zeros((steps, image.shape[0]))
    if noise:
        assert image_ori is not None
        assert shift is False
        assert noise_rate is not None
        image_ori_delta = copy.deepcopy(image_ori)
        idx = image_ori_delta < (lambda_max - 0.001)
        image_ori_delta[idx] += 0.001
        image_ori_reverse = lambda_max - image_ori
        image_ori_delta_reverse = lambda_max - image_ori_delta
        image_noise, image_delta_noise = reverse_pixels(image_ori, image_ori_delta, noise_rate=noise_rate)
        zeta = image_noise / (image_ori**2 + image_ori_reverse**2)**0.5 
        zeta_delta = image_delta_noise/ (image_ori_delta**2 + image_ori_delta_reverse**2)**0.5
        idx_left = zeta < zeta_delta
        phi = np.arctan(image_ori / (image_ori_reverse + eps_))
        zeta = np.clip(zeta, -1, 1)
        zeta = np.arcsin(zeta)
        theta1 = zeta - phi
        theta2 = np.pi - zeta - phi
        theta = np.zeros(theta1.shape)
        theta[idx_left] = theta1[idx_left]
        theta[~idx_left] = theta2[~idx_left]
        theta = np.mean(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        spike_rate = np.abs((lambda_max * sin_theta - image_noise) / (sin_theta - cos_theta + eps_))
        signal_possion = np.random.poisson(spike_rate, (sig_len, spike_rate.shape[0]))
        shift_step = np.rint(np.clip(2*theta/np.pi, a_min=0, a_max=1.0) * (steps - sig_len))
        shift_step = shift_step.astype(np.int)
        signal[shift_step:shift_step+sig_len] = signal_possion[:]
    
    elif shift:
        assert image_ori is not None
        assert noise is False
        assert image_delta is not None
        assert image_ori_delta is not None
        image_ori_reverse = lambda_max - image_ori
        image_ori_delta_reverse = lambda_max - image_ori_delta
        zeta = image / (image_ori**2 + image_ori_reverse**2) ** 0.5
        zeta_delta = image_delta / (image_ori_delta**2 + image_ori_delta_reverse**2)**0.5
        idx_left = zeta < zeta_delta
        phi = np.arctan(image_ori / (image_ori_reverse + eps_))
        zeta = np.clip(zeta, -1, 1)
        zeta = np.arcsin(zeta)
        theta1 = zeta - phi
        theta2 = np.pi - zeta - phi
        theta = np.zeros(theta1.shape)
        theta[idx_left] = theta1[idx_left]
        theta[~idx_left] = theta2[~idx_left]
        theta = np.mean(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        spike_rate = np.abs((lambda_max * sin_theta - image) / (sin_theta - cos_theta + eps_))
        signal_possion = np.random.poisson(spike_rate, (sig_len, spike_rate.shape[0]))
        shift_step = np.rint(np.clip(2*theta/np.pi, a_min=0, a_max=1.0) * (steps - sig_len))
        shift_step = shift_step.astype(np.int)
        signal[shift_step:shift_step+sig_len] = signal_possion[:]
      
    else:
        signal_possion = np.random.poisson(image, (sig_len, image.shape[0]))
        signal[:sig_len] = signal_possion[:]
    return signal.T

def reverse_pixels(image, image_delta, noise_rate, flip_bits=None):
    if flip_bits is None:
        N = int(noise_rate * image.shape[0])
        flip_bits = random.sample(range(image.shape[0]), N)
        img = copy.copy(image)
        img_delta = copy.copy(image_delta)
    
        img[flip_bits] = lambda_max - img[flip_bits]
        img_delta[flip_bits] = lambda_max - img_delta[flip_bits]
    return img, img_delta
    

def gaussian_noise(img, mean, std):
    img = img / lambda_max
    noise = np.random.normal(mean, std, img.shape)
    gaussian_out = np.clip(img + noise, 0, 1)
    gaussian_out *= lambda_max
    return gaussian_out
