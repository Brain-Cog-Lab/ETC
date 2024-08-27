import argparse
import time
import yaml
import numpy as np
np.set_printoptions(threshold=np.inf)

from braincog.base.encoder.encoder import *

import torch
import torch.nn as nn
import torchvision.utils
torch.set_printoptions(threshold=np.inf)


class PEncoder(Encoder):
    def __init__(self, step, encode_type):
        super().__init__(step, encode_type)

    def forward(self, inputs, num_popneurons, *args, **kwargs):
        outputs = self.fun(inputs, num_popneurons, *args, **kwargs)
        return outputs

    @torch.no_grad()
    def population_time(self, inputs, m):
        '''
        inputs:   (N_num, N_feature) array
        m : the number of the gaussian neurons
        i : the i_th gauss_neuron
        one feature will be encoded into gauss_neurons
        the center of i-th neuron is:  gauss -- \mu  u_i = I_min + (2i-3)/2(I_max-I_min)/(m -2)
        the width of i-th neuron is :  gauss -- \sigma sigma_i = 1/1.5(I_max-I_min)/(m -2) 1.5: experience value
        :return: (N_num, N_feature x gauss_neuron) array
        popneurons_spike_t: gauss -- function
        I_min = min(inputs)
        I_max = max(inputs)
        '''
        # m = self.step
        # I_min, I_max = torch.min(inputs), torch.max(inputs)
        mu = [i for i in range(0, m)]
        mu = torch.ones((1, m)) * I_min + ((2 * torch.tensor(mu) - 3) / 2) * ((I_max-I_min) / (m -2))
        sigma = (1 / 1.5) * ((I_max-I_min) / (m -2))
        # shape = (self.step,) + inputs.shape
        shape = (self.step,m)
        popneurons_spike_t = torch.zeros(((m,) + inputs.shape))
        for i in range(m):
            popneurons_spike_t[i, :] = torch.exp(-(inputs - mu[0, i]) ** 2 / (2 * sigma * sigma))

        spike_time = (self.step * popneurons_spike_t).type(torch.int)
        spikes = torch.zeros(shape)
        for spike_time_k in range(self.step):
            if torch.where(spike_time == spike_time_k)[1].numel() != 0:
                spikes[spike_time_k][torch.where(spike_time == spike_time_k)[0]] = 1

        return spikes

    @torch.no_grad()
    def population_voltage(self, inputs, m, VTH):
        '''
        The more similar the input is to the mean,
        the more sensitive the neuron corresponding to the mean is to the input.
        You can change the maen.
        inputs:   (N_num, N_feature) array
        m : the number of the gaussian neurons
        VTH : threshold voltage
        i : the i_th gauss_neuron
        one feature will be encoded into gauss_neurons
        the center of i-th neuron is:  gauss -- \mu  u_i = I_min + (2i-3)/2(I_max-I_min)/(m -2)
        the width of i-th neuron is :  gauss -- \sigma sigma_i = 1/1.5(I_max-I_min)/(m -2) 1.5: experience value
        :return: (N_num, N_feature x gauss_neuron) array
        popneuron_v: gauss -- function
        I_min = min(inputs)
        I_max = max(inputs)
        '''
        ENCODER_REGULAR_VTH = VTH
        I_min, I_max = torch.min(inputs), torch.max(inputs)
        mu = [i for i in range(0, m)]
        mu = torch.ones((1, m)) * I_min + ((2 * torch.tensor(mu) - 3) / 2) * ((I_max-I_min) / (m -2))
        sigma = (1 / 1.5) * ((I_max-I_min) / (m -2))
        popneuron_v = torch.zeros(((m,) + inputs.shape))
        delta_v = torch.zeros(((m,) + inputs.shape))
        for i in range(m):
            delta_v[i] = torch.exp(-(inputs - mu[0, i]) ** 2 / (2 * sigma * sigma))
        spikes = torch.zeros((self.step,) + ((m,) + inputs.shape))
        for spike_time_k in range(self.step):
            popneuron_v = popneuron_v + delta_v
            spikes[spike_time_k][torch.where(popneuron_v.ge(ENCODER_REGULAR_VTH))] = 1
            popneuron_v = popneuron_v - spikes[spike_time_k] * ENCODER_REGULAR_VTH

        popneuron_rate = torch.sum(spikes, dim=0)/self.step

        return spikes, popneuron_rate


if __name__ == '__main__':
    a = (torch.rand((2,3,4))*10).type(torch.int)
    print(a)
    pencoder = PEncoder(10, 'population_time')
    spikes=pencoder(inputs=a, num_popneurons=3)
    print(a[0,0,0],'++++++++++++++++++++++++++++++')
    print(spikes[:,0,0,0])
    print(a[1, 0, 0], '++++++++++++++++++++++++++++++')
    print(spikes[:, 1, 0, 0])
    print(a[0, 1, 0], '++++++++++++++++++++++++++++++')
    print(spikes[:, 0, 1, 0])
    # b = (torch.tensor([2,3,4])).type(torch.int)
    # print(b)
    # pencoder = PEncoder(10, 'population_voltage')
    # spikes, popneuron_rate = pencoder(inputs=b, num_popneurons=5, VTH=0.99)
    # print(spikes[:, 0, 0], 'pop0-x0')
    # print(spikes[:, 1, 0], 'pop1-x0')
    # print(spikes[:, 2, 0], 'pop2-x0')
    # print(spikes[:, 3, 0], 'pop3-x0')
    # print(spikes[:, 4, 0], 'pop4-x0')
    # print(popneuron_rate.shape)
    # print(popneuron_rate[0, 0], 'pop0-x0')
    # print(popneuron_rate[1, 0], 'pop1-x0')
    # print(popneuron_rate[2, 0], 'pop2-x0')