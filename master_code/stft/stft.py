import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from hparams import *

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class STFT(torch.nn.Module, metaclass=Singleton):
    def __init__(self, filter_length=256, hop_length=128):
        super(STFT, self).__init__()

        self.filter_length = filter_length
        self.hop_length = hop_length
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data): #input shape (1,16000) (B, len)
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        input_data = input_data.view(num_batches, 1, num_samples) # respahe to (B, 1, 16000)
        forward_transform = F.conv1d(input_data,
                                     Variable(self.forward_basis, requires_grad=False),
                                     stride = self.hop_length,
                                     padding = self.filter_length)  # (B, 258, 128)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :] # (B, 129, 128)
        imag_part = forward_transform[:, cutoff:, :]  # (B, 129, 128)

        return real_part, imag_part

    # def inverse(self, magnitude, phase):
    # def inverse(self, real_part, imag_part):
    def inverse(self, stft_audio): # ()
        # recombine_magnitude_phase = torch.cat([magnitude*torch.cos(phase),
        #                                        magnitude*torch.sin(phase)], dim=1)

        # recombine_real_img = torch.cat([real_part, imag_part], dim=1)

        # inverse_transform = F.conv_transpose1d(recombine_magnitude_phase,
        #                                        Variable(self.inverse_basis, requires_grad=False),
        #                                        stride=self.hop_length,
        #                                        padding=0)


        new_real_part = stft_audio[:,0,:,:].squeeze(1)
        new_imag_part = stft_audio[:,1,:,:].squeeze(1)

        new_fft = torch.cat((new_real_part, new_imag_part),dim =1)




        inverse_transform = F.conv_transpose1d(new_fft,
                                               Variable(self.inverse_basis, requires_grad=False),
                                               stride=self.hop_length,
                                               padding=0)

        inverse_transform = inverse_transform[:, :, self.filter_length:]
        inverse_transform = inverse_transform[:, :, :self.num_samples]


        inverse_transform = inverse_transform.squeeze(1)
        return inverse_transform