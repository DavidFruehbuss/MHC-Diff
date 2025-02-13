import numpy as np
import torch
import torch.nn as nn

class Noise_Schedule(nn.Module):
    '''
    Defines a polynomial noise schedule (power=2)
    This is precomputed as a lookup table to save time
    '''

    def __init__(self, num_timesteps, offset=1.0e-5, noise_schedule='polynomial'):
        super().__init__()
        self.T = num_timesteps

        if noise_schedule == 'polynomial':    
            x = np.linspace(0, num_timesteps + 1, num_timesteps + 1)
            alpha2 = (1 - (x/num_timesteps)**2)**2
        elif noise_schedule == 'linear':
            x = np.linspace(0, num_timesteps + 1, num_timesteps + 1)
            alpha2 = (1 - (x/num_timesteps)**1)**2
        else:
            raise NotImplementedError

        # for numerical stability and offset for avoiding problems with t = 0
        alpha2 = self.clip_noise_schedule(alpha2)
        alpha2 = (1 - 2 * offset) * alpha2 + offset

        alpha = np.sqrt(alpha2)
        sigma = np.sqrt(1-alpha2)
        
        self.alpha = nn.Parameter(torch.from_numpy(alpha).float(), requires_grad=False)
        self.sigma = nn.Parameter(torch.from_numpy(sigma).float(), requires_grad=False)

    def forward(self, t, type):

        t_unnormalized = torch.round(t * self.T).long()

        if type == 'alpha':
            return self.alpha[t_unnormalized]
        else:
            return self.sigma[t_unnormalized]

    def clip_noise_schedule(self, alphas2, clip_value=0.001):

        """
        For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
        This may help improve stability during sampling.
        """

        alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

        alphas_step = (alphas2[1:] / alphas2[:-1])

        alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
        alphas2 = np.cumprod(alphas_step, axis=0)

        return alphas2