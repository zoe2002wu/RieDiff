# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License for
# CLD-SGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from util.utils import add_dimensions
import os

class CLD(nn.Module):
    def __init__(self, config, beta_fn, beta_int_fn):
        super().__init__()
        self.config = config
        self.beta_fn = beta_fn
        self.beta_int_fn = beta_int_fn
        self.m_inv = config.m_inv
        self.f = 2. / np.sqrt(config.m_inv)
        self.g = 1. / self.f
        self.gamma = config.gamma
        self.numerical_eps = config.numerical_eps
        self.riemann_mix = config.riemann_mix
        self.image_size = config.image_size
        self.image_channels = config.image_channels
        self.beta = config.beta0
        self.Gamma = config.Gamma
        self.geometry = config.geometry

        # Load normalization once but allow updating
        self.normalization = 1



    @property
    def type(self):
        return 'cld'

    @property
    def is_augmented(self):
        return True

    def sde(self, u, t):
        '''
        Evaluating drift and diffusion of the SDE.
        '''
        x, r = torch.chunk(u, 2, dim=1)

        beta = add_dimensions(self.beta_fn(t), self.config.is_image)

        drift_x = self.m_inv * beta * r
        drift_r = -beta * x - self.f * self.m_inv * beta * r

        diffusion_x = torch.zeros_like(x)
        diffusion_r = torch.sqrt(2. * self.f * beta) * torch.ones_like(r)

        return torch.cat((drift_x, drift_r), dim=1), torch.cat((diffusion_x, diffusion_r), dim=1)
    
        
    def sde_reverse(self, u, t, score_x= None):
        '''
        Evaluating drift and diffusion of the SDE.
        '''
        if self.geometry == "Riemann":
            x, r = torch.chunk(u, 2, dim=1)

            G, G_inv, G_sqrt = self.compute_G(score_x, t)

            beta_coeff = self.beta * torch.sqrt(torch.tensor(self.m_inv))
            Gamma_coeff = self.Gamma * torch.sqrt(torch.tensor(self.m_inv))

            beta = beta_coeff * G_sqrt

            hamilton_x_riemann = x 
            hamilton_r_riemann = self.matrix_mult(G_inv, r)
            
            beta_gamma = torch.sqrt((2 * beta_coeff * Gamma_coeff).clone().detach()) * G_sqrt

            drift_x = self.matrix_mult(beta, hamilton_r_riemann) #x portion of f(u, t)
            drift_r = -self.matrix_mult(beta, hamilton_x_riemann) - self.matrix_mult(beta_coeff* Gamma_coeff*G,  hamilton_r_riemann) #r portion of f(u, t)

            diffusion_x = torch.zeros_like(x) #x portion of g(t)
            diffusion_r = self.matrix_mult(beta_gamma, torch.ones_like(r))  #r portion of g(t)

            return torch.cat((drift_x, drift_r), dim=1), torch.cat((diffusion_x, diffusion_r), dim=1)
        else:
            x, r = torch.chunk(u, 2, dim=1)

            beta = add_dimensions(self.beta_fn(t), self.config.is_image)

            drift_x = self.m_inv * beta * r
            drift_r = -beta * x - self.f * self.m_inv * beta * r

            diffusion_x = torch.zeros_like(x)
            diffusion_r = torch.sqrt(2. * self.f * beta) * torch.ones_like(r)

            return torch.cat((drift_x, drift_r), dim=1), torch.cat((diffusion_x, diffusion_r), dim=1)

    
    def get_reverse_sde(self, score_fn=None, probability_flow=False):
        sde_fn = self.sde_reverse

        def reverse_sde(u, t, score=None):
            '''
            Evaluating drift and diffusion of the ReverseSDE.
            '''
            score = score if score is not None else score_fn(u, 1. - t)

            if self.geometry == "Riemann":
                epsilon_x , score_r = torch.chunk(score, 2, dim=1)
                score_x = self.compute_score_x(t, epsilon_x, score_r)

                drift, diffusion = sde_fn(u, 1. - t, score_x)

            else: 
                drift, diffusion = sde_fn(u, 1. - t)
                score_r = score

            drift_x, drift_r = torch.chunk(drift, 2, dim=1)
            _, diffusion_r = torch.chunk(diffusion, 2, dim=1)

            reverse_drift_x = -drift_x
            reverse_drift_r = -drift_r + diffusion_r ** 2. * \
                score_r * (0.5 if probability_flow else 1.)

            reverse_diffusion_x = torch.zeros_like(diffusion_r)
            reverse_diffusion_r = torch.zeros_like(
                diffusion_r) if probability_flow else diffusion_r

            return torch.cat((reverse_drift_x, reverse_drift_r), dim=1), torch.cat((reverse_diffusion_x, reverse_diffusion_r), dim=1)

        return reverse_sde

    def compute_score_x(self, t, epsilon_x, score_r):

        noise_multiplier = self.noise_multiplier(t)
        M_inv = self.m_inv
        gamma = self.gamma

        # Compute covariance components
        ones = torch.ones_like(epsilon_x, device=epsilon_x.device)  # dimension 3
        sigma_xx, sigma_xr, _ = self.var(t, 0. * ones, (gamma / M_inv) * ones)

        epsilon_r = score_r/ noise_multiplier

        score_x = (-epsilon_x / torch.sqrt(sigma_xx)) - noise_multiplier*sigma_xr*epsilon_r / sigma_xx

        return score_x

    def compute_G(self,score_x, t):

        s_ = score_x.permute(0, 2, 3, 1)  # Shape: (batch_size, 32, 32, 3)
        outer = s_.unsqueeze(-1) @ s_.unsqueeze(-2)  # Shape: (batch_size, 32, 32, 3, 3)
        G = outer.mean(dim=0)  # Averaging over batch (reduces gradient strength)
        
        if t[0].item() == 1.0:
            G_diagonal = torch.diagonal(G, dim1=-2, dim2=-1)
            self.normalization = G_diagonal.mean().item()*self.m_inv 

        G = G/self.normalization

        t = t[0].item()
        k = 4.5 # tunes how quickly it goes to G_id
        start = 1 / self.config.m_inv 
        G_id  = start * torch.eye(self.image_channels, device=G.device).reshape(1,1,3,3).expand(self.image_size,self.image_size,self.image_channels,self.image_channels)
        alpha = self.riemann_mix * torch.exp(torch.tensor(-k * (1-t), device = G.device))  

        G = alpha * G + (1 - alpha) * G_id

        G_inv = torch.linalg.inv(G).to(torch.float32)

        G_sqrt = torch.linalg.cholesky(G)
        G_sqrt = G_sqrt.to(torch.float32)
        return G, G_inv, G_sqrt
    

    def matrix_mult(self,A, B):
        # A is 32 x 32 x 3 x 3
        # B is batch-size x 3 x 32 x 32
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        output = (A @ B.permute(0,2,3,1).unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)            
        return output # shape batch-size x 3 x 32 x 32

    def prior_sampling(self, shape):
        return torch.randn(*shape, device=self.config.device), torch.randn(*shape, device=self.config.device) / np.sqrt(self.m_inv)

    def prior_logp(self, u):
        x, v = torch.chunk(u, 2, dim=1)
        N = np.prod(x.shape[1:])

        logx = -N / 2. * np.log(2. * np.pi) - \
            torch.sum(x.view(x.shape[0], -1) ** 2., dim=1) / 2.
        logv = -N / 2. * np.log(2. * np.pi / self.m_inv) - torch.sum(
            v.view(v.shape[0], -1) ** 2., dim=1) * self.m_inv / 2.
        return logx, logv

    def mean(self, u, t):
        '''
        Evaluating the mean of the conditional perturbation kernel.
        '''
        x, v = torch.chunk(u, 2, dim=1)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
        coeff_mean = torch.exp(-2. * beta_int * self.g)

        mean_x = coeff_mean * (2. * beta_int * self.g *
                               x + 4. * beta_int * self.g ** 2. * v + x)
        mean_v = coeff_mean * (-beta_int * x - 2. * beta_int * self.g * v + v)
        return torch.cat((mean_x, mean_v), dim=1)

    def var(self, t, var0x=None, var0v=None):
        '''
        Evaluating the variance of the conditional perturbation kernel.
        '''
        if var0x is None:
            var0x = add_dimensions(torch.zeros_like(
                t, dtype=torch.float64, device=t.device), self.config.is_image)
        if var0v is None:
            if self.config.cld_objective == 'dsm':
                var0v = torch.zeros_like(
                    t, dtype=torch.float64, device=t.device)
            elif self.config.cld_objective == 'hsm':
                var0v = (self.gamma / self.m_inv) * torch.ones_like(t,
                                                                    dtype=torch.float64, device=t.device)

            var0v = add_dimensions(var0v, self.config.is_image)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
        multiplier = torch.exp(-4. * beta_int * self.g)

        var_xx = var0x + (1. / multiplier) - 1. + 4. * beta_int * self.g * (var0x - 1.) + 4. * \
            beta_int ** 2. * self.g ** 2. * \
            (var0x - 2.) + 16. * self.g ** 4. * beta_int ** 2. * var0v
        var_xv = -var0x * beta_int + 4. * self.g ** 2. * beta_int * var0v - 2. * self.g * \
            beta_int ** 2. * (var0x - 2.) - 8. * \
            self.g ** 3. * beta_int ** 2. * var0v
        var_vv = self.f ** 2. * ((1. / multiplier) - 1.) / 4. + self.f * beta_int - 4. * self.g * beta_int * \
            var0v + 4. * self.g ** 2. * beta_int ** 2. * \
            var0v + var0v + beta_int ** 2. * (var0x - 2.)
        return [var_xx * multiplier + self.numerical_eps, var_xv * multiplier, var_vv * multiplier + self.numerical_eps]

    def mean_and_var(self, u, t, var0x=None, var0v=None):
        return self.mean(u, t), self.var(t, var0x, var0v)

    def noise_multiplier(self, t, var0x=None, var0v=None):
        '''
        Evaluating the -\ell_t multiplier. Similar to -1/standard deviaton in VPSDE.
        '''
        var = self.var(t, var0x, var0v)
        coeff = torch.sqrt(var[0] / (var[0] * var[2] - var[1]**2))

        if torch.sum(torch.isnan(coeff)) > 0:
            raise ValueError('Numerical precision error.')

        return -coeff

    def loss_multiplier(self, t):
        '''
        Evaluating the "maximum likelihood" multiplier.
        '''
        return self.beta_fn(t) * self.f

    def perturb_data(self, batch, t, var0x=None, var0v=None):
        '''
        Perturbing data according to conditional perturbation kernel with initial variances
        var0x and var0v. Var0x is generally always 0, whereas var0v is 0 for DSM and 
        \gamma * M for HSM.
        '''
        mean, var = self.mean_and_var(batch, t, var0x, var0v)

        cholesky11 = (torch.sqrt(var[0]))
        cholesky21 = (var[1] / cholesky11)
        cholesky22 = (torch.sqrt(var[2] - cholesky21 ** 2.))

        if torch.sum(torch.isnan(cholesky11)) > 0 or torch.sum(torch.isnan(cholesky21)) > 0 or torch.sum(torch.isnan(cholesky22)) > 0:
            raise ValueError('Numerical precision error.')

        batch_randn = torch.randn_like(batch, device=batch.device)
        batch_randn_x, batch_randn_v = torch.chunk(batch_randn, 2, dim=1)

        noise_x = cholesky11 * batch_randn_x
        noise_v = cholesky21 * batch_randn_x + cholesky22 * batch_randn_v
        noise = torch.cat((noise_x, noise_v), dim=1)

        perturbed_data = mean + noise
        
        return perturbed_data, mean, noise, batch_randn

    def get_discrete_step_fn(self, mode, score_fn=None, probability_flow=False):
        if mode == 'forward':
            sde_fn = self.sde
        elif mode == 'reverse':
            sde_fn = self.get_reverse_sde(
                score_fn=score_fn, probability_flow=probability_flow)

        def discrete_step_fn(u, t, dt):
            vec_t = torch.ones(
                u.shape[0], device=u.device, dtype=torch.float64) * t
            drift, diffusion = sde_fn(u, vec_t)

            drift *= dt
            diffusion *= np.sqrt(dt)

            noise = torch.randn(*u.shape, device=u.device)

            u_mean = u + drift
            u = u_mean + diffusion * noise
            return u, u_mean
        return discrete_step_fn


class VPSDE(nn.Module):
    def __init__(self, config, beta_fn, beta_int_fn):
        super().__init__()
        self.config = config
        self.beta_fn = beta_fn
        self.beta_int_fn = beta_int_fn

    @property
    def type(self):
        return 'vpsde'

    @property
    def is_augmented(self):
        return False

    def sde(self, u, t):
        beta = add_dimensions(self.beta_fn(t), self.config.is_image)

        drift = -0.5 * beta * u
        diffusion = torch.sqrt(beta) * torch.ones_like(u,
                                                       device=self.config.device)

        return drift, diffusion

    def get_reverse_sde(self, score_fn=None, probability_flow=False):
        sde_fn = self.sde

        def reverse_sde(u, t, score=None):
            drift, diffusion = sde_fn(u, 1. - t)
            score = score if score is not None else score_fn(u, 1. - t)

            reverse_drift = -drift + diffusion**2 * \
                score * (0.5 if probability_flow else 1.0)
            reverse_diffusion = torch.zeros_like(
                diffusion) if probability_flow else diffusion

            return reverse_drift, reverse_diffusion

        return reverse_sde

    def prior_sampling(self, shape):
        return torch.randn(*shape, device=self.config.device), None

    def prior_logp(self, u):
        shape = u.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2. * np.pi) - torch.sum(u.view(u.shape[0], -1) ** 2., dim=1) / 2., None

    def var(self, t, var0x=None):
        if var0x is None:
            var0x = add_dimensions(torch.zeros_like(
                t, dtype=torch.float64, device=t.device), self.config.is_image)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)

        coeff = torch.exp(-beta_int)
        return [1. - (1. - var0x) * coeff]

    def mean(self, x, t):
        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)

        return x * torch.exp(-0.5 * beta_int)

    def mean_and_var(self, x, t, var0x=None):
        if var0x is None:
            var0x = torch.zeros_like(x, device=self.config.device)

        return self.mean(x, t), self.var(t, var0x)

    def noise_multiplier(self, t, var0x=None):
        _var = self.var(t, var0x)[0]
        return -1. / torch.sqrt(_var)

    def loss_multiplier(self, t):
        return 0.5 * self.beta_fn(t)

    def perturb_data(self, batch, t, var0x=None):
        mean, var = self.mean_and_var(batch, t, var0x)
        cholesky = torch.sqrt(var[0])

        batch_randn = torch.randn_like(batch, device=batch.device)
        noise = cholesky * batch_randn

        perturbed_data = mean + noise
        return perturbed_data, mean, noise, batch_randn

    def get_discrete_step_fn(self, mode, score_fn=None, probability_flow=False):
        if mode == 'forward':
            sde_fn = self.sde
        elif mode == 'reverse':
            sde_fn = self.get_reverse_sde(
                score_fn=score_fn, probability_flow=probability_flow)

        def discrete_step_fn(u, t, dt):
            vec_t = torch.ones(
                u.shape[0], device=u.device, dtype=torch.float64) * t
            drift, diffusion = sde_fn(u, vec_t)

            drift *= dt
            diffusion *= np.sqrt(dt)

            noise = torch.randn_like(u, device=u.device)
            u_mean = u + drift
            u = u_mean + diffusion * noise
            return u, u_mean
        return discrete_step_fn
