from pathlib import Path
import argparse

import torch
from torch.distributions import MultivariateNormal
import numpy as np
from scipy.linalg import toeplitz


def main(args):
    torch.manual_seed(args.seed)
    
    duration = 10.0
    delta_t = 1e-1
    
    kappa = 0.5
    lamda = 1.0
    rho = 1.0
    mu = 1.0
    
    sigma_ss = kappa / lamda
    sigma_sx = rho * sigma_ss / (lamda + mu)
    sigma_xx = rho / mu * (sigma_ss + sigma_sx)

    def exp2(x, decay, scale):
        return scale * (-decay * x).exp()
    
    def exprel(x):
        return torch.where(x.abs() > 1e-15, torch.special.expm1(x) / x, 1.0)
    
    def cov_ss(t):
        return exp2(t, lamda, sigma_ss)
    
    def cov_sx(t):
        scale1 = rho * sigma_ss * t * exprel((lamda - mu) * t)
        return exp2(t, mu, sigma_sx) + exp2(t, lamda, scale1)
    
    def cov_xs(t):
        return exp2(t, lamda, sigma_sx)
    
    def cov_xx(t):
        scale1 = rho * sigma_sx * t * exprel((lamda - mu) * t)
        return exp2(t, lamda, scale1) + exp2(t, mu, sigma_xx)

    t = torch.arange(0.0, duration, delta_t)

    cov_mat_ss = toeplitz(cov_ss(t), cov_ss(t))
    cov_mat_xs = toeplitz(cov_sx(t), cov_xs(t))
    cov_mat_sx = toeplitz(cov_xs(t), cov_sx(t))
    cov_mat_xx = toeplitz(cov_xx(t), cov_xx(t))

    cov_mat_z = np.block([[cov_mat_ss, cov_mat_sx], 
                          [cov_mat_xs, cov_mat_xx]])

    cov_mat = torch.tensor(cov_mat_z)
    dist = MultivariateNormal(torch.zeros(cov_mat.size(0)), cov_mat)
    
    train_data = dist.sample((args.size_training,)).reshape((-1, 2, cov_mat.size(0) // 2))
    valid_data = dist.sample((args.size_validation,)).reshape((-1, 2, cov_mat.size(0) // 2))

    torch.save({"training": train_data, "validation": valid_data}, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Gaussian training data')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='path to store output file')
    parser.add_argument('--size-training', type=int, default=1000,
                        help='Size of training data (default %(default)s)')
    parser.add_argument('--size-validation', type=int, default=10000,
                        help='Size of validation data (default %(default)s)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default %(default)s)')
    args = parser.parse_args()
    main(args)
