from pathlib import Path

import torch
from torch.distributions import MultivariateNormal
import numpy as np
from scipy.linalg import toeplitz


def main(output_path):
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
    
    train_data = dist.sample((1_000,)).reshape((-1, 2, cov_mat.size(0) // 2))
    valid_data = dist.sample((1_000,)).reshape((-1, 2, cov_mat.size(0) // 2))

    torch.save({"training": train_data, "validation": valid_data}, output_path)


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[2]
    main(project_dir / 'data' / 'gaussian_data.pt')
