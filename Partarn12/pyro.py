import torch
import torch.distributions as dist
import pyro
import matplotlib.pyplot as plt
pyro.set_rng_seed(101)

num = 200
data = torch.cat((dist.MultivariateNormal(-8 * torch.ones(2), torch.eye(2)).sample([num]),
                  dist.MultivariateNormal(8 * torch.ones(2), torch.eye(2)).sample([num]),
                  dist.MultivariateNormal(torch.tensor([-5., 5.]), torch.eye(2)).sample([num]),
                  dist.MultivariateNormal(torch.tensor([6., -5.]), torch.eye(2)).sample([num])
                 ))
plt.scatter(data[:, 0], data[:, 1])
plt.show()