# In this example, we will create a neural network that can learn the distributional pattern of a given training data,
# producing a kde plot of training data and a different color for the given new data at theinference time.
# We will do it via tiny grad and create test cases using a combination of hypothesis and pymc.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import trange
import pymc as pm

# The generator network
class Generator(object):
  def __init__(self, in_features, out_features):
    self.l1 = Tensor.scaled_uniform(in_features, 128)
    self.l2 = Tensor.scaled_uniform(128, 256)
    self.l3 = Tensor.scaled_uniform(256, out_features)

  def __call__(self, x):
    x = x.dot(self.l1).leakyrelu()
    x = x.dot(self.l2).leakyrelu()
    x = x.dot(self.l3)
    return x

# The discriminator network
class Discriminator(object):
  def __init__(self, in_features, out_features):
    self.l1 = Tensor.scaled_uniform(in_features, 256)
    self.l2 = Tensor.scaled_uniform(256, 128)
    self.l3 = Tensor.scaled_uniform(128, out_features)

  def __call__(self, x):
    x = x.dot(self.l1).leakyrelu()
    x = x.dot(self.l2).leakyrelu()
    x = x.dot(self.l3).sigmoid()
    return x

if __name__ == "__main__":
  # Parameters
  latent_dim = 100
  data_dim = 1
  n_samples = 500
  epochs = 500
  batch_size = 32

  # Create a pymc model
  with pm.Model() as model:
    w = pm.Dirichlet('w', np.array([1, 1]))
    mu = pm.Normal('mu', mu=np.array([0, 5]), sigma=1, shape=2)
    tau = pm.Gamma('tau', alpha=1, beta=1, shape=2)
    pm.NormalMixture('x', w=w, mu=mu, tau=tau)

  # Generate data from the model
  with model:
    trace = pm.sample(n_samples)
  real_data = trace.posterior['x'].values.flatten()

  # Train the GAN on the data
  generator = Generator(latent_dim, data_dim)
  discriminator = Discriminator(data_dim, 1)
  optimizer_g = optim.Adam(get_parameters(generator), lr=0.0002, b1=0.5)
  optimizer_d = optim.Adam(get_parameters(discriminator), lr=0.0002, b1=0.5)
  Tensor.training = True
  for epoch in (t := trange(epochs)):
    # Create the real data
    real_data_batch = Tensor(np.random.choice(real_data, batch_size).reshape(batch_size, 1).astype(np.float32))

    # Create the fake data
    noise = Tensor.randn(batch_size, latent_dim)
    fake_data = generator(noise)

    # Train the discriminator
    optimizer_d.zero_grad()
    real_loss = -(discriminator(real_data_batch).log()).mean()
    fake_loss = -((1 - discriminator(fake_data.detach())).log()).mean()
    d_loss = (real_loss + fake_loss) / 2
    d_loss.backward()
    optimizer_d.step()

    # Train the generator
    optimizer_g.zero_grad()
    g_loss = -(discriminator(fake_data).log()).mean()
    g_loss.backward()
    optimizer_g.step()

    t.set_description(f"d_loss: {d_loss.numpy().item():.4f}, g_loss: {g_loss.numpy().item():.4f}")

  # Generate a KDE plot of the real and fake data
  noise = Tensor.randn(n_samples, latent_dim)
  fake_data = generator(noise).numpy()

  sns.kdeplot(real_data, color='blue', label='Real Data')
  sns.kdeplot(fake_data[:, 0], color='red', label='Fake Data')
  plt.legend()
  plt.savefig('examples/pymc_distribution.jpg')
  plt.close()
