import unittest
import pymc as pm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from examples.distribution_gan import Generator, Discriminator
from tinygrad.tensor import Tensor
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import trange
from hypothesis import given, strategies as st

class TestDistributionGAN(unittest.TestCase):
  def test_gan_creation(self):
    generator = Generator(100, 1)
    discriminator = Discriminator(1, 1)
    self.assertIsNotNone(generator)
    self.assertIsNotNone(discriminator)

  def test_gan_forward_pass(self):
    generator = Generator(100, 1)
    discriminator = Discriminator(1, 1)
    noise = Tensor.randn(1, 100)
    fake_data = generator(noise)
    self.assertEqual(fake_data.shape, (1, 1))
    prediction = discriminator(fake_data)
    self.assertEqual(prediction.shape, (1, 1))

  @given(st.floats(min_value=-1, max_value=1))
  def test_gan_hypothesis(self, value):
    generator = Generator(100, 1)
    discriminator = Discriminator(1, 1)
    noise = Tensor.full((1, 100), value)
    fake_data = generator(noise)
    self.assertEqual(fake_data.shape, (1, 1))
    prediction = discriminator(fake_data)
    self.assertEqual(prediction.shape, (1, 1))

  def test_gan_with_pymc(self):
    # Create a pymc model
    with pm.Model() as model:
      w = pm.Dirichlet('w', np.array([1, 1]))
      mu = pm.Normal('mu', mu=np.array([0, 5]), sigma=1, shape=2)
      tau = pm.Gamma('tau', alpha=1, beta=1, shape=2)
      pm.NormalMixture('x', w=w, mu=mu, tau=tau)

    # Generate data from the model
    with model:
      trace = pm.sample(1000)
    real_data = trace.posterior['x'].values.flatten()

    # Train the GAN on the data
    generator = Generator(100, 1)
    discriminator = Discriminator(1, 1)
    optimizer_g = optim.Adam(get_parameters(generator), lr=0.0002, b1=0.5)
    optimizer_d = optim.Adam(get_parameters(discriminator), lr=0.0002, b1=0.5)
    Tensor.training = True
    for epoch in (t := trange(1000)):
      # Create the real data
      real_data_batch = Tensor(np.random.choice(real_data, 64).reshape(64, 1).astype(np.float32))

      # Create the fake data
      noise = Tensor.randn(64, 100)
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
    noise = Tensor.randn(1000, 100)
    fake_data = generator(noise).numpy()

    sns.kdeplot(real_data, color='blue', label='Real Data')
    sns.kdeplot(fake_data[:, 0], color='red', label='Fake Data')
    plt.legend()
    plt.savefig('test/pymc_distribution.png')
    plt.close()

if __name__ == '__main__':
  unittest.main()
