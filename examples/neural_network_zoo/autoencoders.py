from tinygrad import Tensor, nn

# --- 6. Autoencoder (AE) ---
class Autoencoder:
  def __init__(self, input_size, hidden_size):
    self.encoder = nn.Linear(input_size, hidden_size)
    self.decoder = nn.Linear(hidden_size, input_size)

  def __call__(self, x):
    encoded = self.encoder(x).relu()
    decoded = self.decoder(encoded).sigmoid() # Assuming normalized input
    return decoded

# --- 7. Variational Autoencoder (VAE) ---
class VAE:
  def __init__(self, input_size, hidden_size, latent_size):
    self.encoder = nn.Linear(input_size, hidden_size)
    self.fc_mu = nn.Linear(hidden_size, latent_size)
    self.fc_logvar = nn.Linear(hidden_size, latent_size)
    self.decoder_l1 = nn.Linear(latent_size, hidden_size)
    self.decoder_l2 = nn.Linear(hidden_size, input_size)

  def reparameterize(self, mu, logvar):
    std = (logvar * 0.5).exp()
    eps = Tensor.randn(*std.shape, device=std.device)
    return mu + eps * std

  def __call__(self, x):
    h = self.encoder(x).relu()
    mu = self.fc_mu(h)
    logvar = self.fc_logvar(h)
    z = self.reparameterize(mu, logvar)
    h_dec = self.decoder_l1(z).relu()
    return self.decoder_l2(h_dec).sigmoid(), mu, logvar

# --- 8. Denoising Autoencoder (DAE) ---
class DAE:
  def __init__(self, input_size, hidden_size, noise_std=0.1):
    self.ae = Autoencoder(input_size, hidden_size)
    self.noise_std = noise_std

  def __call__(self, x):
    noisy_x = x + Tensor.randn(*x.shape, device=x.device) * self.noise_std
    return self.ae(noisy_x)

# --- 9. Sparse Autoencoder (SAE) ---
class SAE:
  def __init__(self, input_size, hidden_size):
    # Usually has hidden_size > input_size (overcomplete) or sparsity constraint
    self.encoder = nn.Linear(input_size, hidden_size)
    self.decoder = nn.Linear(hidden_size, input_size)

  def __call__(self, x):
    # In practice, sparsity is enforced via loss function (L1 regularization on activations)
    # The architecture is standard AE.
    encoded = self.encoder(x).sigmoid() # Sigmoid to bounded activation [0, 1] often used for sparsity
    decoded = self.decoder(encoded).sigmoid()
    return decoded, encoded
