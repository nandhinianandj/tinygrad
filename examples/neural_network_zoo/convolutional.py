from tinygrad import Tensor, nn

# --- 15. Convolutional Neural Network (CNN) ---
class CNN:
  def __init__(self, in_channels, num_classes):
    self.c1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
    self.c2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.l1 = nn.Linear(32 * 4 * 4, num_classes) # Assuming input 8x8 -> pool -> 4x4

  def __call__(self, x):
    # x: (bs, in_channels, 8, 8)
    x = self.c1(x).relu().max_pool2d() # -> (bs, 16, 4, 4)
    x = self.c2(x).relu() # -> (bs, 32, 4, 4)
    x = x.flatten(1)
    return self.l1(x)

# --- 16. Deconvolutional Network (DN) ---
class DeconvNet:
  def __init__(self, latent_size, out_channels):
    self.l1 = nn.Linear(latent_size, 32 * 2 * 2)
    self.dc1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # 2x2 -> 4x4
    self.dc2 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1) # 4x4 -> 8x8

  def __call__(self, x):
    x = self.l1(x).relu()
    x = x.reshape(x.shape[0], 32, 2, 2)
    x = self.dc1(x).relu()
    x = self.dc2(x).sigmoid()
    return x

# --- 17. Deep Convolutional Inverse Graphics Network (DCIGN) ---
class DCIGN:
  def __init__(self, in_channels, latent_size):
    # Encoder (CNN)
    self.enc_c1 = nn.Conv2d(in_channels, 16, 3, padding=1)
    self.enc_c2 = nn.Conv2d(16, 32, 3, padding=1, stride=2) # 8x8 -> 4x4
    self.enc_flat_size = 32 * 4 * 4
    self.enc_mu = nn.Linear(self.enc_flat_size, latent_size)
    self.enc_logvar = nn.Linear(self.enc_flat_size, latent_size)

    # Decoder (DN)
    self.dec_l1 = nn.Linear(latent_size, 32 * 4 * 4)
    self.dec_dc1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1) # 4x4 -> 8x8
    self.dec_dc2 = nn.ConvTranspose2d(16, in_channels, 3, padding=1) # 8x8 -> 8x8

  def reparameterize(self, mu, logvar):
    std = (logvar * 0.5).exp()
    eps = Tensor.randn(*std.shape, device=std.device)
    return mu + eps * std

  def __call__(self, x):
    # Encode
    h = self.enc_c1(x).relu()
    h = self.enc_c2(h).relu()
    h = h.flatten(1)
    mu = self.enc_mu(h)
    logvar = self.enc_logvar(h)

    # Reparameterize
    z = self.reparameterize(mu, logvar)

    # Decode
    h = self.dec_l1(z).relu()
    h = h.reshape(h.shape[0], 32, 4, 4)
    h = self.dec_dc1(h).relu()
    x_recon = self.dec_dc2(h).sigmoid()

    return x_recon, mu, logvar

# --- 18. Generative Adversarial Network (GAN) ---
class GAN:
  def __init__(self, latent_size, img_channels):
    self.generator = DeconvNet(latent_size, img_channels) # Reuse DeconvNet
    self.discriminator = CNN(img_channels, 1) # Reuse CNN structure but output 1

  # GANs are usually trained with separate forward passes.
  # Here we just instantiate them.
  def __call__(self, x):
    # This doesn't make much sense as a single forward pass,
    # but for structure demonstration:
    fake_img = self.generator(x)
    validity = self.discriminator(fake_img)
    return fake_img, validity
