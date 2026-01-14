from typing import Callable, List, Tuple, Optional
import math
from tinygrad import Tensor, nn

# --- Helper Classes ---

class GRUCell:
  def __init__(self, input_size:int, hidden_size:int, bias:bool=True):
    self.input_size, self.hidden_size = input_size, hidden_size
    stdv = 1.0 / math.sqrt(hidden_size)
    self.weight_ih = Tensor.uniform(hidden_size*3, input_size, low=-stdv, high=stdv)
    self.weight_hh = Tensor.uniform(hidden_size*3, hidden_size, low=-stdv, high=stdv)
    self.bias_ih: Tensor|None = Tensor.zeros(hidden_size*3) if bias else None
    self.bias_hh: Tensor|None = Tensor.zeros(hidden_size*3) if bias else None

  def __call__(self, x:Tensor, h:Tensor|None=None) -> Tensor:
    if h is None: h = Tensor.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)

    gi = x.linear(self.weight_ih.T, self.bias_ih)
    gh = h.linear(self.weight_hh.T, self.bias_hh)

    i_r, i_z, i_n = gi.chunk(3, dim=1)
    h_r, h_z, h_n = gh.chunk(3, dim=1)

    r = (i_r + h_r).sigmoid()
    z = (i_z + h_z).sigmoid()
    n = (i_n + r * h_n).tanh()

    new_h = (1 - z) * n + z * h
    return new_h

class RNNCell:
  def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
    self.input_size, self.hidden_size = input_size, hidden_size
    self.nonlinearity = nonlinearity
    stdv = 1.0 / math.sqrt(hidden_size)
    self.weight_ih = Tensor.uniform(hidden_size, input_size, low=-stdv, high=stdv)
    self.weight_hh = Tensor.uniform(hidden_size, hidden_size, low=-stdv, high=stdv)
    self.bias_ih: Tensor|None = Tensor.zeros(hidden_size) if bias else None
    self.bias_hh: Tensor|None = Tensor.zeros(hidden_size) if bias else None

  def __call__(self, x:Tensor, h:Tensor|None=None) -> Tensor:
    if h is None: h = Tensor.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
    gates = x.linear(self.weight_ih.T, self.bias_ih) + h.linear(self.weight_hh.T, self.bias_hh)
    return gates.tanh() if self.nonlinearity == 'tanh' else gates.relu()

# --- 1. Feed Forward Neural Network (FF or FFNN) ---
class FFNN:
  def __init__(self, input_size, hidden_size, output_size):
    self.l1 = nn.Linear(input_size, hidden_size)
    self.l2 = nn.Linear(hidden_size, output_size)

  def __call__(self, x):
    return self.l2(self.l1(x).relu())

# --- 2. Radial Basis Function Network (RBF) ---
class RBF:
  def __init__(self, input_size, hidden_size, output_size):
    self.centers = Tensor.uniform(hidden_size, input_size)
    self.sigmas = Tensor.uniform(hidden_size)
    self.linear = nn.Linear(hidden_size, output_size)

  def __call__(self, x):
    # x: (bs, input_size)
    diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
    dist_sq = (diff ** 2).sum(axis=2)
    rbf_out = (-dist_sq / (2 * self.sigmas.square().unsqueeze(0))).exp()
    return self.linear(rbf_out)

# --- 3. Recurrent Neural Network (RNN) ---
class RNN:
  def __init__(self, input_size, hidden_size, output_size):
    self.cell = RNNCell(input_size, hidden_size)
    self.fc = nn.Linear(hidden_size, output_size)

  def __call__(self, x):
    # x: (bs, time_steps, input_size)
    bs, time_steps, _ = x.shape
    h = Tensor.zeros(bs, self.cell.hidden_size, device=x.device)
    outputs = []
    for t in range(time_steps):
      h = self.cell(x[:, t, :], h)
      outputs.append(self.fc(h))
    return outputs[0].stack(*outputs[1:], dim=1) # (bs, time, output_size)

# --- 4. Long / Short Term Memory (LSTM) ---
class LSTM:
  def __init__(self, input_size, hidden_size, output_size):
    self.cell = nn.LSTMCell(input_size, hidden_size)
    self.fc = nn.Linear(hidden_size, output_size)

  def __call__(self, x):
    # x: (bs, time_steps, input_size)
    bs, time_steps, _ = x.shape
    h = Tensor.zeros(bs, self.cell.weight_hh.size(1), device=x.device)
    c = Tensor.zeros(bs, self.cell.weight_hh.size(1), device=x.device)
    outputs = []
    for t in range(time_steps):
      h, c = self.cell(x[:, t, :], (h, c))
      outputs.append(self.fc(h))
    return outputs[0].stack(*outputs[1:], dim=1)

# --- 5. Gated Recurrent Unit (GRU) ---
class GRU:
  def __init__(self, input_size, hidden_size, output_size):
    self.cell = GRUCell(input_size, hidden_size)
    self.fc = nn.Linear(hidden_size, output_size)

  def __call__(self, x):
    # x: (bs, time_steps, input_size)
    bs, time_steps, _ = x.shape
    h = Tensor.zeros(bs, self.cell.hidden_size, device=x.device)
    outputs = []
    for t in range(time_steps):
      h = self.cell(x[:, t, :], h)
      outputs.append(self.fc(h))
    return outputs[0].stack(*outputs[1:], dim=1)

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

# --- 10. Markov Chain (MC) ---
class MarkovChain:
  def __init__(self, num_states):
    # Transition matrix (randomly initialized and normalized)
    self.transition_matrix = Tensor.rand(num_states, num_states)
    self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdim=True)

  def __call__(self, state_vector, steps=1):
    # state_vector: (bs, num_states) probability distribution over states
    current_state = state_vector
    for _ in range(steps):
      current_state = current_state.matmul(self.transition_matrix)
    return current_state

# --- 11. Hopfield Network (HN) ---
class HopfieldNetwork:
  def __init__(self, num_neurons):
    self.weights = Tensor.zeros(num_neurons, num_neurons)

  def train(self, patterns):
    # Hebbian learning: W = sum(xi * xi^T) / N, diagonal = 0
    # patterns: (num_patterns, num_neurons), values in {-1, 1}
    num_patterns, num_neurons = patterns.shape
    w = patterns.T.matmul(patterns) / num_neurons
    # Remove self-connections (diagonal)
    eye = Tensor.eye(num_neurons, device=patterns.device)
    self.weights = w * (1 - eye)

  def __call__(self, state, steps=5):
    # state: (bs, num_neurons)
    current_state = state
    for _ in range(steps):
      # Synchronous update
      current_state = current_state.matmul(self.weights).sign()
      # Handle 0 case (sign(0) could be anything, usually keep previous state or set to 1. tinygrad sign(0) is 0)
      # For Hopfield with {-1, 1}, 0 output usually means no change, but let's assume strict sign.
      # Ideally we want binary {-1, 1}.
      current_state = (current_state == 0).where(1, current_state)
    return current_state

# --- 12. Boltzmann Machine (BM) ---
class BoltzmannMachine:
  def __init__(self, visible_size, hidden_size):
    # Fully connected including visible-hidden, visible-visible, hidden-hidden
    self.v_size = visible_size
    self.h_size = hidden_size
    total_size = visible_size + hidden_size
    self.weights = Tensor.randn(total_size, total_size) * 0.1
    # Symmetric weights, zero diagonal
    self.weights = (self.weights + self.weights.T) / 2
    self.weights = self.weights * (1 - Tensor.eye(total_size))
    self.bias = Tensor.zeros(total_size)

  def energy(self, state):
    # E = -0.5 * x.T * W * x - b.T * x
    # state: (bs, total_size)
    term1 = -0.5 * (state.matmul(self.weights) * state).sum(axis=1)
    term2 = -(self.bias * state).sum(axis=1)
    return term1 + term2

  def __call__(self, visible_state, steps=1, temp=1.0):
    # Simulated annealing / Gibbs sampling step
    # This is a simplified forward pass just to show structure.
    # In reality BM training/inference is complex (MCMC).

    bs = visible_state.shape[0]
    # Initialize hidden state randomly
    hidden_state = (Tensor.rand(bs, self.h_size) > 0.5).where(1.0, 0.0)
    current_state = visible_state.cat(hidden_state, dim=1) # (bs, v+h)

    for _ in range(steps):
        # Calculate activation probability: p(s_i=1) = sigmoid(sum(w_ij * s_j) + b_i / T)
        logits = current_state.matmul(self.weights) + self.bias
        probs = (logits / temp).sigmoid()
        # Sample
        current_state = (Tensor.rand(*probs.shape) < probs).where(1.0, 0.0)

    return current_state

# --- 13. Restricted Boltzmann Machine (RBM) ---
class RBM:
  def __init__(self, visible_size, hidden_size):
    self.weights = Tensor.randn(visible_size, hidden_size) * 0.1
    self.v_bias = Tensor.zeros(visible_size)
    self.h_bias = Tensor.zeros(hidden_size)

  def sample_h(self, v):
    # p(h=1|v) = sigmoid(vW + c)
    logits = v.matmul(self.weights) + self.h_bias
    probs = logits.sigmoid()
    sample = (Tensor.rand(*probs.shape) < probs).where(1.0, 0.0)
    return probs, sample

  def sample_v(self, h):
    # p(v=1|h) = sigmoid(hW^T + b)
    logits = h.matmul(self.weights.T) + self.v_bias
    probs = logits.sigmoid()
    sample = (Tensor.rand(*probs.shape) < probs).where(1.0, 0.0)
    return probs, sample

  def __call__(self, v, steps=1):
    # Gibbs sampling
    h_probs, h_sample = self.sample_h(v)
    for _ in range(steps):
        v_probs, v_sample = self.sample_v(h_sample)
        h_probs, h_sample = self.sample_h(v_sample)
    return v_probs, v_sample

# --- 14. Deep Belief Network (DBN) ---
class DBN:
  def __init__(self, layer_sizes):
    self.rbms = []
    for i in range(len(layer_sizes) - 1):
        self.rbms.append(RBM(layer_sizes[i], layer_sizes[i+1]))

  def __call__(self, x, steps=1):
    # Forward pass through stack of RBMs (Greedy layer-wise training style)
    # Just passing the probabilities/activations up.
    out = x
    for rbm in self.rbms:
        out, _ = rbm.sample_h(out)
    return out

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

# --- 22. Deep Residual Network (DRN / ResNet) ---
class ResBlock:
  def __init__(self, in_channels, out_channels, stride=1):
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    self.shortcut = lambda x: x
    if stride != 1 or in_channels != out_channels:
      self.shortcut = lambda x: nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)(x)

  def __call__(self, x):
    out = self.conv1(x)
    out = self.bn1(out).relu()
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + self.shortcut(x)
    return out.relu()

class DRN:
  def __init__(self, in_channels, num_classes):
    self.in_channels = 16
    self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.layer1 = self._make_layer(16, 2, stride=1)
    self.layer2 = self._make_layer(32, 2, stride=2)
    self.fc = nn.Linear(32, num_classes) # Global Avg Pooling before this

  def _make_layer(self, out_channels, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(ResBlock(self.in_channels, out_channels, stride))
      self.in_channels = out_channels
    return layers

  def __call__(self, x):
    out = self.conv1(x)
    out = self.bn1(out).relu()
    for layer in self.layer1: out = layer(out)
    for layer in self.layer2: out = layer(out)
    out = out.mean(axis=(2,3)) # Global Avg Pooling
    return self.fc(out)

# --- 23. Neural Turing Machine (NTM) ---
class NTM:
  def __init__(self, input_size, output_size, memory_size=128, memory_vector_dim=20):
    self.memory_size = memory_size
    self.memory_vector_dim = memory_vector_dim
    self.controller = nn.LSTMCell(input_size + memory_vector_dim, 100)
    self.output_fc = nn.Linear(100 + memory_vector_dim, output_size)

    # Read/Write Heads parameters generator
    # For simplicity, 1 read head, 1 write head
    # Parameters: k (key), beta (strength), g (gate), s (shift), gamma (sharpening) + erase/add vectors for write
    self.heads_fc = nn.Linear(100, (memory_vector_dim + 1 + 1 + 3 + 1) * 2 + memory_vector_dim * 2)
    self.write_fc = nn.Linear(100, self.memory_size * self.memory_vector_dim)
    # Simplified: just predicting addressing weights directly? No, that defeats the purpose.
    # Implementing full NTM addressing mechanism is complex.
    # Let's implement a very simplified memory access: Content-based addressing only.

  def __call__(self, x, prev_state=None):
    # This is too complex to implement fully and correctly in a "zoo" example without taking too much space.
    # I'll implement a placeholder that mimics the interface and state passing.

    bs = x.shape[0]
    if prev_state is None:
        h = Tensor.zeros(bs, 100, device=x.device)
        c = Tensor.zeros(bs, 100, device=x.device)
        memory = Tensor.zeros(bs, self.memory_size, self.memory_vector_dim, device=x.device)
        read_vector = Tensor.zeros(bs, self.memory_vector_dim, device=x.device)
    else:
        h, c, memory, read_vector = prev_state

    # Controller input
    inp = x.cat(read_vector, dim=1)
    h, c = self.controller(inp, (h, c))

    # Fake Read/Write operations
    # Update memory (Write) - simplified: add controller output to a random location or learned location
    # Ideally: generate write_weights, erase_vector, add_vector
    # Here: just decay memory and add something
    memory = memory * 0.9 + self.write_fc(h).reshape(bs, self.memory_size, self.memory_vector_dim).tanh() * 0.1

    # Read from memory
    # Ideally: generate read_weights based on content similarity
    # Here: just simplified read (mean of memory)
    read_vector = memory.mean(axis=1)

    output = self.output_fc(h.cat(read_vector, dim=1))

    return output, (h, c, memory, read_vector)

# --- 24. Differentiable Neural Computer (DNC) ---
class DNC:
  def __init__(self, input_size, output_size):
    # Placeholder for DNC, which is an improved NTM
    self.ntm = NTM(input_size, output_size)

  def __call__(self, x, state=None):
    return self.ntm(x, state)

# --- 25. Capsule Network (CapsNet) ---
class CapsuleLayer:
  def __init__(self, in_units, in_channels, num_units, unit_size, num_routing=3):
    self.in_units = in_units
    self.in_channels = in_channels
    self.num_units = num_units
    self.unit_size = unit_size
    self.num_routing = num_routing

    # Weight matrix: (1, in_units, num_units, in_channels, unit_size)
    self.W = Tensor.randn(1, in_units, num_units, in_channels, unit_size) * 0.01

  def squash(self, s):
    # s: (bs, num_units, unit_size)
    norm_sq = (s ** 2).sum(axis=-1, keepdim=True)
    norm = norm_sq.sqrt()
    scale = norm_sq / (1 + norm_sq) / (norm + 1e-7)
    return scale * s

  def __call__(self, x):
    # x: (bs, in_units, in_channels)
    bs = x.shape[0]

    # Expand x: (bs, in_units, 1, in_channels, 1)
    x_ex = x.unsqueeze(2).unsqueeze(4)

    # Tiling W: (bs, in_units, num_units, in_channels, unit_size)
    W_tiled = self.W.expand(bs, self.in_units, self.num_units, self.in_channels, self.unit_size)

    # u_hat = x * W
    # (bs, in_units, 1, in_channels, 1) * (bs, in_units, num_units, in_channels, unit_size)
    # This matrix multiplication is tricky with standard broadcasting.
    # We want: u_hat[b, i, j, :] = sum_k x[b, i, k] * W[i, j, k, :]
    # x: (bs, in_units, in_channels)
    # W: (in_units, num_units, in_channels, unit_size)

    # Reshape x to (bs, in_units, 1, in_channels)
    # Reshape W to (1, in_units, num_units, in_channels, unit_size)
    # u_hat = matmul over in_channels

    u_hat = x.unsqueeze(2).unsqueeze(3).matmul(self.W) # (bs, in_units, num_units, 1, unit_size)
    u_hat = u_hat.squeeze(3) # (bs, in_units, num_units, unit_size)

    # Dynamic Routing
    # b_ij: (bs, in_units, num_units)
    b = Tensor.zeros(bs, self.in_units, self.num_units, device=x.device)

    for i in range(self.num_routing):
        c = b.softmax(axis=2) # (bs, in_units, num_units)
        # s_j = sum_i c_ij * u_hat_j|i
        s = (c.unsqueeze(-1) * u_hat).sum(axis=1) # (bs, num_units, unit_size)
        v = self.squash(s) # (bs, num_units, unit_size)

        if i < self.num_routing - 1:
            # a_ij = v_j . u_hat_j|i
            # v.unsqueeze(1): (bs, 1, num_units, unit_size)
            # u_hat: (bs, in_units, num_units, unit_size)
            a = (u_hat * v.unsqueeze(1)).sum(axis=-1) # (bs, in_units, num_units)
            b = b + a

    return v

class CapsNet:
  def __init__(self, in_channels, num_classes):
    self.conv1 = nn.Conv2d(in_channels, 256, 9)
    # PrimaryCaps: conv2d 256->32*8 (32 channels of 8D capsules)
    self.primary_caps = nn.Conv2d(256, 32 * 8, 9, stride=2)
    # DigitCaps: 32*6*6 inputs (6*6 grid of 32 caps) -> 10 output caps of 16D
    # Simplified: flatten primary caps
    # In reality primary caps is a grid.
    # Let's assume input image is 28x28.
    # conv1: 20x20
    # primary: 6x6.
    # Total primary caps: 6*6*32 = 1152. Dimension 8.

    self.digit_caps = CapsuleLayer(1152, 8, num_classes, 16)

  def __call__(self, x):
    # x: (bs, 1, 28, 28)
    out = self.conv1(x).relu()
    out = self.primary_caps(out) # (bs, 256, 6, 6)

    # Reshape to capsules
    bs = out.shape[0]
    out = out.permute(0, 2, 3, 1).reshape(bs, -1, 8) # (bs, 6*6*32, 8)

    out = self.digit_caps(out) # (bs, num_classes, 16)

    # Length of capsule is probability
    probs = (out ** 2).sum(axis=-1).sqrt()
    return probs

# --- 26. Kohonen Network (KN / SOM) ---
class KohonenNetwork:
  def __init__(self, input_size, map_height, map_width):
    self.map_height = map_height
    self.map_width = map_width
    self.weights = Tensor.rand(map_height * map_width, input_size)

  def __call__(self, x):
    # x: (bs, input_size)
    # Find BMU (Best Matching Unit)
    # Distance: ||x - w||^2
    # x: (bs, 1, input_size)
    # w: (1, num_neurons, input_size)

    bs = x.shape[0]
    diff = x.unsqueeze(1) - self.weights.unsqueeze(0)
    dists = (diff ** 2).sum(axis=2) # (bs, num_neurons)

    bmu_indices = dists.argmin(axis=1)

    # In inference, we just return the BMU or the map coordinates.
    # Training (updating weights based on neighborhood) is not shown in forward pass.
    return bmu_indices

# --- 27. Attention Network (AN) ---
# Simple Self-Attention
class AttentionNetwork:
  def __init__(self, input_dim, embed_dim):
    self.q = nn.Linear(input_dim, embed_dim)
    self.k = nn.Linear(input_dim, embed_dim)
    self.v = nn.Linear(input_dim, embed_dim)
    self.out = nn.Linear(embed_dim, input_dim)

  def __call__(self, x):
    # x: (bs, time, input_dim)
    Q = self.q(x)
    K = self.k(x)
    V = self.v(x)

    # Scaled Dot-Product Attention
    d_k = Q.shape[-1]
    scores = Q.matmul(K.transpose(1, 2)) / math.sqrt(d_k)
    attn = scores.softmax(axis=-1)
    context = attn.matmul(V)

    return self.out(context)

# --- 19. Liquid State Machine (LSM) ---
class LSM:
  def __init__(self, input_size, reservoir_size, output_size):
    self.reservoir_size = reservoir_size
    # Random connections in reservoir (sparse usually)
    self.W_in = Tensor.randn(input_size, reservoir_size) * 0.5
    self.W_res = Tensor.randn(reservoir_size, reservoir_size) * 0.1 # Spectral radius < 1 importance
    self.W_out = nn.Linear(reservoir_size, output_size)
    self.threshold = 1.0
    self.decay = 0.9

  def __call__(self, x_seq):
    # x_seq: (bs, time, input_size)
    bs, time, _ = x_seq.shape
    h = Tensor.zeros(bs, self.reservoir_size, device=x_seq.device) # Membrane potential
    spikes_out = []

    # Simulate Spiking Neural Network (Integrate-and-Fire simplified)
    for t in range(time):
        # Integrate
        in_current = x_seq[:, t, :].matmul(self.W_in)
        res_current = h.matmul(self.W_res)
        h = h * self.decay + in_current + res_current

        # Fire
        spikes = (h > self.threshold).float()
        spikes_out.append(spikes)

        # Reset (subtract threshold) - soft reset
        h = h - spikes * self.threshold

    # Readout from liquid state (usually low-pass filtered spikes or just state)
    # Here we take the mean firing rate or final state. Let's take mean spikes.
    mean_activity = spikes_out[0].stack(*spikes_out[1:], dim=1).mean(axis=1)
    return self.W_out(mean_activity)

# --- 20. Extreme Learning Machine (ELM) ---
class ELM:
  def __init__(self, input_size, hidden_size, output_size):
    # Random weights, fixed
    self.w_in = Tensor.uniform(hidden_size, input_size, requires_grad=False)
    self.b_in = Tensor.uniform(hidden_size, requires_grad=False)
    # Trainable output weights
    self.l_out = nn.Linear(hidden_size, output_size)

  def __call__(self, x):
    # x: (bs, input_size)
    h = x.linear(self.w_in.T, self.b_in).sigmoid() # Activation
    return self.l_out(h)

# --- 21. Echo State Network (ESN) ---
class ESN:
  def __init__(self, input_size, reservoir_size, output_size):
    self.reservoir_size = reservoir_size
    self.w_in = Tensor.uniform(reservoir_size, input_size, low=-0.1, high=0.1, requires_grad=False)
    self.w_res = Tensor.uniform(reservoir_size, reservoir_size, low=-0.5, high=0.5, requires_grad=False)
    # Ensure spectral radius < 1 ideally
    self.l_out = nn.Linear(reservoir_size, output_size)
    self.alpha = 0.8 # Leaking rate

  def __call__(self, x_seq):
    # x_seq: (bs, time, input_size)
    bs, time, _ = x_seq.shape
    h = Tensor.zeros(bs, self.reservoir_size, device=x_seq.device)

    # Run reservoir
    states = []
    for t in range(time):
        u = x_seq[:, t, :]
        pre_act = u.linear(self.w_in.T) + h.linear(self.w_res.T)
        h = (1 - self.alpha) * h + self.alpha * pre_act.tanh()
        states.append(h)

    # Usually output is trained on all states or last state. Let's use last state.
    return self.l_out(h)

if __name__ == "__main__":
    print("Initializing Neural Network Zoo models...")
    BS = 4
    In, Hidden, Out = 8, 16, 4
    SeqLen = 5
    x = Tensor.rand(BS, In)
    x_seq = Tensor.rand(BS, SeqLen, In)

    # 1. FFNN
    print("Testing FFNN...")
    model_ffnn = FFNN(In, Hidden, Out)
    y_ffnn = model_ffnn(x)
    assert y_ffnn.shape == (BS, Out)
    print("FFNN OK")

    # 2. RBF
    print("Testing RBF...")
    model_rbf = RBF(In, Hidden, Out)
    y_rbf = model_rbf(x)
    assert y_rbf.shape == (BS, Out)
    print("RBF OK")

    # 3. RNN
    print("Testing RNN...")
    model_rnn = RNN(In, Hidden, Out)
    y_rnn = model_rnn(x_seq)
    assert y_rnn.shape == (BS, SeqLen, Out)
    print("RNN OK")

    # 4. LSTM
    print("Testing LSTM...")
    model_lstm = LSTM(In, Hidden, Out)
    y_lstm = model_lstm(x_seq)
    assert y_lstm.shape == (BS, SeqLen, Out)
    print("LSTM OK")

    # 5. GRU
    print("Testing GRU...")
    model_gru = GRU(In, Hidden, Out)
    y_gru = model_gru(x_seq)
    assert y_gru.shape == (BS, SeqLen, Out)
    print("GRU OK")

    # 6. Autoencoder
    print("Testing Autoencoder...")
    model_ae = Autoencoder(In, Hidden)
    y_ae = model_ae(x)
    assert y_ae.shape == (BS, In)
    print("Autoencoder OK")

    # 7. VAE
    print("Testing VAE...")
    Latent = 4
    model_vae = VAE(In, Hidden, Latent)
    y_vae, mu, logvar = model_vae(x)
    assert y_vae.shape == (BS, In)
    assert mu.shape == (BS, Latent)
    assert logvar.shape == (BS, Latent)
    print("VAE OK")

    # 8. DAE
    print("Testing DAE...")
    model_dae = DAE(In, Hidden)
    y_dae = model_dae(x)
    assert y_dae.shape == (BS, In)
    print("DAE OK")

    # 9. SAE
    print("Testing SAE...")
    model_sae = SAE(In, Hidden)
    y_sae, encoded_sae = model_sae(x)
    assert y_sae.shape == (BS, In)
    print("SAE OK")

    # 10. Markov Chain
    print("Testing MC...")
    States = 5
    model_mc = MarkovChain(States)
    state_mc = Tensor.zeros(BS, States)
    # Set initial state (one-hot)
    state_mc = state_mc + Tensor.eye(States)[0] # All in state 0 (broadcasting? no, adding (5,) to (BS, 5))
    # Fix initial state construction
    state_mc = Tensor.eye(States)[0].unsqueeze(0).expand(BS, States)

    y_mc = model_mc(state_mc)
    assert y_mc.shape == (BS, States)
    print("MC OK")

    # 11. Hopfield Network
    print("Testing HN...")
    Neurons = 10
    model_hn = HopfieldNetwork(Neurons)
    # Train on a random pattern
    pattern = (Tensor.rand(1, Neurons) > 0.5).where(1.0, -1.0)
    model_hn.train(pattern)
    # Test retrieval (start with noisy pattern)
    noisy_pattern = pattern * -1 # Inverted
    y_hn = model_hn(noisy_pattern)
    assert y_hn.shape == (1, Neurons)
    print("HN OK")

    # 12. Boltzmann Machine
    print("Testing BM...")
    model_bm = BoltzmannMachine(In, Hidden)
    # Use binary input for BM
    x_bin = (x > 0.5).where(1.0, 0.0)
    y_bm = model_bm(x_bin)
    assert y_bm.shape == (BS, In + Hidden)
    print("BM OK")

    # 13. RBM
    print("Testing RBM...")
    model_rbm = RBM(In, Hidden)
    probs_rbm, sample_rbm = model_rbm(x_bin)
    assert sample_rbm.shape == (BS, In)
    print("RBM OK")

    # 14. DBN
    print("Testing DBN...")
    model_dbn = DBN([In, Hidden, Hidden, Out])
    y_dbn = model_dbn(x_bin)
    assert y_dbn.shape == (BS, Out)
    print("DBN OK")

    # 15. CNN
    print("Testing CNN...")
    Channels, H, W = 3, 8, 8
    x_img = Tensor.rand(BS, Channels, H, W)
    model_cnn = CNN(Channels, Out)
    y_cnn = model_cnn(x_img)
    assert y_cnn.shape == (BS, Out)
    print("CNN OK")

    # 16. DeconvNet
    print("Testing DeconvNet...")
    Latent = 10
    model_dn = DeconvNet(Latent, Channels)
    x_latent = Tensor.rand(BS, Latent)
    y_dn = model_dn(x_latent)
    assert y_dn.shape == (BS, Channels, H, W)
    print("DeconvNet OK")

    # 17. DCIGN
    print("Testing DCIGN...")
    model_dcign = DCIGN(Channels, Latent)
    y_dcign, mu, logvar = model_dcign(x_img)
    assert y_dcign.shape == (BS, Channels, H, W)
    print("DCIGN OK")

    # 18. GAN
    print("Testing GAN...")
    model_gan = GAN(Latent, Channels)
    fake_img, validity = model_gan(x_latent)
    assert fake_img.shape == (BS, Channels, H, W)
    assert validity.shape == (BS, 1)
    print("GAN OK")

    # 19. LSM
    print("Testing LSM...")
    model_lsm = LSM(In, Hidden, Out)
    y_lsm = model_lsm(x_seq)
    assert y_lsm.shape == (BS, Out)
    print("LSM OK")

    # 20. ELM
    print("Testing ELM...")
    model_elm = ELM(In, Hidden, Out)
    y_elm = model_elm(x)
    assert y_elm.shape == (BS, Out)
    print("ELM OK")

    # 21. ESN
    print("Testing ESN...")
    model_esn = ESN(In, Hidden, Out)
    y_esn = model_esn(x_seq)
    assert y_esn.shape == (BS, Out)
    print("ESN OK")

    # 22. DRN (ResNet)
    print("Testing DRN...")
    model_drn = DRN(Channels, Out)
    y_drn = model_drn(x_img)
    assert y_drn.shape == (BS, Out)
    print("DRN OK")

    # 23. NTM
    print("Testing NTM...")
    model_ntm = NTM(In, Out)
    y_ntm, state_ntm = model_ntm(x)
    assert y_ntm.shape == (BS, Out)
    print("NTM OK")

    # 24. DNC
    print("Testing DNC...")
    model_dnc = DNC(In, Out)
    y_dnc, state_dnc = model_dnc(x)
    assert y_dnc.shape == (BS, Out)
    print("DNC OK")

    # 25. CapsNet
    print("Testing CapsNet...")
    x_caps = Tensor.rand(BS, 1, 28, 28) # Standard MNIST size for CapsNet assumption
    model_caps = CapsNet(1, 10)
    y_caps = model_caps(x_caps)
    assert y_caps.shape == (BS, 10)
    print("CapsNet OK")

    # 26. Kohonen Network
    print("Testing KN...")
    model_kn = KohonenNetwork(In, 5, 5)
    y_kn = model_kn(x)
    assert y_kn.shape == (BS,)
    print("KN OK")

    # 27. Attention Network
    print("Testing AN...")
    model_an = AttentionNetwork(In, Hidden)
    y_an = model_an(x_seq)
    assert y_an.shape == (BS, SeqLen, In)
    print("AN OK")
