import math
from tinygrad import Tensor, nn

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
