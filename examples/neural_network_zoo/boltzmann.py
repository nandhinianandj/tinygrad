from tinygrad import Tensor, nn

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
