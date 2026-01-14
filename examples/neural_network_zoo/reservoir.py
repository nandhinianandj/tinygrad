from tinygrad import Tensor, nn

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
