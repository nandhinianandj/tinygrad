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
