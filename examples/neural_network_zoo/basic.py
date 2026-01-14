from tinygrad import Tensor, nn

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
