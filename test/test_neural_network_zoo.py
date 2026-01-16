import unittest
from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist
import examples.neural_network_zoo as zoo

class TestNeuralNetworkZoo(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    # Load Fashion MNIST
    # X_train, Y_train, X_test, Y_test
    # Using a small subset for quick testing
    X_train, Y_train, _, _ = mnist(fashion=True)
    cls.X_train = X_train[:32].float() / 255.0
    cls.Y_train = Y_train[:32]
    cls.BS = 32
    cls.In = 28 * 28
    cls.Hidden = 64
    cls.Out = 10
    cls.SeqLen = 28 # Treat rows as sequence for RNNs
    cls.RowSize = 28

  def test_ffnn(self):
    model = zoo.FFNN(self.In, self.Hidden, self.Out)
    y = model(self.X_train.flatten(1))
    self.assertEqual(y.shape, (self.BS, self.Out))

  def test_rbf(self):
    model = zoo.RBF(self.In, self.Hidden, self.Out)
    y = model(self.X_train.flatten(1))
    self.assertEqual(y.shape, (self.BS, self.Out))

  def test_rnn(self):
    model = zoo.RNN(self.RowSize, self.Hidden, self.Out)
    # X: (BS, SeqLen, InputSize) -> (32, 28, 28)
    y = model(self.X_train.squeeze(1))
    self.assertEqual(y.shape, (self.BS, self.SeqLen, self.Out))

  def test_lstm(self):
    model = zoo.LSTM(self.RowSize, self.Hidden, self.Out)
    y = model(self.X_train.squeeze(1))
    self.assertEqual(y.shape, (self.BS, self.SeqLen, self.Out))

  def test_gru(self):
    model = zoo.GRU(self.RowSize, self.Hidden, self.Out)
    y = model(self.X_train.squeeze(1))
    self.assertEqual(y.shape, (self.BS, self.SeqLen, self.Out))

  def test_autoencoder(self):
    model = zoo.Autoencoder(self.In, self.Hidden)
    y = model(self.X_train.flatten(1))
    self.assertEqual(y.shape, (self.BS, self.In))

  def test_vae(self):
    Latent = 10
    model = zoo.VAE(self.In, self.Hidden, Latent)
    y, mu, logvar = model(self.X_train.flatten(1))
    self.assertEqual(y.shape, (self.BS, self.In))
    self.assertEqual(mu.shape, (self.BS, Latent))

  def test_dae(self):
    model = zoo.DAE(self.In, self.Hidden)
    y = model(self.X_train.flatten(1))
    self.assertEqual(y.shape, (self.BS, self.In))

  def test_sae(self):
    model = zoo.SAE(self.In, self.Hidden)
    decoded, encoded = model(self.X_train.flatten(1))
    self.assertEqual(decoded.shape, (self.BS, self.In))

  def test_mc(self):
    States = 5
    model = zoo.MarkovChain(States)
    state = Tensor.eye(States)[0].unsqueeze(0).expand(self.BS, States)
    y = model(state)
    self.assertEqual(y.shape, (self.BS, States))

  def test_hn(self):
    Neurons = 28 # Small HN
    model = zoo.HopfieldNetwork(Neurons)
    # Use small patterns
    patterns = (Tensor.rand(5, Neurons) > 0.5).where(1.0, -1.0)
    model.train(patterns)
    state = patterns[0:1]
    y = model(state)
    self.assertEqual(y.shape, (1, Neurons))

  def test_bm(self):
    model = zoo.BoltzmannMachine(10, 5) # Small BM
    v = (Tensor.rand(self.BS, 10) > 0.5).where(1.0, 0.0)
    y = model(v)
    self.assertEqual(y.shape, (self.BS, 15))

  def test_rbm(self):
    model = zoo.RBM(self.In, self.Hidden)
    v = (self.X_train.flatten(1) > 0.5).where(1.0, 0.0)
    v_probs, v_sample = model(v)
    self.assertEqual(v_sample.shape, (self.BS, self.In))

  def test_dbn(self):
    model = zoo.DBN([self.In, self.Hidden, self.Out])
    v = (self.X_train.flatten(1) > 0.5).where(1.0, 0.0)
    y = model(v)
    self.assertEqual(y.shape, (self.BS, self.Out))

  def test_cnn(self):
    model = zoo.CNN(1, self.Out)
    # CNN example assumes 8x8 input
    x_small = self.X_train.interpolate(size=(8, 8), mode="linear")
    y = model(x_small)
    self.assertEqual(y.shape, (self.BS, self.Out))

  def test_deconvnet(self):
    Latent = 10
    model = zoo.DeconvNet(Latent, 1) # 1 channel output
    z = Tensor.rand(self.BS, Latent)
    y = model(z)
    self.assertEqual(y.shape, (self.BS, 1, 8, 8)) # Check output size of DeconvNet implementation

  def test_dcign(self):
    Latent = 10
    model = zoo.DCIGN(1, Latent)
    # The DCIGN implementation expects specific sizes (8x8 input for simple case in example)
    # Input X_train is 28x28. The example implementation was hardcoded for 8x8 -> 4x4.
    # We should resize or adjust the test.
    x_small = self.X_train.interpolate(size=(8, 8), mode="linear")
    y, mu, logvar = model(x_small)
    self.assertEqual(y.shape, (self.BS, 1, 8, 8))

  def test_gan(self):
    Latent = 10
    model = zoo.GAN(Latent, 1)
    z = Tensor.rand(self.BS, Latent)
    fake_img, validity = model(z)
    # GAN generator outputs 8x8 based on DeconvNet
    self.assertEqual(fake_img.shape, (self.BS, 1, 8, 8))
    self.assertEqual(validity.shape, (self.BS, 1))

  def test_lsm(self):
    model = zoo.LSM(self.RowSize, self.Hidden, self.Out)
    y = model(self.X_train.squeeze(1))
    self.assertEqual(y.shape, (self.BS, self.Out))

  def test_elm(self):
    model = zoo.ELM(self.In, self.Hidden, self.Out)
    y = model(self.X_train.flatten(1))
    self.assertEqual(y.shape, (self.BS, self.Out))

  def test_esn(self):
    model = zoo.ESN(self.RowSize, self.Hidden, self.Out)
    y = model(self.X_train.squeeze(1))
    self.assertEqual(y.shape, (self.BS, self.Out))

  def test_drn(self):
    model = zoo.DRN(1, self.Out)
    # DRN implementation might expect specific strides.
    # Input 28x28.
    # conv1: 28x28
    # layer1: 28x28
    # layer2: 14x14
    # global avg pool: (BS, 32)
    # fc: (BS, Out)
    y = model(self.X_train)
    self.assertEqual(y.shape, (self.BS, self.Out))

  def test_ntm(self):
    model = zoo.NTM(self.RowSize, self.Out)
    # NTM takes sequence step by step or we iterate. The implementation returns output and state.
    # Let's run one step.
    x_step = self.X_train.squeeze(1)[:, 0, :] # First row
    y, state = model(x_step)
    self.assertEqual(y.shape, (self.BS, self.Out))

  def test_dnc(self):
    model = zoo.DNC(self.RowSize, self.Out)
    x_step = self.X_train.squeeze(1)[:, 0, :]
    y, state = model(x_step)
    self.assertEqual(y.shape, (self.BS, self.Out))

  def test_capsnet(self):
    model = zoo.CapsNet(1, 10) # 10 classes
    # CapsNet implementation assumes 28x28 input.
    y = model(self.X_train)
    self.assertEqual(y.shape, (self.BS, 10))

  def test_kn(self):
    model = zoo.KohonenNetwork(self.In, 5, 5)
    y = model(self.X_train.flatten(1))
    self.assertEqual(y.shape, (self.BS,))

  def test_an(self):
    model = zoo.AttentionNetwork(self.RowSize, self.Hidden)
    y = model(self.X_train.squeeze(1))
    self.assertEqual(y.shape, (self.BS, self.SeqLen, self.RowSize)) # Output dim matches input dim in this AN

if __name__ == '__main__':
  unittest.main()
