import unittest
import torch
from model import Config, TinyLM


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.model = TinyLM(Config(8, context=8, width=16, heads=2, layers=1, dropout=0.0))

    def test_future_tokens_cannot_change_past_predictions(self):
        self.model.eval()
        a = torch.tensor([[1, 2, 3, 4, 5]])
        b = torch.tensor([[1, 2, 3, 7, 6]])
        torch.testing.assert_close(self.model(a)[0][:, :3], self.model(b)[0][:, :3])

    def test_training_reduces_loss(self):
        x = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
        y = torch.tensor([[1, 2, 3, 0, 1, 2, 3, 0]])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        initial = self.model(x, y)[1].item()
        for _ in range(40):
            optimizer.zero_grad()
            loss = self.model(x, y)[1]
            loss.backward()
            optimizer.step()
        self.assertLess(self.model(x, y)[1].item(), initial * 0.3)

    def test_generation_beyond_context(self):
        prompt = torch.tensor([[1, 2, 3]])
        result = self.model.generate(prompt, count=12)
        self.assertEqual(tuple(result.shape), (1, 15))
        torch.testing.assert_close(result[:, :3], prompt)
        self.assertTrue(((result >= 0) & (result < 8)).all())


if __name__ == '__main__':
    unittest.main()
