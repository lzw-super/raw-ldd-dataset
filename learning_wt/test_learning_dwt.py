"""CPU correctness tests. No denoising-quality or hardware performance claims."""
import unittest
import torch
from learning_wt.learning_dwt import LearningDWT, band_layout, dwt_atlas, iwt_atlas, soft_shrink


class LearningDWTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def test_ten_bands_cover_exactly_once(self):
        layout = band_layout(32, 48)
        self.assertEqual(len(layout), 10)
        count = torch.zeros(32, 48)
        for band in layout:
            count[band.rows, band.cols] += 1
        self.assertTrue(torch.all(count == 1))

    def test_inverse_and_energy(self):
        for h, w in [(8, 8), (16, 24), (32, 48)]:
            x = torch.randn(2, 4, h, w, dtype=torch.float64)
            z = dwt_atlas(x)
            torch.testing.assert_close(iwt_atlas(z), x, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(z.square().sum(), x.square().sum())

    def test_constant_ll_and_cfa_isolation(self):
        x = torch.zeros(1, 4, 16, 24, dtype=torch.float64)
        x[:, 2] = 0.125
        z = dwt_atlas(x)
        torch.testing.assert_close(z[:, 2, :2, :3], torch.ones(1, 2, 3, dtype=x.dtype))
        for band in band_layout(16, 24)[1:]:
            self.assertEqual(float(z[..., band.rows, band.cols].abs().max()), 0)
        self.assertEqual(float(z[:, [0, 1, 3]].abs().max()), 0)

    def test_autograd_transform(self):
        x = torch.randn(1, 1, 8, 8, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a: iwt_atlas(dwt_atlas(a)), (x,)))

    def test_shrink_zero_identity_and_sign(self):
        x = torch.randn(1, 4, 8, 8)
        torch.testing.assert_close(soft_shrink(x, torch.zeros_like(x)), x)
        z = soft_shrink(x, torch.full_like(x, 0.2))
        self.assertTrue(torch.all(z.abs() <= x.abs()+1e-7))
        self.assertTrue(torch.all(z*x >= 0))

    def test_full_model_backward_all_modes(self):
        for mode in ("atlas", "bandwise"):
            model = LearningDWT(wavelet="haar", width=8, depth=3, context=mode)
            x = torch.randn(2, 4, 16, 24) * 0.2
            y, aux = model(x, return_aux=True)
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(aux["threshold"].shape, x.shape)
            self.assertTrue(torch.all(aux["threshold"] >= 0))
            y.square().mean().backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(float(p.grad.abs().sum()), 0, name)

    def test_ll_switch_and_leak_identity(self):
        x = torch.randn(1, 4, 16, 24)
        model = LearningDWT(wavelet="haar", shrink_ll=False)
        _, aux = model(x, return_aux=True)
        ll = band_layout(16, 24)[0]
        self.assertEqual(float(aux["threshold"][..., ll.rows, ll.cols].max().detach()), 0)
        torch.testing.assert_close(aux["atlas"][..., ll.rows, ll.cols],
                                   aux["filtered_atlas"][..., ll.rows, ll.cols])
        torch.testing.assert_close(LearningDWT(wavelet="haar", leak=1.0)(x), x, atol=1e-6, rtol=1e-6)

    def test_optional_padding(self):
        x = torch.randn(1, 4, 13, 19)
        with self.assertRaises(ValueError):
            LearningDWT(wavelet="haar", )(x)
        y, aux = LearningDWT(wavelet="haar", pad_input=True, leak=1.0)(x, return_aux=True)
        self.assertEqual(aux["threshold"].shape, (1, 4, 16, 24))
        self.assertEqual(aux["padding"], (3, 5))
        torch.testing.assert_close(y, x, atol=1e-6, rtol=1e-6)

    def test_reject_wrong_input(self):
        for x in (torch.randn(1, 3, 16, 16), torch.zeros(1, 4, 16, 16, dtype=torch.int32)):
            with self.assertRaises(ValueError):
                LearningDWT(wavelet="haar", )(x)

    def test_aligned_halo_interior_both_modes(self):
        # For this Haar/local CNN configuration, enough crop also excludes
        # atlas-seam influence: it maps to boundaries of the original tile.
        x = torch.randn(1, 4, 160, 192)*0.2
        for context in ("atlas", "bandwise"):
            model = LearningDWT(wavelet="haar", context=context).eval()
            with torch.no_grad():
                full = model(x)
                tile = model(x[..., 32:128, 32:160])
            torch.testing.assert_close(tile[..., 40:-40, 40:-40],
                                       full[..., 72:88, 72:120],
                                       atol=2e-6, rtol=2e-6)

    def test_optimizer_can_reduce_fixed_toy_loss(self):
        # Deliberately zero target: verifies optimization, not realistic denoising.
        model = LearningDWT(wavelet="haar", width=4, depth=2, init_threshold=0.05)
        x = torch.randn(1, 4, 16, 16)*0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial = float(model(x).square().mean().detach())
        for _ in range(12):
            optimizer.zero_grad()
            loss = model(x).square().mean()
            loss.backward()
            optimizer.step()
        self.assertLess(float(model(x).square().mean().detach()), initial)


if __name__ == "__main__":
    unittest.main()
