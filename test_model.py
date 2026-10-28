"""小型CPU结构检查：数学参照、mask、残差、两种底物输入、梯度和回载。

仅使用合成数据；没有优化器、参数更新、真实标签或正式训练。
运行：python -m unittest -v test_model
"""

import io
import unittest

import torch
from torch import nn

from example_forward import build_review_example, synthetic_inputs
from model import AttentionConfig


class ModelLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(668)
        self.model = build_review_example().eval()
        self.inputs = synthetic_inputs()

    def test_matches_pytorch_multihead_attention(self):
        """用PyTorch官方MultiheadAttention核对展开计算，检查Q/K/V及归一化轴。"""
        layer = self.model.interaction
        p, pm, m, mm, c = self.inputs
        result = layer(*self.inputs, return_attention=True)
        memory = torch.cat([
            layer.molt5_projection(m.masked_fill(~mm.unsqueeze(-1), 0)),
            layer.maccs_projection(c).unsqueeze(1),
        ], dim=1)
        reference = nn.MultiheadAttention(1024, 8, dropout=0.0, batch_first=True).eval()
        with torch.no_grad():
            reference.in_proj_weight.copy_(torch.cat([
                layer.query_projection.weight, layer.key_projection.weight, layer.value_projection.weight,
            ], dim=0))
            reference.in_proj_bias.copy_(torch.cat([
                layer.query_projection.bias, layer.key_projection.bias, layer.value_projection.bias,
            ], dim=0))
            reference.out_proj.load_state_dict(layer.output_projection.state_dict())
        delta, weights = reference(
            p.masked_fill(~pm.unsqueeze(-1), 0), memory, memory,
            key_padding_mask=~result.substrate_mask,
            need_weights=True, average_attn_weights=False,
        )
        delta = delta.masked_fill(~pm.unsqueeze(-1), 0)
        weights = weights.masked_fill(~pm[:, None, :, None], 0)
        torch.testing.assert_close(result.delta, delta, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(result.attention, weights, rtol=1e-5, atol=1e-6)
        self.assertEqual(result.residues.shape, (2, 5, 1024))
        torch.testing.assert_close(result.residues, p.masked_fill(~pm.unsqueeze(-1), 0) + delta)

    def test_padding_and_nan_padding_do_not_change_valid_predictions(self):
        """增加padding且写入NaN，真实残基输出及最终预测仍不变。"""
        p, pm, m, mm, c = self.inputs
        old = self.model(*self.inputs, return_attention=True)
        p = p.masked_fill(~pm.unsqueeze(-1), float("nan"))
        m = m.masked_fill(~mm.unsqueeze(-1), float("nan"))
        p = torch.cat([p, torch.full((2, 3, 1024), float("nan"))], dim=1)
        pm = torch.cat([pm, torch.zeros(2, 3, dtype=torch.bool)], dim=1)
        m = torch.cat([m, torch.full((2, 2, 768), float("nan"))], dim=1)
        mm = torch.cat([mm, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
        new = self.model(p, pm, m, mm, c, return_attention=True)
        torch.testing.assert_close(new["interaction"].residues[:, :5], old["interaction"].residues)
        torch.testing.assert_close(new["pred"], old["pred"])
        self.assertEqual(torch.count_nonzero(new["interaction"].residues[~pm]).item(), 0)
        weights = new["interaction"].attention
        invalid_keys = (~new["interaction"].substrate_mask)[:, None, None, :].expand_as(weights)
        self.assertEqual(torch.count_nonzero(weights[invalid_keys]).item(), 0)
        torch.testing.assert_close(weights.sum(-1), pm[:, None, :].expand(-1, 8, -1).float())

    def test_raw_residual_is_exact_when_update_is_zero(self):
        """修正为零时必须直接回到原始T，不能暗中归一化或池化残差。"""
        with torch.no_grad():
            self.model.interaction.output_projection.weight.zero_()
            self.model.interaction.output_projection.bias.zero_()
        result = self.model.interaction(*self.inputs)
        expected = self.inputs[0].masked_fill(~self.inputs[1].unsqueeze(-1), 0)
        self.assertTrue(torch.equal(result.residues, expected))
        self.assertEqual(torch.count_nonzero(result.delta).item(), 0)

    def test_each_substrate_modality_and_each_residue_can_affect_interaction(self):
        layer = self.model.interaction
        p, pm, m, mm, c = self.inputs
        base = layer(*self.inputs, return_attention=True)
        # 改变有效MolT5内容；再独立改变MACCS，均应影响残基更新。
        changed_m = m.clone()
        changed_m[:, 0] += 1.0
        self.assertFalse(torch.allclose(base.delta, layer(p, pm, changed_m, mm, c).delta))
        self.assertFalse(torch.allclose(base.delta, layer(p, pm, m, mm, 1.0 - c).delta))
        # 同一蛋白的不同残基有不同Q，而非把同一全局更新广播L次。
        self.assertFalse(torch.allclose(base.attention[0, :, 0], base.attention[0, :, 1]))

    def test_permutations_preserve_residue_alignment(self):
        """固定编码后的token：交换蛋白位置应同步交换输出；memory置换不改输出。"""
        layer = self.model.interaction
        p, pm, m, mm, c = self.inputs
        old = layer(*self.inputs)
        protein_order = torch.tensor([2, 0, 4, 1, 3])
        molecule_order = torch.tensor([3, 1, 0, 2])
        new = layer(p[:, protein_order], pm[:, protein_order],
                    m[:, molecule_order], mm[:, molecule_order], c)
        torch.testing.assert_close(new.residues, old.residues[:, protein_order])

    def test_batch_rows_do_not_mix_in_eval(self):
        batch = self.model(*self.inputs)["pred"]
        singles = torch.cat([self.model(*(x[i:i+1] for x in self.inputs))["pred"] for i in range(2)])
        torch.testing.assert_close(batch, singles, rtol=1e-5, atol=1e-6)

    def test_gradients_are_finite_and_padding_has_zero_gradient(self):
        # 只反向传播一次检查计算图；没有optimizer.step，不做训练拟合。
        p, pm, m, mm, c = self.inputs
        p = p.masked_fill(~pm.unsqueeze(-1), float("nan")).requires_grad_()
        m = m.masked_fill(~mm.unsqueeze(-1), float("nan")).requires_grad_()
        c = c.requires_grad_()
        self.model.train()
        self.model(p, pm, m, mm, c)["pred"].square().sum().backward()
        for name, parameter in self.model.named_parameters():
            # Key bias在softmax中可被抵消，不能要求每个参数梯度都严格非零。
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        for tensor in (p, m, c):
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(tensor.grad.abs().sum().item(), 0.0)
        self.assertEqual(torch.count_nonzero(p.grad[~pm]).item(), 0)
        self.assertEqual(torch.count_nonzero(m.grad[~mm]).item(), 0)

    def test_state_round_trip(self):
        # state_dict同时含权重和BN运行统计；内存保存，不产生模型检查点文件。
        self.model.train()
        self.model(*self.inputs)
        self.model.eval()
        before = self.model(*self.inputs)["pred"]
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = build_review_example().eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        after = restored(*self.inputs)["pred"]
        self.assertTrue(torch.equal(before, after))

    def test_invalid_inputs_fail_instead_of_silent_repair(self):
        p, pm, m, mm, c = self.inputs
        invalid = [
            (p.mean(1), pm, m, mm, c),                  # 全局蛋白不是逐残基输入
            (p, pm, m.mean(1), mm, c),                  # 全局MolT5不是token输入
            (p, torch.zeros_like(pm), m, mm, c),       # 空蛋白
            (p, pm, m, torch.zeros_like(mm), c),       # 不静默省略SMILES输入
            (p, pm.float(), m, mm, c),                 # mask必须为布尔值
            (p, pm, m.double(), mm, c),                # 不隐式改变特征精度
            (p, pm, m, mm, c[:, :166]),                # 指纹维度错误
        ]
        for inputs in invalid:
            with self.subTest(shapes=[tuple(x.shape) for x in inputs]):
                with self.assertRaises((ValueError, TypeError)):
                    self.model(*inputs)
        bad = p.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.model(bad, pm, m, mm, c)
        with self.assertRaises(ValueError):
            AttentionConfig(3, 0.0, 0.0, True)
        self.model.train()
        with self.assertRaisesRegex(ValueError, "B>=2"):
            self.model(*(x[:1] for x in self.inputs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
