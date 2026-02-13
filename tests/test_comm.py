import unittest
#python3 -m unittest -q
from hardware.gpu import gpu_map
from comm.estimator import CommContext, CommEstimator


class TestCommEstimator(unittest.TestCase):
    def test_world_size_1_always_zero(self):
        ctx = CommContext(
            world_size=1,
            num_nodes=1,
            gpu=gpu_map["H800"],
            hidden_size=4096,
            num_experts_per_tok=2,
            enable_deepep=True,
        )
        est = CommEstimator(ctx)
        self.assertEqual(est.estimate("allreduce", [128, 4096]), 0.0)
        self.assertEqual(est.estimate(None, [128, 4096], stage="prefill"), 0.0)
        self.assertEqual(est.estimate(None, [1, 4096], stage="decode"), 0.0)

    def test_stage_dispatch(self):
        ctx = CommContext(
            world_size=8,
            num_nodes=2,
            gpu=gpu_map["H800"],
            hidden_size=4096,
            num_experts_per_tok=2,
            enable_deepep=True,
        )
        est = CommEstimator(ctx)
        prefill = est.estimate(None, [128, 4096], stage="prefill")
        decode = est.estimate(None, [128, 4096], stage="decode")
        self.assertGreater(prefill, 0.0)
        self.assertGreater(decode, 0.0)
        # decode uses low_latency which usually sends more tokens than normal on multi-node
        # so we only assert it's non-negative and not equal by default
        self.assertNotEqual(prefill, decode)


if __name__ == "__main__":
    unittest.main()
