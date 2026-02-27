import unittest

from comm.estimator import CommContext, CommEstimator
from hardware.gpu import gpu_map


class TestCommRegistry(unittest.TestCase):
    def test_register_new_op_does_not_require_engine_changes(self):
        ctx = CommContext(
            world_size=8,
            num_nodes=1,
            gpu=gpu_map["H20"],
            hidden_size=4096,
            num_experts_per_tok=2,
            enable_deepep=False,
        )
        est = CommEstimator(ctx)

        called = {"n": 0}

        def my_allgather(tensor, mode=None):
            called["n"] += 1
            return 0.123

        est.register_op("allgather", my_allgather)

        t = est.estimate("allgather", tensor=[128, 4096])
        self.assertAlmostEqual(t, 0.123)
        self.assertEqual(called["n"], 1)


if __name__ == "__main__":
    unittest.main()