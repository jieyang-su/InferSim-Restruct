import os
import unittest


from config.model_config import ModelConfig
from engine.sim_engine import SimConfig, SimulationEngine


class TestEngineSmoke(unittest.TestCase):
    def test_engine_runs_and_comm_zero_world_size_1(self):
        cfg_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "hf_configs",
            "qwen3-8B_config.json",
        )
        cfg_path = os.path.abspath(cfg_path)
        config = ModelConfig(cfg_path)

        class _Args:
            # minimal arg shim
            device_type = "H20"
            world_size = 1
            num_nodes = 1
            max_prefill_tokens = 256
            target_isl = 256
            target_osl = 128
            target_tgs = 2560
            target_tpot = 50
            decode_bs = 1
            use_fp8_gemm = False
            use_fp8_kv = False
            enable_deepep = True
            enable_tbo = False
            sm_ratio = 108 / 132
            prefill_only = False
            decode_only = False

        sim = SimConfig.from_args(_Args)
        engine = SimulationEngine(sim, config)
        payload = engine.run(cfg_path)

        self.assertIn("prefill", payload)
        self.assertIn("decode", payload)

        # With world_size=1, comm time must be 0 (explicit downgrade behavior).
        prefill_br = payload["prefill"]["breakdown"]
        self.assertEqual(prefill_br["comm_before_us"], 0.0)
        self.assertEqual(prefill_br["comm_after_us"], 0.0)


if __name__ == "__main__":
    unittest.main()
