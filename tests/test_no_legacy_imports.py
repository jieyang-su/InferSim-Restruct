import unittest
import ast
from pathlib import Path


class TestNoLegacyImports(unittest.TestCase):
    def test_ops_do_not_import_layers(self):
        repo_root = Path(__file__).resolve().parents[1]
        ops_dir = repo_root / "ops"
        for p in ops_dir.glob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(
                        node.module.startswith("layers"),
                        f"{p.name} imports legacy module: {node.module}",
                    )


if __name__ == "__main__":
    unittest.main()
