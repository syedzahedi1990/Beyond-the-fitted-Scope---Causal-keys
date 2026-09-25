import ast
import io
import json
import tempfile
import tokenize
import unittest
from pathlib import Path

from reproduce.integrity import ROOT, digest, verify


class ReleaseTests(unittest.TestCase):
    def test_python_sources_contain_no_comments_or_docstrings(self):
        for folder in ("reproduce", "gpu", "tests"):
            for path in (ROOT / folder).rglob("*.py"):
                with self.subTest(path=str(path.relative_to(ROOT))):
                    text = path.read_text()
                    tree = ast.parse(text)
                    comments = [token for token in tokenize.generate_tokens(io.StringIO(text).readline)
                                if token.type == tokenize.COMMENT]
                    self.assertEqual(comments, [])
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.assertIsNone(ast.get_docstring(node))

    def test_integrity_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "evidence.json"
            file.write_text("[]\n")
            (root / "RELEASE.json").write_text(json.dumps({"files": {
                "evidence.json": {"bytes": file.stat().st_size, "sha256": digest(file)}}}))
            self.assertEqual(verify(root)["files_checked"], 1)
            file.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                verify(root)

    def test_integrity_rejects_parent_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "RELEASE.json").write_text(json.dumps({"files": {"../outside": {}}}))
            with self.assertRaisesRegex(ValueError, "Unsafe release path"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
