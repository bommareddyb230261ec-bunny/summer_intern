import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCFACE_ROOT = PROJECT_ROOT.parent / "arcface"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ARCFACE_ROOT) not in sys.path:
    sys.path.insert(0, str(ARCFACE_ROOT))

from arcface.config import compute_video_hash, video_cache_paths


class PipelineCacheTests(unittest.TestCase):
    def test_compute_video_hash_uses_file_contents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            payload = b"video-bytes-123"
            video_path.write_bytes(payload)

            expected = hashlib.sha256(payload).hexdigest()
            actual = compute_video_hash(video_path)

            self.assertEqual(actual, expected)

    def test_video_cache_paths_use_per_video_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            video_path.write_bytes(b"abc")
            cache_paths = video_cache_paths(video_path)
            self.assertEqual(cache_paths["root"].name, compute_video_hash(video_path))


if __name__ == "__main__":
    unittest.main()
