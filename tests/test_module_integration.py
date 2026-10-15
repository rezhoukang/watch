"""Integration checks for HTTP assets and extracted dataset persistence."""
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app
import dataset_store


class ModuleIntegrationTests(unittest.TestCase):
    def test_static_module_graph_is_served_as_javascript(self):
        client = TestClient(app.app)
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn('type="module"', page.text)
        pending, visited = ["/app.js"], set()
        while pending:
            url = pending.pop()
            if url in visited:
                continue
            visited.add(url)
            response = client.get(url)
            self.assertEqual(response.status_code, 200, url)
            self.assertIn("javascript", response.headers["content-type"], url)
            for relative in re.findall(r"from\s+['\"]([^'\"]+)['\"]|import\s+['\"]([^'\"]+)['\"]", response.text):
                from urllib.parse import urljoin

                pending.append(urljoin(url, relative[0] or relative[1]))
        self.assertEqual(len(visited), 16)
        styles = client.get("/style.css")
        self.assertEqual(styles.status_code, 200)
        for relative in re.findall(r"url\('([^']+)'\)", styles.text):
            response = client.get("/" + relative.removeprefix("./"))
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/css", response.headers["content-type"])

    def test_labels_classes_and_cache_invalidation_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            datasets = Path(directory)
            ds = datasets / "课堂"
            images = ds / "images" / "train"
            images.mkdir(parents=True)
            image = images / "教室.png"
            Image.new("RGB", (100, 80)).save(image)
            with patch.object(dataset_store, "DATASETS_DIR", datasets):
                dataset_store.write_dataset_classes(ds, ["head"])
                self.assertEqual(dataset_store.resolve_annotate_image("课堂/train/教室.png"), image)
                count, label = dataset_store.save_labels_for_image(image, [{"class_id": 0, "bbox": [10, 20, 50, 60]}])
                self.assertEqual(count, 1)
                self.assertTrue((ds / label).is_file())
                self.assertEqual(dataset_store.read_labels_for_image(image)["detections"][0]["bbox"], [10, 20, 50, 60])
                cache = ds / "labels" / "train.cache"
                cache.write_bytes(b"old labels")
                dataset_store.write_dataset_classes(ds, ["人头"])
                self.assertFalse(cache.exists())
                self.assertEqual(dataset_store.read_labels_for_image(image)["detections"][0]["class_name"], "人头")
                detail = dataset_store.dataset_detail(ds)
                self.assertEqual(detail["subsets"]["train"]["labeled"], 1)
                self.assertEqual(dataset_store.list_datasets()[0]["images"], 1)


if __name__ == "__main__":
    unittest.main()
