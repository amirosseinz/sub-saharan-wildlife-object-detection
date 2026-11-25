import csv
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, PngImagePlugin

from audit_split import audit, pixel_digest


class SplitAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "images.csv"

    def write_manifest(self, rows):
        with self.manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["path", "split", "group"])
            writer.writerows(rows)

    def images(self, second_colour="red"):
        with Image.new("RGB", (4, 5), "red") as image:
            image.save(self.root / "first.png", compress_level=0)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("fixture", "Same pixels with different metadata and encoding")
        with Image.new("RGB", (4, 5), second_colour) as image:
            image.save(self.root / "second.png", compress_level=9, pnginfo=metadata)
        self.write_manifest([("first.png", "train", "camera-a"), ("second.png", "test", "camera-b")])

    def test_reencoded_duplicate_is_detected_only_with_pixel_check(self):
        self.images()
        original = audit(self.manifest)
        self.assertFalse(original["cross_split_image_hashes"])
        self.assertNotIn("cross_split_pixel_hashes", original)
        result = audit(self.manifest, decoded_pixels=True)
        self.assertEqual(list(result["cross_split_pixel_hashes"].values()), [["test", "train"]])
        self.assertFalse(result["cross_split_image_hashes"])

    def test_distinct_pixels_pass(self):
        self.images("blue")
        result = audit(self.manifest, decoded_pixels=True)
        self.assertFalse(result["cross_split_pixel_hashes"])

    def test_dimensions_are_part_of_pixel_identity(self):
        with Image.new("RGBA", (2, 6), "red") as image:
            image.save(self.root / "a.png")
        with Image.new("RGBA", (3, 4), "red") as image:
            image.save(self.root / "b.png")
        self.assertNotEqual(pixel_digest(self.root / "a.png"), pixel_digest(self.root / "b.png"))

    def test_transparency_is_preserved_and_opaque_rgb_matches_rgba(self):
        for name, mode, colour in (("rgb", "RGB", "red"), ("rgba", "RGBA", (255, 0, 0, 255)),
                                   ("transparent", "RGBA", (255, 0, 0, 0))):
            with Image.new(mode, (3, 3), colour) as image:
                image.save(self.root / f"{name}.png")
        self.assertEqual(pixel_digest(self.root / "rgb.png"), pixel_digest(self.root / "rgba.png"))
        self.assertNotEqual(pixel_digest(self.root / "rgba.png"), pixel_digest(self.root / "transparent.png"))

    def test_same_split_duplicates_do_not_report_cross_split_leakage(self):
        self.images()
        with Image.new("RGB", (2, 2), "blue") as image:
            image.save(self.root / "test.png")
        self.write_manifest([("first.png", "train", "a"), ("second.png", "train", "b"),
                             ("test.png", "test", "c")])
        self.assertFalse(audit(self.manifest, decoded_pixels=True)["cross_split_pixel_hashes"])

    def test_existing_group_and_byte_checks_still_apply(self):
        self.images()
        self.write_manifest([("first.png", "train", "a"), ("first.png", "test", "a")])
        result = audit(self.manifest, decoded_pixels=True)
        self.assertEqual(result["cross_split_groups"], {"a": ["test", "train"]})
        self.assertEqual(len(result["cross_split_image_hashes"]), 1)
        self.assertEqual(len(result["cross_split_pixel_hashes"]), 1)

    def test_cli_exit_codes_for_clean_leaking_and_corrupt_images(self):
        command = [sys.executable, str(Path(__file__).with_name("audit_split.py")),
                   str(self.manifest), "--decoded-pixels"]
        self.images("blue")
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.images()
        leaking = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(leaking.returncode, 1)
        self.assertIn("cross_split_pixel_hashes", leaking.stdout)
        (self.root / "second.png").write_bytes(b"not an image")
        invalid = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(invalid.stdout, "")

    def test_multiframe_input_is_rejected(self):
        with Image.new("RGB", (3, 3), "red") as first, Image.new("RGB", (3, 3), "blue") as second:
            first.save(self.root / "animated.gif", save_all=True, append_images=[second])
        with self.assertRaisesRegex(ValueError, "single-frame"):
            pixel_digest(self.root / "animated.gif")


if __name__ == "__main__":
    unittest.main()
