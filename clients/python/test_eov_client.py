"""Tests of eov_client.py.

The offline tests always run (``python3 test_eov_client.py``). The live test
talks to a running eov; eov's own test suite runs it against a test server
(``scripting::tests`` in app/src/scripting/tests.rs), passing:

- ``EOV_TEST_INFO``: the endpoint's ``.info`` file;
- ``EOV_TEST_SLIDE``: a 1024 x 512 slide (0.25 um/px) open in the window, its
  left half DAB-positive;
- ``EOV_TEST_OUT``: a folder for exports;
- ``EOV_TEST_IDLE``: seconds after which eov hangs up on an idle connection;
- ``EOV_TEST_HOME``, ``EOV_TEST_CONFIG``, ``EOV_TEST_PLUGINS`` and
  ``EOV_TEST_NAME``: a setup and the endpoint name eov computes for it.
"""

import os
import struct
import sys
import tempfile
import time
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eov_client  # noqa: E402
from eov_client import Client, Endpoint, EovError  # noqa: E402


def png_size(data):
    """Width and height of a PNG file, checking its signature and IHDR CRC."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    length, kind = struct.unpack(">I4s", data[8:16])
    assert kind == b"IHDR" and length == 13
    header = data[16:29]
    (crc,) = struct.unpack(">I", data[29:33])
    assert zlib.crc32(kind + header) == crc, "bad IHDR CRC"
    return struct.unpack(">II", header[:8])


class Offline(unittest.TestCase):
    def test_paths_are_absolute_and_survive_odd_names(self):
        params = eov_client._path_params("slide.svs")
        self.assertEqual(params, {"path": os.path.abspath("slide.svs")})
        if os.name == "posix":
            odd = os.fsdecode(b"/slides/\xff.svs")
            self.assertEqual(eov_client._path_params(odd), {"os_path": b"/slides/\xff.svs".hex()})

    def test_parameters_leave_out_what_is_not_given(self):
        self.assertEqual(eov_client._params(a=1, b=None, c=False), {"a": 1, "c": False})

    def test_info_files_are_read(self):
        with tempfile.TemporaryDirectory() as folder:
            info = os.path.join(folder, "eov-0123456789abcdef.info")
            with open(info, "w") as file:
                file.write("0 4242 %s\n" % ("ab" * 32))
            endpoint = Endpoint(info)
            self.assertEqual((endpoint.port, endpoint.pid), (0, 4242))
            self.assertEqual(endpoint.secret, b"\xab" * 32)
            self.assertEqual(endpoint.socket_path, os.path.join(folder, "eov-0123456789abcdef.sock"))
            self.assertFalse(endpoint.answers(timeout=0.2), "a crashed eov's leftovers")
            with open(info, "w") as file:
                file.write("51234 1 %s\n" % ("cd" * 32))
            self.assertIsNone(Endpoint(info).socket_path)
            with open(info, "w") as file:
                file.write("garbage")
            with self.assertRaises(EovError) as raised:
                Endpoint(info)
            self.assertEqual(raised.exception.kind, "not_running")

    def test_names_hash_the_setup(self):
        a = eov_client.endpoint_name("/h", "/h/config.toml", "/h/plugins")
        self.assertRegex(a, r"^eov-[0-9a-f]{16}$")
        self.assertNotEqual(a, eov_client.endpoint_name("/h", "/h/other.toml", "/h/plugins"))

    def test_the_runtime_folder_can_be_given(self):
        saved = os.environ.get("EOV_RUNTIME_DIR")
        try:
            os.environ["EOV_RUNTIME_DIR"] = os.path.abspath("run-here")
            self.assertEqual(eov_client.runtime_dirs(), [os.path.abspath("run-here")])
        finally:
            if saved is None:
                del os.environ["EOV_RUNTIME_DIR"]
            else:
                os.environ["EOV_RUNTIME_DIR"] = saved

    @unittest.skipUnless(os.environ.get("EOV_TEST_NAME"), "eov gives the expected name")
    def test_names_match_eov(self):
        name = eov_client.endpoint_name(
            os.environ["EOV_TEST_HOME"], os.environ["EOV_TEST_CONFIG"], os.environ["EOV_TEST_PLUGINS"]
        )
        self.assertEqual(name, os.environ["EOV_TEST_NAME"])


@unittest.skipUnless(os.environ.get("EOV_TEST_INFO"), "needs a running eov (EOV_TEST_INFO)")
class Live(unittest.TestCase):
    def test_a_script_drives_eov(self):
        slide_path = os.environ["EOV_TEST_SLIDE"]
        out = os.environ["EOV_TEST_OUT"]
        with Client(info=os.environ["EOV_TEST_INFO"], timeout=60) as eov:
            version = eov.version()
            self.assertEqual(version["protocol"], 1)
            self.assertIn("region.read", version["methods"])

            slides = eov.slides()
            mine = [s for s in slides if s["path"] == slide_path]
            self.assertEqual(len(mine), 1, slides)
            slide = mine[0]
            self.assertEqual((slide["width"], slide["height"]), (1024, 512))
            self.assertAlmostEqual(slide["mpp_x"], 0.25, places=6)
            self.assertTrue(slide["identity"], "the identity key is known")
            file_id = slide["file_id"]

            panes = eov.panes()
            shown = [p for p in panes if p["file_id"] == file_id]
            self.assertTrue(shown, panes)
            pane = shown[0]["pane"]

            view = eov.set_view(pane=pane, x="128um", y="64um", magnification=20)
            self.assertAlmostEqual(view["center_x"], 512, delta=1)
            self.assertAlmostEqual(view["center_y"], 256, delta=1)
            self.assertAlmostEqual(view["magnification"], 20, places=6)
            self.assertEqual(eov.view(pane=pane)["zoom"], view["zoom"])

            # Pixels: the left half is DAB-positive, the right half is not.
            left = eov.read_region(slide=file_id, x=0, y=0, width=8, height=4, format="rgba")
            right = eov.read_region(path=slide_path, x=1000, y=0, width=8, height=4, format="rgb")
            self.assertEqual(len(left["data"]), 8 * 4 * 4)
            self.assertEqual(len(right["data"]), 8 * 4 * 3)
            self.assertNotEqual(left["data"][:3], right["data"][:3])
            self.assertEqual(left["data"][3], 255)
            tile = eov.read_region(slide=file_id, x=0, y=0, width=100, height=50, downsample=3)
            self.assertEqual(png_size(tile["data"]), (100, 50))
            self.assertEqual((tile["level"], tile["downsample"]), (1, 3))
            self.assertAlmostEqual(tile["mpp"], 0.75, places=6)
            with self.assertRaises(EovError) as raised:
                eov.read_region(slide=file_id, x=0, y=0, width=20000, height=20000)
            self.assertEqual(raised.exception.kind, "too_large")

            # The region of interest, measurements and bookmarks.
            self.assertEqual(eov.set_roi(256, 0, 512, 256, slide=file_id), {"x": 256, "y": 0, "width": 512, "height": 256})
            self.assertEqual(eov.roi(slide=file_id)["width"], 512)
            line = eov.add_measurement("line", points=[[0, 0], [0, 400]], slide=file_id)
            self.assertAlmostEqual(line["length_um"], 100, places=6)
            self.assertIn(line["measurement_id"], [m["measurement_id"] for m in eov.measurements(slide=file_id)])
            removed = eov.remove_measurements(ids=[line["measurement_id"]], slide=file_id)
            self.assertEqual(removed["removed"], [line["measurement_id"]])
            mark = eov.add_bookmark(name="From Python", slide=file_id, x=100, y=200, zoom=1)
            listed = eov.bookmarks(slide=file_id)
            self.assertEqual(listed[mark["index"]]["name"], "From Python")
            self.assertAlmostEqual(listed[mark["index"]]["magnification"], 40, places=6)

            # Analysis.
            quantified = eov.quantify(slide=file_id, model="h-dab", stain="DAB")
            self.assertEqual(quantified["analysed_pixels"], 512 * 256)
            self.assertAlmostEqual(quantified["positive_fraction"], 0.5, delta=0.02)
            self.assertAlmostEqual(quantified["h_score"], 100, delta=2)
            figure = os.path.join(out, "python-figure.jpg")
            exported = eov.export_view(figure, pane=pane, dpi=96, jpeg_quality=80)
            self.assertTrue(os.path.isfile(figure))
            self.assertEqual(exported["format"], "jpeg")
            with self.assertRaises(EovError) as raised:
                eov.export_view(figure, pane=pane)
            self.assertEqual(raised.exception.kind, "refused")

            # Errors name their kind.
            for call, kind in [
                (lambda: eov.call("no.such.method"), "method_not_found"),
                (lambda: eov.call("view.get", {"pane": 424242}), "not_found"),
                (lambda: eov.call("view.set", {"zom": 2}), "invalid_params"),
                (lambda: eov.open_slide(os.path.join(out, "missing.svs")), "not_found"),
            ]:
                with self.assertRaises(EovError) as raised:
                    call()
                self.assertEqual(raised.exception.kind, kind)

            # eov hangs up on an idle connection; the client reconnects.
            idle = float(os.environ.get("EOV_TEST_IDLE", "0"))
            if idle:
                time.sleep(idle + 0.5)
                self.assertEqual(eov.version()["protocol"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
