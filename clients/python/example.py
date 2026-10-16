"""An example of driving eov from Python with eov_client.py (stdlib only).

Start eov, then run:

    python3 example.py [SLIDE] [OUTPUT-FOLDER]

It opens SLIDE (or uses the slide eov shows), prints what eov knows about it,
reads a tiled overview at the stored level closest to 1.25x-equivalent and
finds the darkest tile, goes there at 20x, bookmarks the place, measures
100 um, quantifies DAB in the region of interest if the slide has one, and
exports the view as a PNG into OUTPUT-FOLDER (default: the current folder).
"""

import os
import sys

from eov_client import Client, EovError

TILE = 512


def mean_brightness(rgb):
    """Mean of the channel values of raw RGB bytes."""
    return sum(rgb) / len(rgb) if rgb else 255.0


def main():
    slide_path = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    with Client() as eov:
        print("eov", eov.version()["eov"])
        if slide_path:
            opened = eov.open_slide(slide_path, view={"zoom": "fit"})
            slide_id, pane = opened["file_id"], opened["pane"]
        else:
            focused = [p for p in eov.panes() if p["focused"] and p["file_id"] is not None]
            if not focused:
                sys.exit("open a slide in eov, or name one: python3 example.py SLIDE")
            slide_id, pane = focused[0]["file_id"], focused[0]["pane"]
        slide = next(s for s in eov.slides() if s["file_id"] == slide_id)
        print("%s: %d x %d px, %d levels, pixel size %s, identity %s" % (
            slide["filename"], slide["width"], slide["height"], len(slide["levels"]),
            "%g um" % slide["mpp_x"] if slide["mpp_x"] else "unknown", slide["identity"]))

        # An overview in tiles, read at the stored level closest to 1.25x
        # (reading a level as stored is cheaper than resampling).
        wanted = 32.0
        if slide["native_magnification"]:
            wanted = max(1.0, slide["native_magnification"] / 1.25)
        level = max((l for l in slide["levels"] if l["downsample"] <= wanted),
                    key=lambda l: l["downsample"], default=slide["levels"][0])
        step = TILE * level["downsample"]
        darkest = None
        for y in range(0, slide["height"], int(step)):
            for x in range(0, slide["width"], int(step)):
                tile = eov.read_region(slide=slide_id, x=x, y=y, width=TILE, height=TILE,
                                       level=level["level"], format="rgb")
                brightness = mean_brightness(tile["data"])
                if darkest is None or brightness < darkest[0]:
                    darkest = (brightness, x + step / 2, y + step / 2)
        print("darkest tile (most tissue) around %.0f, %.0f" % darkest[1:])

        # Go there at 20x (or as close as the slide's pixel size allows).
        try:
            view = eov.set_view(pane=pane, x=darkest[1], y=darkest[2], magnification=20)
        except EovError as error:
            print("cannot zoom to 20x (%s); using a zoom factor" % error)
            view = eov.set_view(pane=pane, x=darkest[1], y=darkest[2], zoom=0.5)
        mark = eov.add_bookmark(name="Densest tissue (script)", slide=slide_id)
        print("bookmarked %r at %s" % (mark["name"], mark["magnification"] or view["zoom"]))

        # A 100 um scale line from the centre of the view, as a measurement.
        if slide["mpp_x"]:
            length_px = 100.0 / slide["mpp_x"]
            cx, cy = view["center_x"], view["center_y"]
            line = eov.add_measurement("line", points=[[cx, cy], [cx + length_px, cy]], slide=slide_id)
            print("measured", line["label"])

        if eov.roi(slide=slide_id):
            result = eov.quantify(slide=slide_id, model="h-dab", stain="DAB")
            print("DAB-positive: %.1f %% of %d tissue pixels, H-score %.0f" % (
                100 * result["positive_fraction"], result["analysed_pixels"], result["h_score"]))

        figure = os.path.join(out, os.path.splitext(slide["filename"])[0] + "-view.png")
        exported = eov.export_view(figure, pane=pane, dpi=150, scale_bar=True, overwrite=True)
        print("exported %d x %d px to %s" % (exported["width"], exported["height"], exported["path"]))


if __name__ == "__main__":
    try:
        main()
    except EovError as error:
        sys.exit("eov: %s" % error)
