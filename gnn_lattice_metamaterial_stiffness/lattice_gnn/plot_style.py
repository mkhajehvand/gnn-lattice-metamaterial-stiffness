# shared plotting style with a redistributable fallback font

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
from matplotlib import font_manager


FONT_FAMILY = "Helvetica Neue"
FALLBACK_FONT_FAMILY = "DejaVu Sans"
FONT_COLLECTION_RELATIVE_PATH = Path("fonts") / "HelveticaNeue.ttc"
FONT_FACE_INDICES = {
    "regular": 0,
    "bold": 1,
    "italic": 2,
    "bold_italic": 3,
}
_CONFIGURED = False


# apply the figure style with a portable font fallback
def configure_matplotlib() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    package_dir = Path(__file__).resolve().parent.parent
    font_path = package_dir / FONT_COLLECTION_RELATIVE_PATH
    selected_family = FALLBACK_FONT_FAMILY
    if font_path.is_file():
        from fontTools.ttLib import TTCollection

        digest = hashlib.sha256(font_path.read_bytes()).hexdigest()[:12]
        face_dir = Path(matplotlib.get_cachedir()) / f"helvetica_neue_{digest}"
        face_dir.mkdir(parents=True, exist_ok=True)
        face_paths = {
            face_name: face_dir / f"HelveticaNeue-{face_name}.ttf"
            for face_name in FONT_FACE_INDICES
        }
        if not all(path.is_file() for path in face_paths.values()):
            collection = TTCollection(str(font_path))
            try:
                for face_name, face_index in FONT_FACE_INDICES.items():
                    collection.fonts[face_index].save(str(face_paths[face_name]))
            finally:
                collection.close()

        # remove system copies so every style resolves to the bundled ttc
        font_manager.fontManager.ttflist[:] = [
            entry
            for entry in font_manager.fontManager.ttflist
            if entry.name != FONT_FAMILY
        ]
        for face_path in face_paths.values():
            font_manager.fontManager.addfont(str(face_path))
        font_manager.fontManager._findfont_cached.cache_clear()
        selected_family = FONT_FAMILY
    else:
        print(
            "[plot] bundled Helvetica Neue was not found; "
            f"using {FALLBACK_FONT_FAMILY}"
        )

    matplotlib.rcParams.update(
        {
            "font.family": selected_family,
            "font.sans-serif": [selected_family],
            "mathtext.fontset": "custom",
            "mathtext.rm": selected_family,
            "mathtext.it": f"{selected_family}:italic",
            "mathtext.bf": f"{selected_family}:bold",
            "mathtext.cal": f"{selected_family}:italic",
            "mathtext.sf": selected_family,
            "mathtext.tt": selected_family,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.format": "pdf",
        }
    )
    _CONFIGURED = True
