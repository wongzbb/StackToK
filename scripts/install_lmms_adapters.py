from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"


def install() -> None:
    spec = importlib.util.find_spec("lmms_eval")
    if spec is None or spec.origin is None:
        raise RuntimeError("lmms_eval is not installed in the active environment.")
    pkg_dir = Path(spec.origin).resolve().parent
    models_dir = pkg_dir / "models"
    if not models_dir.exists():
        raise RuntimeError(f"lmms_eval models directory not found: {models_dir}")

    copies = {
        ADAPTERS / "llava_stacktok.py": models_dir / "llava_stacktok.py",
        ADAPTERS / "qwen2_5_vl_stacktok.py": models_dir / "qwen2_5_vl_stacktok.py",
    }
    for src, dst in copies.items():
        shutil.copy2(src, dst)

    init_file = models_dir / "__init__.py"
    marker_start = "# StackTok local adapters: start"
    marker_end = "# StackTok local adapters: end"
    block = (
        f"{marker_start}\n"
        "try:\n"
        "    from .registry_v2 import ModelManifest\n"
        "    MODEL_REGISTRY_V2.register_manifest(\n"
        "        ModelManifest(\n"
        "            model_id=\"llava_stacktok\",\n"
        "            simple_class_path=\"lmms_eval.models.llava_stacktok.Llava_StackTok\",\n"
        "        ),\n"
        "        overwrite=True,\n"
        "    )\n"
        "    MODEL_REGISTRY_V2.register_manifest(\n"
        "        ModelManifest(\n"
        "            model_id=\"qwen2_5_vl_stacktok\",\n"
        "            simple_class_path=\"lmms_eval.models.qwen2_5_vl_stacktok.Qwen2_5_VL_StackTok\",\n"
        "        ),\n"
        "        overwrite=True,\n"
        "    )\n"
        "    AVAILABLE_MODELS.update(\n"
        "        {\n"
        "            \"llava_stacktok\": \"Llava_StackTok\",\n"
        "            \"qwen2_5_vl_stacktok\": \"Qwen2_5_VL_StackTok\",\n"
        "        }\n"
        "    )\n"
        "except Exception as exc:\n"
        "    import warnings\n"
        "    warnings.warn(f\"Failed to register StackTok local adapters: {exc}\")\n"
        f"{marker_end}\n")
    text = init_file.read_text() if init_file.exists() else ""
    if marker_start in text and marker_end in text:
        pre = text.split(marker_start)[0].rstrip()
        post = text.split(marker_end, 1)[1].lstrip()
        text = f"{pre}\n{block}{post}"
    else:
        text = f"{text.rstrip()}\n\n{block}"
    init_file.write_text(text)
    print(f"Installed StackTok adapters into {models_dir}")


if __name__ == "__main__":
    install()
