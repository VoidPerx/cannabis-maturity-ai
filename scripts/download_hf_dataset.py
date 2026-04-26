"""
Descarga el dataset de trichomas desde HuggingFace:
  siccan/tricomas-semillero-cannabis
y lo organiza en data/processed/trichome/ listo para YOLOv8.

Uso: python scripts/download_hf_dataset.py
"""
import sys
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    print("Descargando siccan/tricomas-semillero-cannabis desde HuggingFace...")
    ds = load_dataset("siccan/tricomas-semillero-cannabis")
    print(f"Splits disponibles: {list(ds.keys())}")

    out_root = root / "data" / "processed" / "trichome_hf"
    out_root.mkdir(parents=True, exist_ok=True)

    for split, dataset in ds.items():
        split_dir = out_root / split / "images"
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Guardando {len(dataset)} imagenes en {split_dir} ...")

        for i, item in enumerate(dataset):
            # Guardar imagen
            if "image" in item and item["image"] is not None:
                img = item["image"]
                img_path = split_dir / f"{i:06d}.jpg"
                img.save(str(img_path))

            # Guardar label si existe
            if "label" in item or "annotations" in item:
                labels_dir = out_root / split / "labels"
                labels_dir.mkdir(exist_ok=True)
                label_val = item.get("label", item.get("annotations", ""))
                (labels_dir / f"{i:06d}.txt").write_text(str(label_val))

    print(f"\n[OK] Dataset guardado en {out_root}")
    print("Puedes usarlo directamente para entrenamiento YOLOv8.")

if __name__ == "__main__":
    main()
