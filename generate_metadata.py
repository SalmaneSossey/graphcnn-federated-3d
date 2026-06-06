import zipfile
import pandas as pd
from pathlib import Path


synset_dict = {
    "04379243": "table",
    "02958343": "car",
    "03001627": "chair",
    "02871439": "bookshelf",
    "02691156": "airplane",
    "03642806": "laptop",
    "04256520": "sofa",
    "03624134": "knife",
    "04090263": "rifle",
    "04468005": "train",
    "03636649": "lamp",
    "02747177": "trash bin",
    "04530566": "watercraft",
    "03790512": "motorbike",
    "02828884": "bench",
    "03948459": "pistol",
    "04099429": "rocket",
    "03691459": "loudspeaker",
    "02933112": "cabinet",
    "02818832": "bed",
    "03211117": "display",
    "03928116": "piano",
    "03261776": "earphone",
    "04401088": "telephone",
    "04330267": "stove",
    "03759954": "microphone",
    "02924116": "bus",
    "03797390": "mug",
    "04074963": "remote",
    "02880940": "bowl",
    "03085013": "keyboard",
    "03467517": "guitar",
    "04554684": "washer",
    "02834778": "bicycle",
    "03325088": "faucet",
    "04004475": "printer",
    "02954340": "cap",
}

zip_path = Path("data/raw/3dData.zip")
records = []

if not zip_path.exists():
    raise FileNotFoundError(f"Ensure your ZIP file has been moved to: {zip_path}")

print("Scanning 3dData.zip archive to generate metadata...")
with zipfile.ZipFile(zip_path, "r") as archive:
    for name in archive.namelist():
        if name.endswith("/") or not name.lower().endswith(".ply"):
            continue

        stem = Path(name).stem
        parts = stem.split("_")
        if len(parts) >= 2:
            synset_id = parts[0]
            label_name = synset_dict.get(synset_id)
            if label_name:
                records.append({"id": stem, "label": label_name})

df = pd.DataFrame(records)
output_dir = Path("data/metadata")
output_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(output_dir / "labeled_dataset.csv", index=False)

print(
    f"Successfully generated {output_dir}/labeled_dataset.csv with {len(df)} mapped items."
)
