
# Creates the data folder skeleton
from pathlib import Path

root = Path("data")
for split in ["train", "val"]:
    for cls in ["Healthy", "Diseased"]:
        d = root / split / cls
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").write_text("")
print("Created data skeleton under ./data")
