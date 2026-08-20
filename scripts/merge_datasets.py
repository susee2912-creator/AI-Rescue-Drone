import os
import shutil

SARD = "../dataset/search-and-rescue"
VIS = "../dataset/VisDroneHuman"
OUT = "../dataset/HumanDataset"

# SARD split -> VisDrone split -> Output split
splits = [
    ("train", "train", "train"),
    ("valid", "val", "val"),
    ("test", "test", "test")
]

for sard_split, vis_split, out_split in splits:

    out_images = os.path.join(OUT, out_split, "images")
    out_labels = os.path.join(OUT, out_split, "labels")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # ---------- SARD ----------
    sard_images = os.path.join(SARD, sard_split, "images")
    sard_labels = os.path.join(SARD, sard_split, "labels")

    for img in os.listdir(sard_images):

        shutil.copy(
            os.path.join(sard_images, img),
            os.path.join(out_images, "sard_" + img)
        )

        label = os.path.splitext(img)[0] + ".txt"

        if os.path.exists(os.path.join(sard_labels, label)):
            shutil.copy(
                os.path.join(sard_labels, label),
                os.path.join(out_labels, "sard_" + label)
            )

    # ---------- VisDrone ----------
    vis_images = os.path.join(VIS, vis_split, "images")
    vis_labels = os.path.join(VIS, vis_split, "labels")

    for img in os.listdir(vis_images):

        shutil.copy(
            os.path.join(vis_images, img),
            os.path.join(out_images, "vis_" + img)
        )

        label = os.path.splitext(img)[0] + ".txt"

        if os.path.exists(os.path.join(vis_labels, label)):
            shutil.copy(
                os.path.join(vis_labels, label),
                os.path.join(out_labels, "vis_" + label)
            )

print("Datasets merged successfully!")