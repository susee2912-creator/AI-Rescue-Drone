import os
import shutil
from PIL import Image

# Dataset path
BASE = "../dataset/VisDrone2019"

# Output dataset
OUTPUT = "../dataset/VisDroneHuman"

# Classes to keep
KEEP_CLASSES = [1, 2]      # pedestrian, people


def convert_split(split):

    image_folder = os.path.join(BASE, split, "images")
    ann_folder = os.path.join(BASE, split, "annotations")

    out_img = os.path.join(OUTPUT, split, "images")
    out_lbl = os.path.join(OUTPUT, split, "labels")

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    image_list = os.listdir(image_folder)

    count = 0

    for img_name in image_list:

        txt_name = img_name.replace(".jpg", ".txt")

        ann_path = os.path.join(ann_folder, txt_name)

        if not os.path.exists(ann_path):
            continue

        img_path = os.path.join(image_folder, img_name)

        img = Image.open(img_path)
        w, h = img.size

        yolo_lines = []

        with open(ann_path, "r") as f:

            for line in f:

                data = line.strip().split(",")

                if len(data) < 8:
                    continue

                x = float(data[0])
                y = float(data[1])
                bw = float(data[2])
                bh = float(data[3])

                cls = int(data[5])

                if cls not in KEEP_CLASSES:
                    continue

                xc = (x + bw/2)/w
                yc = (y + bh/2)/h
                bw /= w
                bh /= h

                yolo_lines.append(
                    f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )

        if len(yolo_lines) == 0:
            continue

        shutil.copy(img_path, os.path.join(out_img, img_name))

        with open(os.path.join(out_lbl, txt_name), "w") as out:

            out.write("\n".join(yolo_lines))

        count += 1

    print(f"{split}: {count} human images copied")


convert_split("train")
convert_split("val")
convert_split("test")

print("\nFinished!")