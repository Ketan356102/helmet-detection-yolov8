import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'images')
ANN_DIR = os.path.join(BASE_DIR, 'annotations')

OUT_IMG_TRAIN = os.path.join(BASE_DIR, 'dataset/images/train')
OUT_IMG_VAL = os.path.join(BASE_DIR, 'dataset/images/val')
OUT_LABEL_TRAIN = os.path.join(BASE_DIR, 'dataset/labels/train')
OUT_LABEL_VAL = os.path.join(BASE_DIR, 'dataset/labels/val')

CLASSES = ['with_helmet', 'without_helmet']

for folder in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LABEL_TRAIN, OUT_LABEL_VAL]:
    os.makedirs(folder, exist_ok=True)

images_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
train_files, val_files = train_test_split(images_files, test_size=0.2, random_state=42)

def convert_bbox(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh

def convert_annotation(img_file, mode):
    name,_ = os.path.splitext(img_file)
    xml_file = os.path.join(ANN_DIR, f'{name}.xml')
    if not os.path.exists(xml_file): return

    out_label_path = os.path.join(
        OUT_LABEL_TRAIN if mode == 'train' else OUT_LABEL_VAL, f'{name}.txt'
    )

    with open(out_label_path, 'w') as out_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        if size is None: return

        w, h = int(size.find('width').text), int(size.find('height').text)

        for obj in root.findall('object'):
            cls = obj.find('name').text.strip().lower()
            cls = cls.replace("without helmet", "without_helmet").replace("with helmet", "with_helmet")

            if cls not in CLASSES:
                continue

            cls_id = CLASSES.index(cls)
            xmlbox = obj.find('bndbox')
            box = (
                int(xmlbox.find('xmin').text),
                int(xmlbox.find('ymin').text),
                int(xmlbox.find('xmax').text),
                int(xmlbox.find('ymax').text)
            )
            bb = convert_bbox((w, h), box)
            out_file.write(f"{cls_id} " + " ".join(map(str, bb)) + "\n")

for mode, files in [('train', train_files), ('val', val_files)]:
    for img_file in files:
        shutil.copy(os.path.join(IMG_DIR, img_file),
                    OUT_IMG_TRAIN if mode == 'train' else OUT_IMG_VAL)
        convert_annotation(img_file, mode)

print("✅ VOC to YOLO conversion completed successfully!")        