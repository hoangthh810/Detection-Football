import os
import shutil
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="merged dataset")
    parser.add_argument("--dataset1", "-d1", type=str, default="public_1", help="Root of dataset 1")
    parser.add_argument("--dataset2", "-d2", type=str, default="public_2", help="Root of dataset 2")
    parser.add_argument("--output", "-o", type=str, default="data/global",
                        help="Output of dataset is merged from dataset1 and dataset 2")

    return parser.parse_args()

def handle_label(label_path):
    with open(label_path, 'r') as f:
        annotations = f.readlines()
    new_lines = []
    for annotation in annotations:
        annotation = annotation.rstrip().split()
        if annotation[0] == "0" or annotation[0] == "1":
            new_lines.append(" ".join(annotation) + "\n")
        elif annotation[0] == "2":
            annotation[0] = "1"
            new_lines.append(" ".join(annotation) + "\n")
    with open(label_path, "w") as f:
        f.writelines(new_lines)

def copy_dataset(src, val, output, prefix):
    if val:
        container = "val"
    else:
        container = "train"
    imgs_path = os.path.join(src, f"{container}/images")
    labels_path = os.path.join(src, f"{container}/labels")
    for filename in os.listdir(imgs_path):
        new_name = f"{prefix}_{filename}"
        shutil.copy(os.path.join(imgs_path, filename), os.path.join(output, f"{container}/images", new_name))

        name, ext = os.path.splitext(filename)
        label_path = os.path.join(labels_path, filename.replace(ext, '.txt'))
        if os.path.exists(label_path):
            handle_label(label_path)
            new_name_label = f"{prefix}_{filename.replace(ext, '.txt')}"
            shutil.copy(label_path, os.path.join(output, f"{container}/labels", new_name_label))


def merge_dataset(dataset1, dataset2, output):
    os.makedirs(os.path.join(output, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(output, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(output, "val/images"), exist_ok=True)
    os.makedirs(os.path.join(output, "val/labels"), exist_ok=True)

    copy_dataset(dataset1, False, output, "set1")
    copy_dataset(dataset2, False, output, "set2")
    copy_dataset(dataset1, True, output, "set1")
    copy_dataset(dataset2, True, output, "set2")


if __name__ == '__main__':
    args = get_args()
    merge_dataset(args.dataset1, args.dataset2, args.output)
