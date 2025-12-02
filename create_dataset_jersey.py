import json
import os
import glob
import cv2

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="create dataset video")
    parser.add_argument("--root-train", "-rt", type=str, default="local_train", help="Root of video train")
    parser.add_argument("--root-val", "-rv", type=str, default="local_val", help="Root of video val")
    parser.add_argument("--dataset", "-d", type=str, default="data_cls", help="Dataset name")
    parser.add_argument("--save-frame", "-s", type=int, default=5, help="Save frame")

    return parser.parse_args()


def get_paths(root):
    video_paths = sorted(glob.glob(f"{root}/*/*/*.mp4"))
    annotation_paths = sorted(glob.glob(f"{root}/*/*/*.json"))

    return video_paths, annotation_paths


def create_format_jersey(annotation_paths):
    jersey_list = []
    jerseys = {}
    for annotation_path in annotation_paths:
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
        players_annotation = [anno for anno in annotations["annotations"] if int(anno["category_id"]) == 4]
        for player in players_annotation:
            if int(player["attributes"]["jersey_number"]) not in jersey_list:
                jersey_list.append(int(player["attributes"]["jersey_number"]))
    for i, jersey in enumerate(sorted(jersey_list)):
        jerseys[str(jersey)] = i

    return jerseys

def save_images(dataset, video_paths, annotation_paths, jerseys, save_frame, val=False):
    index = 0

    for i, (video_path, annotation_path) in enumerate(zip(video_paths, annotation_paths)):
        with open(annotation_path, "r") as f:
            annotations = json.load(f)
        player_annotations = [anno for anno in annotations["annotations"] if int(anno["category_id"]) == 4]

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if index % save_frame != 0:
                index += 1
                continue
            elif (ret == True) and (index % save_frame == 0):
                players = [player for player in player_annotations if int(player["image_id"]) == index + 1]
                for player in players:
                    x, y, w, h = player["bbox"]
                    crop_image = frame[int(y):int(y + h), int(x):int(x + w)]
                    jersey = jerseys[player["attributes"]["jersey_number"]]
                    if not val:
                        cv2.imwrite(f"{dataset}/train/{jersey}/frame_{i + 1}_{index}.jpg",
                                    crop_image)
                    else:
                        cv2.imwrite(f"{dataset}/val/{jersey}/frame_{i + 1}_{index}.jpg",
                                    crop_image)
                index += 1
        index = 0
        cap.release()

def create_dataset_jersey(root_train, root_val, dataset, save_frame):
    os.makedirs(os.path.join(dataset, "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset, "val"), exist_ok=True)

    video_train_paths, annotation_train_paths = get_paths(root_train)
    video_val_paths, annotation_val_paths = get_paths(root_val)

    jerseys = create_format_jersey(annotation_train_paths)

    for cls in jerseys.values():
        os.makedirs(os.path.join(dataset, "train", str(cls)), exist_ok=True)
    for cls in jerseys.values():
        os.makedirs(os.path.join(dataset, "val", str(cls)), exist_ok=True)

    save_images(dataset, video_train_paths, annotation_train_paths, jerseys, save_frame, val=False)
    save_images(dataset, video_val_paths, annotation_val_paths, jerseys, save_frame, val=True)


if __name__ == '__main__':
    args = get_args()
    create_dataset_jersey(args.root_train, args.root_val, args.dataset, args.save_frame)
