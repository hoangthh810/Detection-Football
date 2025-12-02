import glob
from argparse import ArgumentParser
import cv2
import os
import json
from pprint import pprint


def get_args():
    parser = ArgumentParser(description="create dataset video")
    parser.add_argument("--root", "-r", type=str, default="local_train", help="Root of video")
    parser.add_argument("--dataset", "-d", type=str, default="data/local", help="Dataset name")
    parser.add_argument("--ratio", "-ra", type=int, default=2, help="Ratio of train and validation, 1 = 10% validation")

    return parser.parse_args()


def create_dataset_from_video(root, dataset):
    os.makedirs(os.path.join(dataset, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(dataset, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(dataset, "val/images"), exist_ok=True)
    os.makedirs(os.path.join(dataset, "val/labels"), exist_ok=True)

    videos = glob.glob(root + "/*/*/*mp4")
    labels = glob.glob(root + "/*/*/*json")
    video_number = 1
    numberOfVideo = len(videos)
    for video, label in zip(videos, labels):
        name_video = video.split("\\")[-1]
        namev, ext = os.path.splitext(name_video)
        if video_number > int(numberOfVideo - args.ratio * 0.1 * numberOfVideo):
            container = "val"
        else:
            container = "train"
        with open(label, "r") as f:
            jsons = json.load(f)
        names_img = jsons["images"]
        annotations = jsons["annotations"]
        w = jsons["images"][0]["width"]
        h = jsons["images"][0]["height"]

        index = 0
        cap = cv2.VideoCapture(video)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ((ret == True) and (index % 5 == 0)):
                boxes = [anno for anno in annotations if
                              ((int(anno["image_id"]) == index + 1) and anno["category_id"] >= 3)]
                name = [name for name in names_img if int(name["id"]) == index + 1][0]["file_name"]
                with open (os.path.join(dataset, f"{container}/labels/{namev}.{name}.txt"), "w") as f:
                    for box in boxes:
                        category = box["category_id"]
                        if category == 4:
                            category = 1
                        else :
                            category = 0
                        x_min, y_min, width, height = box["bbox"]

                        x_min = (float(x_min))
                        y_min = (float(y_min))
                        width = float(width)
                        height = float(height)
                        x_cen = x_min + width / 2
                        y_cen = y_min + height / 2
                        f.write(f'{category} {x_cen / w} {y_cen / h} {width / w} {height / h}\n')
                cv2.imwrite(os.path.join(dataset, f"{container}/images/{namev}.{name}.png", ), frame)
            elif (ret == False):
                break
            index += 1
        cap.release()
        video_number += 1
if __name__ == '__main__':
    args = get_args()
    create_dataset_from_video(args.root, args.dataset)
