import json
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob
from torchvision.transforms import transforms, Compose, Resize, ToTensor
from pprint import pprint


class FootballDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.matches_path = glob.glob(f"{self.root}/*/*")
        self.videos_path = glob.glob(f"{self.root}/*/*/*.mp4")
        self.annotations_path = glob.glob(f"{self.root}/*/*/*.json")
        self.from_id = 0
        self.to_id = 0
        self.frame = {}
        self.jerseys = []
        self.labels = {}

        for annotation_path in self.annotations_path:
            with open(annotation_path, "r") as f:
                annotations = json.load(f)
            annotations = annotations["annotations"]
            player_annotations = [anno for anno in annotations if int(anno["category_id"]) == 4]
            for player_annotation in player_annotations:
                number = player_annotation["attributes"]["jersey_number"]
                if int(number) not in self.jerseys:
                   self.jerseys.append(int(number))
        for i, jersey in enumerate(sorted(self.jerseys)):
            self.labels[jersey] = i

        for video_path, annotation_path in zip(self.videos_path, self.annotations_path):
            with open(annotation_path, "r") as f:
                annotation = json.load(f)
            self.to_id = len(annotation["images"]) + self.to_id
            name, ext = os.path.splitext(video_path)
            self.frame[name] = [self.from_id + 1, self.to_id]
            self.from_id = self.to_id

    def __len__(self):
        return self.to_id

    def __getitem__(self, idx):
        for key, value in self.frame.items():
            if value[0] <= idx + 1 <= value[1]:
                idx = idx + 1 - value[0]
                select_path = key
        video_path = select_path + ".mp4"
        json_path  = select_path + ".json"
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        flag, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with open(json_path, "r") as f:
            annotations = json.load(f)
        annotations = [anno for anno in annotations["annotations"] if
                       int(anno["image_id"]) == idx and int(anno["category_id"]) == 4]
        bbox = [annotation["bbox"] for annotation in annotations]
        crop_images = [frame[int(y):int(y + h), int(x):int(x + w)] for x,y,w,h in bbox]
        crop_images = [self.transform(Image.fromarray(crop_image)) for crop_image in crop_images]
        jerseys = [self.labels[int(annotation["attributes"]["jersey_number"])] for annotation in annotations]
        return crop_images, jerseys


