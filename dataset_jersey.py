from torch.utils.data import Dataset
import os
import glob
from PIL import Image

class jerseyDataset(Dataset):
    def __init__(self, root, val=False, transform=None):
        self.root = root
        self.transform = transform
        self.jerseys = [int(jersey) for jersey in os.listdir(os.path.join(root, 'train'))]
        if val:
            self.paths = sorted(glob.glob(f"{self.root}/val/*"), key=lambda x: int(os.path.basename(x)))
        else:
            self.paths = sorted(glob.glob(f"{self.root}/train/*"), key=lambda x: int(os.path.basename(x)))
        self.images = []
        self.labels = []
        for i, path in enumerate(self.paths):
            for path_img in os.listdir(path):
                self.images.append(os.path.join(path, path_img))
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    dataset = jerseyDataset(root="data_cls")






