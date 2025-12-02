from argparse import ArgumentParser

import os
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import transforms, Compose, Resize, ToTensor, Normalize
from torch import nn, optim
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from dataset_jersey import jerseyDataset
import matplotlib.pyplot as plt
import numpy as np


def get_args():
    parser = ArgumentParser(description="create dataset emotion")
    parser.add_argument("--root-train", "-rt", type=str, default="data_cls", help="Root of dataset train")
    parser.add_argument("--root-val", "-rv", type=str, default="data_cls", help="Root of dataset val")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument("--image_size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--checkpoint", "-cp", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--num-classes", "-n", type=int, default=18, help="classes")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    return parser.parse_args()


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


if __name__ == '__main__':
    args = get_args()
    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    train_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    val_transform = Compose([
        Resize((args.image_size, args.image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = jerseyDataset(args.root_train, val=False, transform=train_transform)
    dataset_val = jerseyDataset(args.root_val, val=True, transform=val_transform)

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4)

    val_dataloader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                num_workers=4)

    writer = SummaryWriter(args.logging)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, args.num_classes)
    model = model.to(device)
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0
    epochs = args.epochs
    num_iterations = len(train_dataloader)

    for epoch in range(start_epoch, epochs):
        model.train()

        progress_bar = tqdm(train_dataloader, colour='green')
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss_value = criterion(outputs, labels)
            progress_bar.set_description(
                "Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch + 1, args.epochs, iter + 1, num_iterations,
                                                                   loss_value))
            writer.add_scalar("Train/Loss", loss_value, epoch * num_iterations + iter)
            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        progress_bar = tqdm(val_dataloader, colour='yellow')
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(progress_bar):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(images)
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions),
                              class_names=dataset_val.jerseys, epoch=epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
        print("Epoch {}: Accuracy: {}".format(epoch + 1, accuracy))
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": accuracy,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy
