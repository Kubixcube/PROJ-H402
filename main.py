import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torch.utils.data import random_split

from dataset import BraTSDataset
from model import UNet
from utils import dice_score, visualize_random_predictions, evaluate_model, trainCurves


# === Variable ===
DATA_DIR = "TrainingData"
DEVICE = "cuda"
NUM_EPOCHS = 20
MAX_PATIENTS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# CUDA
torch.backends.cudnn.benchmark = True

def train(train_set, val_set):

    hist_loss = []
    hist_dice = []

    meilleur_dice_val = 0.0
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet(in_channels=1, num_classes=4).to(DEVICE)

    if os.path.exists("modele.pth"):
        print("Loading from modele.pth")
        model.load_state_dict(torch.load("modele.pth", map_location=DEVICE))

    weights = torch.tensor([0.2, 2.0, 2.0, 3.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        hist_loss.append(avg_loss)
        duration = time.time() - start_time
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Loss : {avg_loss:.4f} - Time : {duration:.2f}s")
        
        model.eval()
        dice_totals = {1: 0.0, 2: 0.0, 3: 0.0}
        counts = {1: 0, 2: 0, 3: 0}

        with torch.no_grad():
            for i in range(len(val_set)):
                image, mask = val_set[i]
                image = image.unsqueeze(0).to(DEVICE)
                output = model(image)
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().cpu().numpy()
                mask = mask.numpy()
                for cls in [1, 2, 3]:
                    if (mask == cls).sum() > 0:
                        score = dice_score(pred, mask, class_id=cls)
                        dice_totals[cls] += score
                        counts[cls] += 1

        print("  - Dice val :", end=" ")

        dice_moyen = 0
        nb_classes = 0

        for cls in [1, 2, 3]:
            if counts[cls] > 0:
                avg = dice_totals[cls] / counts[cls]
                print(f"c{cls}:{avg:.3f}", end=" ")
                dice_moyen += dice_totals[cls] / counts[cls]
                nb_classes += 1
            else:
                print(f"c{cls}:N/A", end=" ")
        print()
        if nb_classes > 0:
            dice_moyen /= nb_classes
            if dice_moyen > meilleur_dice_val:
                meilleur_dice_val = dice_moyen
                torch.save(model.state_dict(), "meilleur_modele.pth")
                print(f"  -> New best Model saved (average Dice val : {dice_moyen:.4f})")

        hist_dice.append(dice_moyen)

    torch.save(model.state_dict(), "modele.pth")
    print("Model saved in modele.pth")


    trainCurves(hist_loss, hist_dice)

    return model

if __name__ == "__main__":

    dataset = BraTSDataset(
        data_dir=DATA_DIR,
        slice_step=5,
        max_patients=MAX_PATIENTS,
        random_selection=True,
        random_seed=42
    )

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    model = train(train_set, val_set)

    if os.path.exists("meilleur_modele.pth"):
        print("Loading best Model for Evaluation...")
        model.load_state_dict(torch.load("meilleur_modele.pth", map_location=DEVICE))
        model.to(DEVICE)
    else:
        print("No Saved Model Found.")

    evaluate_model(model, test_set, device=DEVICE)
    visualize_random_predictions(model, test_set, device=DEVICE, num_samples=5)