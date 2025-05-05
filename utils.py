import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from datetime import datetime

def dice_score(pred, target, class_id):
    pred_bin = (pred == class_id).astype(np.uint8)
    target_bin = (target == class_id).astype(np.uint8)

    intersection = np.sum(pred_bin * target_bin)
    union = np.sum(pred_bin) + np.sum(target_bin)

    if np.sum(target_bin) == 0 and np.sum(pred_bin) == 0:
        return 1.0
    elif np.sum(target_bin) == 0 or np.sum(pred_bin) == 0:
        return 0.0

    return 2.0 * intersection / union


def visualize_random_predictions(model, dataset, device, num_samples=3, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().cpu().numpy()

        image_np = image.squeeze().cpu().numpy()
        mask_np = mask.numpy()

        ax_row = axes[i] if num_samples > 1 else axes

        ax_row[0].imshow(image_np, cmap='viridis')
        ax_row[0].set_title("IRM")
        ax_row[0].axis('off')

        ax_row[1].imshow(mask_np, cmap='viridis')
        ax_row[1].set_title("Real Mask")
        ax_row[1].axis('off')

        ax_row[2].imshow(pred, cmap='viridis')
        ax_row[2].set_title("Predicted Mask")
        ax_row[2].axis('off')

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/ensemble_predictions_{timestamp}.png"
    plt.savefig(filename)
    plt.show()

    print(f"Figure saved : {filename}")

def evaluate_model(model, dataset, device):
    model.eval()
    dice_totals = {1: 0.0, 2: 0.0, 3: 0.0}
    counts = {1: 0, 2: 0, 3: 0}

    for i in range(len(dataset)):
        image, mask = dataset[i]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze().cpu().numpy()

        mask = mask.numpy()

        for cls in [1, 2, 3]:
            if (mask == cls).sum() > 0:
                score = dice_score(pred, mask, class_id=cls)
                dice_totals[cls] += score
                counts[cls] += 1

    for cls in [1, 2, 3]:
        if counts[cls] > 0:
            avg = dice_totals[cls] / counts[cls]
            print(f"Average Dice class {cls} : {avg:.4f} (on {counts[cls]} slices)")
        else:
            print(f"Class {cls} missing in dataset.")

    total_samples = sum(counts[cls] for cls in [1, 2, 3] if counts[cls] > 0)

    if total_samples > 0:
        dice_pondere = sum(dice_totals[cls] for cls in [1, 2, 3]) / total_samples
        print(f"\nWeightened average Dice (samples) : {dice_pondere:.4f}")

def trainCurves(hist_loss, hist_dice):
    plt.figure(figsize=(8, 5))
    plt.plot(hist_loss, label="Loss (train)", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/loss_curves_{timestamp}.png"
    plt.savefig(save_path)
    plt.show(block=False)
    plt.pause(0.001)


    print(f"Loss Curve saved : {save_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(hist_dice, label="Average Dice (val)", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Average Dice Curve")
    plt.legend()
    plt.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/dice_curves_{timestamp}.png"
    plt.savefig(save_path)
    plt.show(block=False)
    plt.pause(0.001)


    print(f"Dice curve Saved : {save_path}")

