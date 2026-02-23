import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import glob

# your modules
from source.dataset import GraphDataModule
from train import GravNetLightning
from source.model import GravNetModel
from tqdm import tqdm

@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    model.freeze()

    true_classes = []
    pred_classes = []
    true_energy = []
    pred_energy = []

    for batch in tqdm(dataloader, desc="Running Inference..."):
        batch = batch.to(device)

        class_logits, energy_pred = model(batch)
        preds = torch.argmax(class_logits, dim=1)

        true_classes.extend(batch.y_class.cpu().numpy().tolist())
        pred_classes.extend(preds.cpu().numpy().tolist())

        true_energy.extend(batch.y_energy.cpu().numpy().tolist())
        pred_energy.extend(energy_pred.cpu().numpy().tolist())

    # convert AFTER loop
    true_classes = np.array(true_classes)
    pred_classes = np.array(pred_classes)

    true_energy = 10 ** np.array(true_energy)
    pred_energy = 10 ** np.array(pred_energy)

    return (true_classes, pred_classes), (true_energy, pred_energy)



########################################
# Config
########################################

CHECKPOINT = "/home/benwilson/work/nuGraph/lightning_logs/version_6/checkpoints/epoch=6-step=194117.ckpt"
BATCH_SIZE = 1
NUM_WORKERS = 8
DEVICE = "cpu"

########################################
# Load model
########################################

backbone = GravNetModel(
        ).to(DEVICE)
model = GravNetLightning.load_from_checkpoint(
        CHECKPOINT,
        model=backbone,
        map_location="cpu"
    )


########################################
# Build test dataset
########################################

# device = torch.device("cuda" if cfg.device == "gpu" else "cpu")
pt_files = glob.glob("data/pt/*/*.pt")    

datamodule = GraphDataModule(
    pt_files=pt_files,
    batch_size=1,
    num_workers=8,
)
datamodule.setup()

########################################
# Inference
########################################

(true_classes, pred_classes), (true_energy, pred_energy) = run_inference(
    model,
    datamodule.test_dataloader(),
    device=DEVICE
)


########################################
# Confusion matrix
########################################

cm = confusion_matrix(true_classes, pred_classes)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Neutrino Classification Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

########################################
# Energy regression plot
########################################

plt.figure(figsize=(7,7))
plt.scatter(true_energy, pred_energy, s=5, alpha=0.5)

min_e = min(true_energy.min(), pred_energy.min())
max_e = max(true_energy.max(), pred_energy.max())

plt.plot([min_e, max_e], [min_e, max_e])

plt.xlabel("True Energy")
plt.ylabel("Reconstructed Energy")
plt.title("Energy Regression")

plt.savefig("energy_regression.png", dpi=300)
plt.close()

########################################
# Optional: resolution plot
########################################

resolution = (pred_energy - true_energy) / true_energy

plt.figure()
plt.hist(resolution, bins=100)
plt.xlabel("(Reco - True) / True")
plt.ylabel("Events")
plt.title("Energy Resolution")
plt.savefig("energy_resolution.png", dpi=300)
plt.close()

print("Inference complete.")
print("Saved:")
print(" - confusion_matrix.png")
print(" - energy_regression.png")
print(" - energy_resolution.png")