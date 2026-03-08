import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset
import json, os, numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FILE = 'PA5/results_task1.json'
DATA = 'PA5/data/a5_data_new'

# Load previous results
all_results = json.load(open(RESULTS_FILE)) if os.path.exists(RESULTS_FILE) else {}
if all_results:
    print(f"Loaded previous results: {list(all_results.keys())}")

# ── Transforms & Loaders ──
tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
tf_aug = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20), transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
])
train_loader     = DataLoader(ImageFolder(f'{DATA}/train', tf),     batch_size=32, shuffle=True)
# slightly changed
train_loader_aug = DataLoader(ImageFolder(f'{DATA}/train', tf_aug), batch_size=32, shuffle=True)
val_loader       = DataLoader(ImageFolder(f'{DATA}/val', tf),       batch_size=32)

# ── ResBlock ──
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
    def forward(self, x):
        return nn.functional.relu(self.block(x) + x)

# ── All Models ──
models = {
    'baseline': nn.Sequential(
        nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128*16*16, 2), 
    ),
    'batch_norm': nn.Sequential(
        nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128*16*16, 2),
    ),
    'layer_norm': nn.Sequential(
        nn.Conv2d(3,32,3,padding=1), nn.LayerNorm([32,128,128]), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1), nn.LayerNorm([64,64,64]), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64,128,3,padding=1), nn.LayerNorm([128,32,32]), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128*16*16, 2),
    ),
    'group_norm': nn.Sequential(
        nn.Conv2d(3,32,3,padding=1), nn.GroupNorm(4, 32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64,128,3,padding=1), nn.GroupNorm(16, 128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128*16*16, 2),
    ),
    'residual': nn.Sequential(
        nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        ResBlock(32),
        nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        ResBlock(64),
        nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        ResBlock(128),
        nn.Flatten(), nn.Linear(128*16*16, 2),
    ),
    'augmented': nn.Sequential(
        nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        ResBlock(32),
        nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        ResBlock(64),
        nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        ResBlock(128),
        nn.Flatten(), nn.Linear(128*16*16, 2),
    ),
}

# ── Train & Eval ──
loss_fn = nn.CrossEntropyLoss()

def train_eval(model, name, train_ldr, val_ldr, epochs=10, lr=0.001):
    if name in all_results:
        print(f"SKIP '{name}' (best_acc={all_results[name]['best_acc']:.4f})")
        return
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, accs = 0.0, []

    for epoch in range(epochs):
        model.train()
        for X, y in train_ldr:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss_fn(model(X), y).backward(); opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in val_ldr:
                correct += (model(X.to(DEVICE)).argmax(1).cpu() == y).sum().item()
        acc = correct / len(val_ldr.dataset)
        best_acc = max(best_acc, acc)
        accs.append(acc)
        print(f'  {name} Epoch {epoch+1}/{epochs}  val_acc={acc:.4f}')

    all_results[name] = {'best_acc': best_acc, 'epochs': accs}
    json.dump(all_results, open(RESULTS_FILE, 'w'), indent=2)
    print(f'  Best: {best_acc:.4f} — saved.\n')

# ── VGG16 Transfer Learning ──
def run_transfer():
    if 'transfer_vgg16' in all_results:
        print(f"SKIP 'transfer_vgg16' (best_acc={all_results['transfer_vgg16']['best_acc']:.4f})")
        return
    feat_file = 'PA5/vgg_features.npz'
    if os.path.exists(feat_file):
        print("Loading cached VGG features...")
        d = np.load(feat_file)
        X_tr, y_tr, X_va, y_va = [torch.tensor(d[k]) for k in ['X_tr','y_tr','X_va','y_va']]
    else:
        print("Extracting VGG features (one time only)...")
        w = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        vgg = torchvision.models.vgg16(weights=w).to(DEVICE).eval()
        def extract(path):
            feats, labels = [], []
            for X, y in DataLoader(ImageFolder(path, w.transforms()), batch_size=32):
                with torch.no_grad():
                    f = vgg.avgpool(vgg.features(X.to(DEVICE))).flatten(1).cpu()
                feats.append(f); labels.append(y)
            return torch.cat(feats), torch.cat(labels)
        X_tr, y_tr = extract(f'{DATA}/train')
        X_va, y_va = extract(f'{DATA}/val')
        np.savez(feat_file, X_tr=X_tr.numpy(), y_tr=y_tr.numpy(), X_va=X_va.numpy(), y_va=y_va.numpy())
    clf = nn.Sequential(
        nn.Linear(X_tr.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 2),
    )
    train_eval(clf, 'transfer_vgg16',
               DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True),
               DataLoader(TensorDataset(X_va, y_va), batch_size=64), epochs=30)

if __name__ == "__main__":
    for name, model in models.items():
        ldr = train_loader_aug if name == 'augmented' else train_loader
        train_eval(model, name, ldr, val_loader, epochs=10)
    run_transfer()

    print("\n" + "="*50 + "\nFINAL RESULTS\n" + "="*50)
    for name, res in all_results.items():
        print(f"  {name:20s} best_acc={res['best_acc']:.4f}")