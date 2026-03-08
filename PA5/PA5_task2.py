import torch, torch.nn as nn, torchvision, json, os, numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = 'PA5/data/a5_data_test'

# Eval vgg16 features on test
def evaluate_test():
    feat_file = 'PA5/vgg_features.npz'

    # Extract test features using VGG16
    w = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    vgg = torchvision.models.vgg16(weights=w).to(DEVICE).eval()

    test_path = f'{DATA}/test'
    if not os.path.exists(test_path):
        print(f"ERROR: test set not found at {test_path}")
        print("Download and extract the test set first.")
        return

    print("Extracting VGG features for test set...")
    feats, labels = [], []
    for X, y in DataLoader(ImageFolder(test_path, w.transforms()), batch_size=32):
        with torch.no_grad():
            f = vgg.avgpool(vgg.features(X.to(DEVICE))).flatten(1).cpu()
        feats.append(f); labels.append(y)
    X_test, y_test = torch.cat(feats), torch.cat(labels)
    print(f"Test set: {len(y_test)} images")

    # Load training features to get input size
    d = np.load(feat_file)
    X_tr, y_tr = torch.tensor(d['X_tr']), torch.tensor(d['y_tr'])

    # Rebuild the same classifier
    clf = nn.Sequential(
        nn.Linear(X_tr.shape[1], 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 2),
    ).to(DEVICE)

    # Retrain on full training data (same as task 1)
    print("Training classifier on VGG features...")
    opt = torch.optim.Adam(clf.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train_ldr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

    for epoch in range(30):
        clf.train()
        for X, y in train_ldr:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss_fn(clf(X), y).backward(); opt.step()

    # Evaluate on test set
    clf.eval()
    with torch.no_grad():
        preds = clf(X_test.to(DEVICE)).argmax(1).cpu()
    acc = (preds == y_test).float().mean().item()
    print(f"\n{'='*40}")
    print(f"TEST SET ACCURACY: {acc:.4f}")
    print(f"{'='*40}")
    return acc

if __name__ == "__main__":
    evaluate_test()