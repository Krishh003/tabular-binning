"""Quick ICML binning experiments and evaluation of the authors' CH encoder."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def quantile_bins(train, rows, count):
    edges = np.quantile(train, np.arange(1, count) / count, axis=0)
    return np.stack([
        np.searchsorted(np.unique(boundaries), values, side="right")
        for boundaries, values in zip(edges.T, rows.T)
    ], axis=1)


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, outputs))


def california(epochs, bins):
    X, y = fetch_california_housing(return_X_y=True)
    train, held = train_test_split(np.arange(len(X)), test_size=.4, random_state=42)
    validation, test = train_test_split(held, test_size=.5, random_state=42)
    X = StandardScaler().fit(X[train]).transform(X)
    x = torch.tensor(X, dtype=torch.float32)
    n_features = x.shape[1]
    ordinal = quantile_bins(X[train], X[train], bins)
    bin_class = torch.as_tensor(ordinal, dtype=torch.long)
    bin_order = torch.tensor(StandardScaler().fit_transform(ordinal), dtype=torch.float32)
    results = {}
    for objective, masking in [
        ("ValueRecon", "none"), ("ValueRecon", "constant"), ("BinXent", "none"),
        ("BinRecon", "none"), ("BinRecon", "constant"), ("BinRecon", "random"),
    ]:
        torch.manual_seed(42)
        encoder = mlp(n_features, 128)
        cross_entropy = objective == "BinXent"
        decoder = nn.Linear(128, n_features * bins if cross_entropy else n_features)
        optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=.002)
        x_train = x[train]
        for _ in range(epochs):
            for batch in torch.randperm(len(train)).split(256):
                original = x_train[batch]
                inputs = original
                if masking != "none":
                    mask = torch.rand_like(original) < .2
                    replacement = torch.zeros_like(original)
                    if masking == "random":
                        rows = torch.randint(len(train), original.shape)
                        replacement = x_train[rows, torch.arange(n_features)]
                    inputs = torch.where(mask, replacement, original)
                prediction = decoder(encoder(inputs))
                if cross_entropy:
                    loss = F.cross_entropy(prediction.reshape(-1, bins), bin_class[batch].flatten())
                else:
                    target = original if objective == "ValueRecon" else bin_order[batch]
                    loss = F.mse_loss(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            embedding = encoder(x).numpy().astype(np.float64)
        heads = [Ridge(alpha=alpha).fit(embedding[train], y[train]) for alpha in (.01, .1, 1, 10)]
        model = min(heads, key=lambda h: mean_squared_error(y[validation], h.predict(embedding[validation])))
        results[f"{objective}/{masking}"] = {
            "test_rmse": float(mean_squared_error(y[test], model.predict(embedding[test])) ** .5),
            "ridge_alpha": model.alpha,
        }
    return {"dataset": "California Housing", "epochs": epochs, "bins": bins, "results": results}


def pretrained_churn():
    assets = ROOT / "assets" / "tabularbinning"
    data = pd.read_csv(assets / "Churn_Modelling.csv").drop(columns=["RowNumber", "CustomerId", "Surname"])
    y = data.pop("Exited").to_numpy()
    for name in ("Geography", "Gender"):
        data[name] = pd.Categorical(data[name]).codes
    x = data.to_numpy(dtype=np.float32)
    rng = np.random.RandomState(123456)
    train, test = train_test_split(np.arange(len(x)), train_size=.9, random_state=rng)
    train, _ = train_test_split(train, train_size=.9, random_state=rng)
    numeric = [i for i in range(x.shape[1]) if len(np.unique(x[train, i])) > 20]
    x[:, numeric] = (x[:, numeric] - x[train][:, numeric].mean(0)) / x[train][:, numeric].std(0, ddof=1)
    categorical = [i for i in range(x.shape[1]) if i not in numeric]
    x = x[:, numeric + categorical]
    state = torch.load(assets / "model_encoder_ep1000.pt", map_location="cpu", weights_only=True)
    encoder = mlp(x.shape[1], 128)
    names = ("blocks.0.linear.weight", "blocks.0.linear.bias", "head.weight", "head.bias")
    encoder.load_state_dict({
        local: state[f"encoder.{original}"]
        for local, original in zip(encoder.state_dict(), names)
    })
    encoder.eval()
    with torch.no_grad():
        features = encoder(torch.from_numpy(x)).numpy()
    head = LogisticRegression(max_iter=1000).fit(features[train], y[train])
    return {"dataset": "Bank Churn", "encoder_pretraining_epochs_local": 0,
            "test_roc_auc": float(roc_auc_score(y[test], head.predict_proba(features[test])[:, 1]))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("california", "churn"), default="california")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1 or args.bins < 2:
        parser.error("Require at least one epoch and two bins.")
    torch.set_num_threads(2)
    result = california(args.epochs, args.bins) if args.dataset == "california" else pretrained_churn()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    report = json.dumps(result, indent=2)
    (out / f"binning_{args.dataset}.json").write_text(report, encoding="utf-8")
    print(report)
