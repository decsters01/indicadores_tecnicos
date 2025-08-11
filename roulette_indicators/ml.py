from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import pandas as pd

from .features import (
    spins_to_colors,
    spins_to_highlow,
    spins_to_dozens,
    spins_to_columns,
    rolling_category_frequencies,
)

_TASK_TO_ENCODER = {
    "color": spins_to_colors,
    "highlow": spins_to_highlow,
    "dozen": spins_to_dozens,
    "column": spins_to_columns,
}

_TASK_LABELS = {
    "color": ["R", "B"],
    "highlow": ["H", "L"],
    "dozen": ["D1", "D2", "D3"],
    "column": ["C1", "C2", "C3"],
}


def _one_hot_last_k(labels: pd.Series, classes: List[str], k: int) -> pd.DataFrame:
    frames = []
    for lag in range(1, k + 1):
        lab_lag = labels.shift(lag)
        for c in classes:
            col = (lab_lag == c).astype(float).rename(f"lag{lag}_{c}")
            frames.append(col)
    return pd.concat(frames, axis=1)


def _streak_features(labels: pd.Series, classes: List[str]) -> pd.DataFrame:
    # tamanho da sequência corrente por classe (até t-1)
    lab = labels.shift(1)
    frames = []
    for c in classes:
        mask = (lab == c).astype(int)
        # run-length encoding cumulativa
        streak = mask.groupby((mask != mask.shift()).cumsum()).cumsum()
        frames.append(streak.rename(f"streak_{c}"))
    return pd.concat(frames, axis=1)


def build_features(spins: pd.Series, task: str, n_lags: int = 20, windows: List[int] = [20, 50, 100]) -> Tuple[pd.DataFrame, pd.Series]:
    if task not in _TASK_TO_ENCODER:
        raise ValueError("task inválida")
    labels = _TASK_TO_ENCODER[task](spins)
    classes = _TASK_LABELS[task]

    # One-hot dos últimos k rótulos (exclui 'Z' implicitamente por ser 0)
    lag_df = _one_hot_last_k(labels, classes, n_lags)

    # Frequências móveis por janela (com shift para evitar lookahead)
    freq_frames = []
    for w in windows:
        f = rolling_category_frequencies(spins, task, w).shift(1)
        f = f.rename(columns={c: f"freq{w}_{c}" for c in f.columns})
        freq_frames.append(f)

    # Streak atual por classe
    streak_df = _streak_features(labels, classes)

    X = pd.concat([lag_df] + freq_frames + [streak_df], axis=1)
    y = labels  # y_t a ser previsto

    # Remover 'Z' do alvo e linhas com NaN nas features
    mask_valid = (y != 'Z')
    X = X[mask_valid]
    y = y[mask_valid]
    X = X.dropna()
    y = y.loc[X.index]
    return X, y


@dataclass
class StandardScaler:
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


@dataclass
class LogisticOVR:
    classes_: List[str]
    W: Optional[np.ndarray] = None  # shape (n_features, n_classes)
    b: Optional[np.ndarray] = None  # shape (n_classes,)
    scaler: Optional[StandardScaler] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, lr: float = 0.05, epochs: int = 400, reg: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        Xn = X.values.astype(float)
        n, d = Xn.shape
        classes = self.classes_
        C = len(classes)
        Y = np.zeros((n, C), dtype=float)
        lab_to_idx = {c: i for i, c in enumerate(classes)}
        for i, lab in enumerate(y.values):
            Y[i, lab_to_idx[lab]] = 1.0

        self.scaler = StandardScaler().fit(Xn)
        Xs = self.scaler.transform(Xn)

        W = rng.normal(0, 0.01, size=(d, C))
        b = np.zeros(C, dtype=float)

        mW = np.zeros_like(W); vW = np.zeros_like(W)
        mb = np.zeros_like(b); vb = np.zeros_like(b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        for t in range(1, epochs + 1):
            Z = Xs @ W + b
            P = sigmoid(Z)
            E = (P - Y)
            grad_W = (Xs.T @ E) / n + reg * W
            grad_b = E.mean(axis=0)

            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * (grad_W ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

            mb = beta1 * mb + (1 - beta1) * grad_b
            vb = beta2 * vb + (1 - beta2) * (grad_b ** 2)
            mb_hat = mb / (1 - beta1 ** t)
            vb_hat = vb / (1 - beta2 ** t)
            b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        self.W, self.b = W, b
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X.values.astype(float))
        Z = Xs @ self.W + self.b
        P = 1.0 / (1.0 + np.exp(-Z))
        # normalizar para somar 1 (não é softmax, mas OVR); mantém proporção
        P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
        return P

    def predict(self, X: pd.DataFrame) -> pd.Series:
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        labs = [self.classes_[i] for i in idx]
        return pd.Series(labs, index=X.index)

    def save(self, path: str, feature_names: List[str]):
        obj = {
            "classes": self.classes_,
            "W": self.W.tolist(),
            "b": self.b.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "features": feature_names,
        }
        with open(path, "w") as f:
            json.dump(obj, f)

    @staticmethod
    def load(path: str) -> Tuple['LogisticOVR', List[str]]:
        with open(path, "r") as f:
            obj = json.load(f)
        model = LogisticOVR(classes_=obj["classes"])
        model.W = np.array(obj["W"], dtype=float)
        model.b = np.array(obj["b"], dtype=float)
        sc = StandardScaler()
        sc.mean_ = np.array(obj["scaler_mean"], dtype=float)
        sc.scale_ = np.array(obj["scaler_scale"], dtype=float)
        model.scaler = sc
        return model, obj["features"]


def time_split_index(n: int, val_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    split = int(n * (1 - val_frac))
    idx = np.arange(n)
    return idx[:split], idx[split:]


def train_and_evaluate(spins: pd.Series, task: str, n_lags: int = 20, windows: List[int] = [20, 50, 100],
                       lr: float = 0.05, epochs: int = 400, reg: float = 1e-3) -> Dict:
    X, y = build_features(spins, task, n_lags=n_lags, windows=windows)
    classes = _TASK_LABELS[task]
    tr_idx, va_idx = time_split_index(len(X), val_frac=0.2)
    Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
    Xva, yva = X.iloc[va_idx], y.iloc[va_idx]

    model = LogisticOVR(classes_=classes)
    model.fit(Xtr, ytr, lr=lr, epochs=epochs, reg=reg)

    yhat = model.predict(Xva)
    acc = (yhat == yva).mean()

    return {
        "model": model,
        "feature_names": X.columns.tolist(),
        "val_accuracy": float(acc),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
    }