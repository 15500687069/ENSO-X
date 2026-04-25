import numpy as np


def corr_array_np(pred, true):
    pred_ = pred - np.mean(pred, axis=0, keepdims=True)
    true_ = true - np.mean(true, axis=0, keepdims=True)
    num = np.sum(pred_ * true_, axis=0)
    den = np.sqrt(np.sum(pred_ ** 2, axis=0) * np.sum(true_ ** 2, axis=0)) + 1e-6
    return num / den


def weighted_skill_np(pred, true, pred_time):
    corr = corr_array_np(pred, true)
    base = [1.5] * 6 + [2.0] * 6 + [3.0] * 6 + [4.0] * 6
    if pred_time > len(base):
        base.extend([5.0] * (pred_time - len(base)))
    weights = np.asarray(base[:pred_time], dtype=np.float32) * np.log(np.arange(pred_time) + 2)
    rmse = np.sum(np.sqrt(np.mean((pred - true) ** 2, axis=0)))
    acc = np.sum(weights * corr)
    return (2.0 / 3.0) * acc - rmse, corr
