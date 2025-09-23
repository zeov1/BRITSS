import copy
import time
import argparse
import os
import sys
import importlib.util

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
import matplotlib.pyplot as plt

import utils
import data_loader


# Загружаем модели вручную (rits, brits и т.д.)
module_names = ['rits_i', 'brits_i', 'rits', 'brits']
models = type('models', (object,), {})()

for module_name in module_names:
    module_path = f'/content/BRITSS/models/{module_name}.py'
    if not os.path.exists(module_path):
        print(f"Error: Module file not found at {module_path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    setattr(models, module_name, module)
    spec.loader.exec_module(module)


# ПАРСЕР аргументов
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()


# ==================== ОБУЧЕНИЕ ====================

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    data_iter = data_loader.get_loader(batch_size=args.batch_size)

    val_mae_hist = []

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer)
            run_loss += ret['loss'].item()
            pct = (idx + 1) * 100.0 / len(data_iter)
            avg_loss = run_loss / (idx + 1.0)
            print(f"\rProgress epoch {epoch}, {pct:.2f}%, average loss {avg_loss:.6f}",
                  end='', flush=True)
        print()

        mae, mre, auc = evaluate(model, data_iter)
        val_mae_hist.append(mae)

    # сохраняем веса
    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved to model.pth")

    # график MAE (Fig.4)
    plt.figure()
    plt.plot(range(1, len(val_mae_hist)+1), val_mae_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Imputation Error (MAE)")
    plt.title("Validation imputation error (BRITS)")
    plt.tight_layout()
    plt.savefig("val_imputation_error.png")
    plt.show()

    # после обучения — сохраняем графики
    visualize_examples(model, data_iter, feature_idx=0, min_obs=5, n_examples=3)

    # плюс можно сохранить модель
    #torch.save(model.state_dict(), "model.pth")
    #print("✅ Model saved to model.pth")


def evaluate(model, val_iter):
    model.eval()
    labels, preds = [], []
    evals, imputations = [], []

    with torch.no_grad():
        for data in val_iter:
            data = utils.to_var(data)
            ret = model.run_on_batch(data, None)

            pred = ret['predictions'].data.cpu().numpy()
            label = ret['labels'].data.cpu().numpy()
            is_train = ret['is_train'].data.cpu().numpy()

            eval_masks = ret['eval_masks'].data.cpu().numpy()
            eval_ = ret['evals'].data.cpu().numpy()
            imputation = ret['imputations'].data.cpu().numpy()

            evals += eval_[np.where(eval_masks == 1)].tolist()
            imputations += imputation[np.where(eval_masks == 1)].tolist()

            preds += pred[np.where(is_train == 0)].tolist()
            labels += label[np.where(is_train == 0)].tolist()

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    mae = np.abs(evals - imputations).mean()
    mre = np.abs(evals - imputations).sum() / np.abs(evals).sum()

    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)
    auc = metrics.roc_auc_score(labels, preds)

    print(f"AUC {auc:.6f}")
    print(f"MAE {mae:.6f}")
    print(f"MRE {mre:.6f}")
    return mae, mre, auc


# ==================== ВИЗУАЛИЗАЦИЯ (Fig.3) ====================


def visualize_examples(model, loader, feature_idx=0, min_obs=5, n_examples=3, out_prefix="imputation_example"):
    """
    Сохраняет несколько графиков (png) с примерами иммутации временных рядов.

    model       : обученная модель
    loader      : DataLoader
    feature_idx : индекс признака для визуализации
    min_obs     : минимальное количество наблюдаемых точек
    n_examples  : сколько примеров сохранить
    out_prefix  : имя файлов (будут out_prefix_0.png, out_prefix_1.png, ...)
    """
    model.eval()
    examples_plotted = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = utils.to_var(batch)
            ret = model.run_on_batch(batch, None)

            values = batch['forward']['values'].cpu().numpy()
            masks = batch['forward']['masks'].cpu().numpy()
            imputations = ret['imputations'].cpu().numpy()

            B, T, D = values.shape

            for b in range(B):
                obs = values[b, :, feature_idx]
                msk = masks[b, :, feature_idx]
                imp = imputations[b, :, feature_idx]

                if msk.sum() < min_obs:
                    continue

                plt.figure(figsize=(6, 3))
                plt.plot(obs, "b-", label="observations")
                plt.plot(np.where(msk == 0)[0], obs[msk == 0], "g--", label="missing values")
                plt.plot(imp, "orange", linestyle="--", label="BRITS imputations")
                plt.title("Time series imputation example")
                plt.xlabel("steps")
                plt.ylabel("value")
                plt.legend()
                plt.tight_layout()

                out_file = f"{out_prefix}_{examples_plotted}.png"
                plt.savefig(out_file)
                plt.close()
                print(f"✅ Saved {out_file}")

                examples_plotted += 1
                if examples_plotted >= n_examples:
                    return




# ==================== RUN ====================

def run():
    model = getattr(models, args.model).Model()
    if torch.cuda.is_available():
        model = model.cuda()
    train(model)


if __name__ == '__main__':
    run()




