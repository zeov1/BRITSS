import os
from typing import Dict, List

import numpy as np
import torch
import ujson as json
from torch.utils.data import DataLoader, Dataset


class MySet(Dataset):
    """Dataset читает JSON Lines файл ./json/json.

    Каждая строка — один пример с ключами:
      - 'forward': список записей по времени
        (каждая запись — dict с полями values, masks, deltas, forwards, evals, eval_masks)
      - 'backward': то же, но в обратном порядке
      - 'label': скалярный float/int
    Поле 'is_train' добавляется здесь: 1 для train, 0 для val (20% случайно).
    """

    def __init__(self, path: str = "./json/json", val_ratio: float = 0.2, seed: int = 42):
        super().__init__()
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSONL file not found at: {os.path.abspath(path)}")

        with open(path) as f:
            self.content: list[str] = f.readlines()

        n = len(self.content)
        indices = np.arange(n)
        np.random.seed(seed)
        val_sz = max(1, round(n * val_ratio))
        val_indices = np.random.choice(indices, size=val_sz, replace=False)
        self.val_indices = {int(i) for i in val_indices}

    def __len__(self) -> int:
        return len(self.content)

    def __getitem__(self, idx: int) -> Dict:
        rec = json.loads(self.content[idx])
        rec["is_train"] = 0 if idx in self.val_indices else 1
        return rec


def _pack_time_series(recs) -> Dict:
    # Вложенные списки: [batch][time][feature]
    values = [[x["values"] for x in r] for r in recs]
    masks = [[x["masks"] for x in r] for r in recs]
    deltas = [[x["deltas"] for x in r] for r in recs]
    forwards = [[x["forwards"] for x in r] for r in recs]
    evals = [[x["evals"] for x in r] for r in recs]
    eval_ms = [[x["eval_masks"] for x in r] for r in recs]

    return {
        "values": torch.tensor(values, dtype=torch.float32),
        "masks": torch.tensor(masks, dtype=torch.float32),
        "deltas": torch.tensor(deltas, dtype=torch.float32),
        "forwards": torch.tensor(forwards, dtype=torch.float32),
        "evals": torch.tensor(evals, dtype=torch.float32),
        "eval_masks": torch.tensor(eval_ms, dtype=torch.float32),
    }


def collate_fn(recs: List) -> Dict:
    forward = [x["forward"] for x in recs]
    backward = [x["backward"] for x in recs]

    return {
        "forward": _pack_time_series(forward),
        "backward": _pack_time_series(backward),
        "labels": torch.tensor([x["label"] for x in recs], dtype=torch.float32),
        "is_train": torch.tensor([x["is_train"] for x in recs], dtype=torch.float32),
    }


def get_loader(batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    data_set = MySet()

    # Безопасное число воркеров (Colab часто ругается на 4+)
    max_suggested = 4
    cpu_cnt = os.cpu_count() or 2
    num_workers = min(max_suggested, cpu_cnt)

    return DataLoader(
        dataset=data_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
    )
