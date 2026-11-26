import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from functools import partial


def build_ingredient_map(ingredients_csv_path):
    ingr_df = pd.read_csv(ingredients_csv_path)
    return {f"ingr_{row['id']:010d}": str(row['ingr']).strip() for _, row in ingr_df.iterrows()}


def map_ingredients_str(ingr_str, ingr_map, shuffle_ingrs=False):
    if pd.isna(ingr_str) or not str(ingr_str).strip():
        return ""
    ids = str(ingr_str).split(';')
    names = []
    for item in ids:
        name = ingr_map.get(item.strip(), "")
        if name and name.lower() != "deprecated":
            names.append(name)
    if shuffle_ingrs and len(names) > 1:
        random.shuffle(names)
    return ", ".join(names)


def prepare_dataframe(dish_csv_path, ingredients_csv_path, remove_train_outliers=True):
    df = pd.read_csv(dish_csv_path)
    ingr_map = build_ingredient_map(ingredients_csv_path)
    df['ingredients_list'] = df['ingredients'].apply(
        lambda x: [
            ingr_map.get(i.strip(), "")
            for i in str(x).split(';')
            if pd.notna(x) and ingr_map.get(i.strip(), "").lower() not in ("", "deprecated")
        ]
    )
    df['ingredients_text'] = df['ingredients_list'].apply(lambda lst: ", ".join(lst))

    if remove_train_outliers:
        outlier_mask = (df['split'] == 'train') & (df['total_calories'] > 2500)
        df = df[~outlier_mask].reset_index(drop=True)

    return df


def get_transforms(cfg, ds_type="train"):
    model_cfg = timm.get_pretrained_cfg(cfg.IMAGE_MODEL_NAME)
    img_size = cfg.IMG_SIZE if hasattr(cfg, "IMG_SIZE") and cfg.IMG_SIZE else model_cfg.input_size[1]

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(max_size=int(img_size * 1.1), p=1.0),
                A.RandomCrop(height=img_size, width=img_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),
                    rotate=(-25, 25),
                    translate_percent=(-0.05, 0.05),
                    p=0.5,
                ),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=0.5),
                A.Normalize(mean=model_cfg.mean, std=model_cfg.std),
                ToTensorV2(p=1.0),
            ],
            seed=cfg.SEED,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(max_size=img_size, p=1.0),
                A.CenterCrop(height=img_size, width=img_size, p=1.0),
                A.Normalize(mean=model_cfg.mean, std=model_cfg.std),
                ToTensorV2(p=1.0),
            ],
            seed=cfg.SEED,
        )

    return transforms


class DishMultimodalDataset(Dataset):
    def __init__(self, df, images_dir, transforms=None, mass_mean=0.0, mass_std=1.0, is_train=True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transforms = transforms
        self.mass_mean = mass_mean
        self.mass_std = mass_std if mass_std > 0 else 1.0
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = str(row['dish_id'])

        img_path = os.path.join(self.images_dir, dish_id, "rgb.png")
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        ingrs = list(row['ingredients_list'])
        if self.is_train and len(ingrs) > 1:
            random.shuffle(ingrs)
        text = ", ".join(ingrs) if ingrs else "empty dish"

        norm_mass = (float(row['total_mass']) - self.mass_mean) / self.mass_std
        target = float(row['total_calories'])

        return {
            "dish_id": dish_id,
            "image": image,
            "text": text,
            "mass": torch.tensor([norm_mass], dtype=torch.float32),
            "label": torch.tensor(target, dtype=torch.float32),
        }


def collate_fn(batch, tokenizer, max_length=64):
    dish_ids = [item["dish_id"] for item in batch]
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    masses = torch.stack([item["mass"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    tokenized_input = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    return {
        "dish_id": dish_ids,
        "image": images,
        "mass": masses,
        "label": labels,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
    }


def create_dataloaders(cfg, tokenizer):
    df = prepare_dataframe(
        cfg.DISH_CSV_PATH,
        cfg.INGREDIENTS_CSV_PATH,
        remove_train_outliers=getattr(cfg, "REMOVE_OUTLIERS", True),
    )

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    mass_mean = float(train_df["total_mass"].mean())
    mass_std = float(train_df["total_mass"].std())

    train_transforms = get_transforms(cfg, ds_type="train")
    test_transforms = get_transforms(cfg, ds_type="test")

    train_dataset = DishMultimodalDataset(
        train_df,
        images_dir=cfg.IMAGES_DIR,
        transforms=train_transforms,
        mass_mean=mass_mean,
        mass_std=mass_std,
        is_train=True,
    )
    test_dataset = DishMultimodalDataset(
        test_df,
        images_dir=cfg.IMAGES_DIR,
        transforms=test_transforms,
        mass_mean=mass_mean,
        mass_std=mass_std,
        is_train=False,
    )

    collate_fn_with_tok = partial(collate_fn, tokenizer=tokenizer, max_length=cfg.MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=getattr(cfg, "NUM_WORKERS", 2),
        pin_memory=True,
        collate_fn=collate_fn_with_tok,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=getattr(cfg, "NUM_WORKERS", 2),
        pin_memory=True,
        collate_fn=collate_fn_with_tok,
    )


    return train_loader, test_loader, {"mass_mean": mass_mean, "mass_std": mass_std}
