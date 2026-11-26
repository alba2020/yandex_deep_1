import os
import random
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer


from scripts.dataset import create_dataloaders


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalCalorieModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(cfg.TEXT_MODEL_NAME)
        self.image_encoder = timm.create_model(
            cfg.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0,
        )

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, cfg.HIDDEN_DIM),
            nn.LayerNorm(cfg.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_encoder.num_features, cfg.HIDDEN_DIM),
            nn.LayerNorm(cfg.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
        )
        self.mass_projection = nn.Sequential(
            nn.Linear(1, cfg.HIDDEN_DIM // 2),
            nn.LayerNorm(cfg.HIDDEN_DIM // 2),
            nn.GELU(),
        )

        fusion_dim = cfg.HIDDEN_DIM * 2 + (cfg.HIDDEN_DIM // 2)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, cfg.HIDDEN_DIM),
            nn.LayerNorm(cfg.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.HIDDEN_DIM, 1),
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        image_features = self.image_encoder(image)

        text_emb = self.text_projection(text_features)
        image_emb = self.image_projection(image_features)
        mass_emb = self.mass_projection(mass)

        fused_emb = torch.cat([text_emb, image_emb, mass_emb], dim=-1)
        output = self.classifier(fused_emb)
        return output


def validate(model, val_loader, device, mae_metric):
    model.eval()
    mae_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device),
            }
            labels = batch["label"].to(device)

            preds = model(**inputs).squeeze(-1)
            mae_metric.update(preds, labels)

    return mae_metric.compute().item()


def train(cfg):
    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Запуск обучения на устройстве: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)
    train_loader, val_loader, norm_stats = create_dataloaders(cfg, tokenizer)

    model = MultimodalCalorieModel(cfg).to(device)
    set_requires_grad(model.text_encoder, unfreeze_pattern=cfg.TEXT_MODEL_UNFREEZE, verbose=False)
    set_requires_grad(model.image_encoder, unfreeze_pattern=cfg.IMAGE_MODEL_UNFREEZE, verbose=False)

    optimizer = AdamW([
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR},
        {"params": model.text_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.image_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.mass_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.classifier.parameters(), "lr": cfg.CLASSIFIER_LR},
    ], weight_decay=getattr(cfg, "WEIGHT_DECAY", 1e-2))

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    criterion = nn.L1Loss()

    train_mae = torchmetrics.MeanAbsoluteError().to(device)
    val_mae = torchmetrics.MeanAbsoluteError().to(device)

    best_val_mae = float("inf")
    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True) if os.path.dirname(cfg.SAVE_PATH) else None

    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0.0
        train_mae.reset()

        for batch in train_loader:
            optimizer.zero_grad()

            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device),
            }
            labels = batch["label"].to(device)

            preds = model(**inputs).squeeze(-1)
            loss = criterion(preds, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            train_mae.update(preds, labels)

        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        epoch_train_mae = train_mae.compute().item()
        epoch_val_mae = validate(model, val_loader, device, val_mae)

        print(
            f"Epoch {epoch + 1}/{cfg.EPOCHS} | "
            f"Train Loss: {avg_train_loss:.2f} | "
            f"Train MAE: {epoch_train_mae:.2f} | "
            f"Val MAE: {epoch_val_mae:.2f}"
        )

        if epoch_val_mae < best_val_mae:
            best_val_mae = epoch_val_mae
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "best_val_mae": best_val_mae,
                "norm_stats": norm_stats,
            }
            torch.save(checkpoint, cfg.SAVE_PATH)
            print(f"--> Сохранена лучшая модель (Val MAE: {best_val_mae:.2f}) в {cfg.SAVE_PATH}")

    print(f"\nОбучение завершено. Лучший Val MAE: {best_val_mae:.2f} ккал")
    return model, best_val_mae
