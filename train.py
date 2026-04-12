import torch, torch.nn as nn, copy, time, os
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm

# LABEL SMOOTHING LOSS
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=2):
        super().__init__()
        self.smoothing    = smoothing
        self.num_classes  = num_classes

    def forward(self, logits, targets):
        log_probs  = torch.log_softmax(logits, dim=-1)
        nll_loss   = -log_probs.gather(dim=-1,
                         index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing)*nll_loss + self.smoothing*smooth_loss
        return loss.mean()

# DUAL-CRITERION EARLY STOPPING
class EarlyStopping:
    def __init__(self, patience=5, gap_patience=3,
                 min_delta=0.001, max_gap=0.12):
        self.patience     = patience
        self.gap_patience = gap_patience
        self.min_delta    = min_delta
        self.max_gap      = max_gap

        self.acc_counter  = 0
        self.gap_counter  = 0
        self.best_acc     = 0.0
        self.best_weights = None
        self.should_stop  = False
        self.stop_reason  = ""

    def step(self, val_acc, train_acc, model):
        gap = train_acc - val_acc

        # Criterion (a): val accuracy stagnation
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc     = val_acc
            self.best_weights = copy.deepcopy(model.state_dict())
            self.acc_counter  = 0
        else:
            self.acc_counter += 1

        # Criterion (b): gap too large
        if gap > self.max_gap:
            self.gap_counter += 1
        else:
            self.gap_counter = 0

        # Fire whichever criterion triggers first
        if self.acc_counter >= self.patience:
            self.should_stop = True
            self.stop_reason = (f"val accuracy flat for {self.patience} epochs "
                                f"(best={self.best_acc:.4f})")
        elif self.gap_counter >= self.gap_patience:
            self.should_stop = True
            self.stop_reason = (f"train-val gap > {self.max_gap:.0%} for "
                                f"{self.gap_patience} epochs "
                                f"(gap={gap:.3f})")

    def restore_best(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

# STOCHASTIC WEIGHT AVERAGING (lightweight EMA version)
class SWAHelper:
    def __init__(self, model, start_epoch=0, decay=0.99):
        self.start_epoch = start_epoch
        self.decay       = decay
        self.avg_weights = None
        self._enabled    = (start_epoch > 0)

    def update(self, model, epoch):
        if not self._enabled or epoch < self.start_epoch:
            return
        state = {k: v.float().clone()
                 for k, v in model.state_dict().items()}
        if self.avg_weights is None:
            self.avg_weights = state
        else:
            for k in self.avg_weights:
                self.avg_weights[k] = (self.decay * self.avg_weights[k]
                                       + (1 - self.decay) * state[k])

    def apply(self, model):
        if self.avg_weights is not None:
            model.load_state_dict(
                {k: v.to(next(model.parameters()).device)
                 for k, v in self.avg_weights.items()}
            )
            print("  [SWA] Applied averaged weights.")


# WARMUP SCHEDULER  (Transformer only)
def get_warmup_scheduler(optimizer, warmup_steps, d_model):
    def lr_lambda(step):
        step = max(step, 1)
        return (d_model ** -0.5) * min(step ** -0.5,
                                       step * (warmup_steps ** -1.5))
    return LambdaLR(optimizer, lr_lambda)


# TRAIN ONE EPOCH
def train_epoch(model, loader, optimizer, scheduler,
                criterion, device, model_type):
    model.train()
    total_loss = total_correct = total_n = 0

    for ids, lens, labels in tqdm(loader, desc="  Train", leave=False):
        ids    = ids.to(device)
        lens   = lens.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(ids) if model_type == "transformer" \
                 else model(ids, lens)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Step batch-level schedulers (warmup, cosine)
        if scheduler is not None and not isinstance(
                scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau,)):
            scheduler.step()

        total_loss    += loss.item() * labels.size(0)
        total_correct += (logits.argmax(-1) == labels).sum().item()
        total_n       += labels.size(0)

    return total_loss / total_n, total_correct / total_n


# EVALUATE
@torch.no_grad()
def evaluate(model, loader, device, model_type="transformer"):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss = total_correct = total_n = 0
    all_preds = []; all_labels = []

    for ids, lens, labels in tqdm(loader, desc="  Val  ", leave=False):
        ids    = ids.to(device)
        lens   = lens.to(device)
        labels = labels.to(device)

        logits = model(ids) if model_type == "transformer" \
                 else model(ids, lens)
        loss   = crit(logits, labels)
        preds  = logits.argmax(-1)

        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_n       += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return (total_loss / total_n,
            total_correct / total_n,
            all_preds, all_labels)


# FULL TRAINING LOOP
def train_model(model, train_loader, val_loader, config, save_path, model_type="transformer"):
    device    = config["device"]
    model     = model.to(device)
    smoothing = config.get("label_smoothing", 0.1)
    criterion = LabelSmoothingLoss(smoothing=smoothing)

    # Early stopping — now dual-criterion
    patience     = config.get("patience",     5)
    gap_patience = config.get("gap_patience", 3)
    max_gap      = config.get("max_gap",      0.12)
    early_stop   = EarlyStopping(
        patience=patience,
        gap_patience=gap_patience,
        min_delta=0.001,
        max_gap=max_gap
    )

    # SWA (start after swa_start epochs, 0 = disabled)
    swa_start = config.get("swa_start", 0)
    swa       = SWAHelper(model, start_epoch=swa_start,
                          decay=config.get("swa_decay", 0.99))

    # Optimizer + scheduler
    if model_type == "transformer":
        optimizer = Adam(model.parameters(), lr=1.0,
                         betas=(0.9, 0.98), eps=1e-9,
                         weight_decay=config.get("weight_decay", 0.0))
        scheduler = get_warmup_scheduler(
            optimizer,
            warmup_steps=config.get("warmup_steps", 400),
            d_model=config["d_model"])
        sched_type = "Vaswani warmup"
    else:
        optimizer = Adam(model.parameters(),
                         lr=config["lr"],
                         weight_decay=config.get("weight_decay", 5e-4))
        T_max = config.get("epochs", 20)
        lr_min= config.get("lr_min", 1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=lr_min)
        sched_type = f"CosineAnnealing(T={T_max}, min={lr_min})"

    history = {"train_loss":[], "train_acc":[],
               "val_loss":[], "val_acc":[], "lr":[]}
    best_val_acc  = 0.0
    stopped_epoch = config["epochs"]

    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} | max {config['epochs']} epochs")
    print(f"Device: {device} | "
          f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loss  : LabelSmoothing(ε={smoothing})")
    print(f"LR    : {sched_type}")
    print(f"Stop  : val_patience={patience} | "
          f"gap_patience={gap_patience} | max_gap={max_gap:.0%}")
    if swa_start > 0:
        print(f"SWA   : starts epoch {swa_start}")
    print(f"{'='*60}")

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, model_type)
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, device, model_type)

        # Cosine scheduler steps every EPOCH
        if model_type == "bilstm":
            scheduler.step()

        # SWA update
        swa.update(model, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        gap        = tr_acc - val_acc

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_dir = os.path.dirname(save_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc,
                        "config": config}, save_path)

        # Early stopping check (BOTH criteria)
        early_stop.step(val_acc, tr_acc, model)

        # History
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        flag = ""
        if gap > max_gap: flag = " ⚠ overfit"
        elif val_acc < 0.57 and epoch > 3: flag = " ⚠ underfit"

        print(f"Epoch {epoch:02d}/{config['epochs']} | "
              f"Train {tr_acc:.4f} Loss {tr_loss:.4f} | "
              f"Val {val_acc:.4f} Loss {val_loss:.4f} | "
              f"Gap {gap:+.3f} | Best {best_val_acc:.4f} | "
              f"LR {current_lr:.2e} | {elapsed:.1f}s{flag}")

        if early_stop.should_stop:
            stopped_epoch = epoch
            print(f"\n  ⛔ Early stop: {early_stop.stop_reason}")
            early_stop.restore_best(model)
            break

    # Apply SWA weights if trained long enough
    swa.apply(model)

    if not early_stop.should_stop:
        print(f"\n  Completed all {config['epochs']} epochs.")

    # If SWA was used, evaluate its weights and potentially re-save
    if swa.avg_weights is not None:
        _, swa_acc, _, _ = evaluate(model, val_loader, device, model_type)
        print(f"  SWA val accuracy: {swa_acc:.4f}  "
              f"(vs best checkpoint: {best_val_acc:.4f})")
        if swa_acc > best_val_acc:
            best_val_acc = swa_acc
            torch.save({"epoch": stopped_epoch,
                        "model_state_dict": model.state_dict(),
                        "val_acc": swa_acc,
                        "config": config}, save_path)
            print(f"  SWA weights saved (better than checkpoint).")

    print(f"  Best Val Accuracy : {best_val_acc:.4f}")
    print(f"  Stopped at epoch  : {stopped_epoch}")
    print(f"  Checkpoint        → {save_path}")
    return history