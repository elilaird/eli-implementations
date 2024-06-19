import copy
import os
import sys
from datetime import datetime
import time
import importlib
import argparse
import tqdm
import torch

from utils.training_utils import AverageMeter

# print available devices
print("Available devices:")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

print("Available cpus: ", os.cpu_count())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.py")
    args = parser.parse_args()
    return args


def get_config(config_name):
    spec = importlib.util.spec_from_file_location("config", config_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def get_model(config, device):
    pass


def get_optimizer(config, model, num_steps):

    pass


# loads the dataset and returns a dataloader
def get_dataset(config, device):
    pass


def run(config, name):

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get dataset
    train_loader, val_loader = get_dataset(config, device)

    # get model
    model = get_model(config, device)

    # get optimizer
    optimizer, lr_scheduler = get_optimizer(config, model, len(train_loader))

    # closure to update lr
    def update_lr(step):
        new_lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    # train loop
    pbar = tqdm.tqdm(
        total=config.epochs,
        desc=f"Epoch 0 - Train_Loss: 0.0 - Eval_Loss 0.0 - LR: 0 - Time: 0.0",
    )

    losses = []
    best_loss = float("inf")
    best_model = None
    early_stopping_counter = 0
    for epoch in pbar:
        model.train()

        train_loss_meter = AverageMeter()
        eval_loss_meter = AverageMeter()
        epoch_start = time.time()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # update lr
            curr_lr = update_lr(epoch * len(train_loader) + i)

            loss = model(x, y)
            loss.backward()

            # apply gradients
            optimizer.step()

            # update meters
            train_loss_meter.update(loss.item())

        # validation loop
        model.eval()
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                loss = model(x, y)

            eval_loss_meter.update(loss.item())

        # logging
        epoch_time = time.time() - epoch_start
        losses.append((train_loss_meter.avg, eval_loss_meter.avg))

        if eval_loss_meter.avg < best_loss:
            best_loss = eval_loss_meter.avg
            early_stopping_counter = 0
            best_model = copy.deepcopy(model.state_dict())
            # save_checkpoint
            pass  # not implemented yet
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}")
            break

        losses
        pbar.set_description(
            f"Epoch {epoch} - Train_Loss: {train_loss_meter.avg:.4f} - Eval_Loss {eval_loss_meter.avg:.4f} - LR: {curr_lr} - Time: {epoch_time:.2f}"
        )

    results = {
        "losses": losses,
        "best_loss": best_loss,
        "epochs": list(range(config.epochs)),
    }

    return results


if __name__ == "__main__":
    args = parse_args()
    config = get_config(args.config)

    name = f"{config.backbone}_{config.loss_head}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"

    results = run(config)

    # save results
    # not implemented yet
