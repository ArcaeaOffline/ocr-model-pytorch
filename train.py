import copy
import json
import os
import time
from datetime import datetime

import torch
from rich.console import Console
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from dataset import build_dataloaders
from src import engine
from src.models import CRNN
from src.utils.early_stopping import EarlyStopping
from src.utils.logging_config import predictions_table, setup_logging
from src.utils.model_info import dump_model_info
from src.utils.model_decoders import decode_padded_predictions, decode_predictions
from src.utils.plot import plot_acc, plot_losses

# Setup rich console
console = Console()


def run_training():
    # Setup logging
    if not os.path.exists("logs"):
        os.makedirs("logs")
    logger = setup_logging()
    writer = SummaryWriter(f"logs/tensorboard/{cfg.MISC.start_timestamp_str}")

    # 1. Dataset and dataloaders
    train_loader, test_loader, test_original_targets, classes = build_dataloaders()

    logger.info("Dataset number of classes: %d", len(classes))
    logger.info("Classes are: %r", classes)

    time.sleep(1)

    # 2. Setup model, optim and scheduler
    device = cfg.TRAINING.device
    model = CRNN(
        resolution=(cfg.MODEL.image_width, cfg.MODEL.image_height),
        dims=cfg.MODEL.dims,
        num_chars=len(classes),
        use_attention=cfg.MODEL.use_attention,
        use_ctc=cfg.MODEL.use_ctc,
        grayscale=cfg.MODEL.grayscale,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAINING.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer  # , factor=0.8, patience=5
    )
    early_stopping = EarlyStopping(
        patience=cfg.TRAINING.early_stop_patience, verbose=True, trace_func=logger.info
    )

    best_acc = 0.0
    train_loss_data = []
    valid_loss_data = []
    accuracy_data = []

    # This is the same list of characters from dataset, but with the '∅' token
    # which denotes blank for ctc, or pad for cross_entropy
    training_classes = [cfg.MODEL.blank_token]
    training_classes.extend(classes)
    writer.add_text(
        "Summary",
        f"classes: {classes}\n"
        + f"{len(train_loader)} items in `train_loader`\n"
        + f"{len(test_loader)} items in `test_loader`\n"
        + f"training_classes: {training_classes}",
    )
    writer.flush()

    # 3. Training
    train_start = datetime.now()
    for epoch in range(cfg.TRAINING.epochs):
        # Train
        train_loss = engine.train_fn(model, train_loader, optimizer, device)
        train_loss_data.append(train_loss)
        # Eval
        valid_preds, test_loss = engine.eval_fn(model, test_loader, device)
        valid_loss_data.append(test_loss)
        # Eval + decoding for logging purposes
        valid_captcha_preds = []

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

        for vp in valid_preds:
            if model.use_ctc:
                # print(vp.shape)
                current_preds = decode_predictions(vp, training_classes)
            else:
                # print(vp)
                current_preds = decode_padded_predictions(vp, training_classes)

            if cfg.TRAINING.trim_paddings_at_end:
                current_preds = [x.replace("-", "") for x in current_preds]
            valid_captcha_preds.extend(current_preds)

        # Logging
        combined = list(zip(test_original_targets, valid_captcha_preds))[:10]
        if cfg.TRAINING.view_inference_while_training:
            table = predictions_table()
            for idx in combined:
                if cfg.TRAINING.display_only_wrong_inferences:
                    if idx[0] != idx[1]:
                        table.add_row(idx[0], idx[1])
                else:
                    table.add_row(idx[0], idx[1])
            console.print(table)

        accuracy = metrics.accuracy_score(test_original_targets, valid_captcha_preds)
        accuracy_data.append(accuracy)

        writer.add_scalar("Accuracy", accuracy, epoch)

        if accuracy > best_acc:
            best_acc = accuracy
            logger.info(f"New best accuracy {best_acc} achieved at epoch {epoch}.")
            best_model_wts = copy.deepcopy(model.state_dict())
            if cfg.TRAINING.save_checkpoints:
                torch.save(model, f"logs/checkpoint-{(best_acc*100):.2f}.pth")

        scheduler.step(test_loss)
        writer.add_scalar("Learning rate", torch.Tensor(scheduler.get_last_lr()), epoch)

        logger.info(
            f"Epoch {epoch}, Train loss: {train_loss}, Test loss: {test_loss}, Accuracy: {accuracy}"
        )
        writer.flush()

        early_stopping(test_loss, model)
        if early_stopping.early_stop:
            logger.warning("Early stop criteria met! Terminating...")
            writer.add_text("Early stop", f"Early stop triggered at epoch {epoch}")
            writer.flush()
            break

    # 4. Save model + logging and plotting
    logger.info("Finished training. Best Accuracy was: %.2f%%", best_acc * 100)

    cfg.MODEL.save_base_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), cfg.MODEL.save_base_path)

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), cfg.MODEL.save_best_acc_path)

    early_stopping.restore_model(model)
    torch.save(model.state_dict(), cfg.MODEL.save_early_stop_path)

    logger.info("Training time: %s", datetime.now() - train_start)
    logger.info(
        "Early stop saved to %s\n"
        "Best accuracy (%.2f%%) saved to %s\n"
        "Raw model saved to %s",
        cfg.MODEL.save_early_stop_path,
        best_acc * 100,
        cfg.MODEL.save_best_acc_path,
        cfg.MODEL.save_base_path,
    )

    cfg.MODEL.save_info_path.write_text(
        json.dumps(dump_model_info(training_classes).asdict(), ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("ModelInfo saved to %s", cfg.MODEL.save_info_path)

    plot_losses(train_loss_data, valid_loss_data)
    plot_acc(accuracy_data)


if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        run_training()
    except Exception:
        console.print_exception()
