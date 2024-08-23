import logging

from rich.table import Table

import config as cfg


def setup_logging():
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(
        filename=f"logs/train-{cfg.MISC.start_timestamp_str}.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    formatter = logging.Formatter(
        fmt="[%(levelname)s][%(asctime)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)
    return logger


def predictions_table():
    table = Table(show_header=True, header_style="hot_pink")
    table.add_column("Ground Truth", width=12)
    table.add_column("Predicted")
    table.border_style = "bright_yellow"
    table.columns[0].style = "violet"
    table.columns[1].style = "grey93"
    return table
