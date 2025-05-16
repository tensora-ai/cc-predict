import logging

logging.basicConfig(level=logging.INFO)


def get_logger(class_name: str):
    return logging.getLogger(class_name)
