import os
import sys
import logging
from datetime import datetime


class Logger:
    def __init__(self, save_dir):
        self.save_dir = os.path.join(save_dir, "logs")
        os.makedirs(self.save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y_%B_%d___%H:%M:%S")
        self.log_file = os.path.join(self.save_dir, f'train_{timestamp}.log')

        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"kaggle_logger_{datetime.now().strftime('%H%M')}")
        logger.setLevel(logging.INFO)

        if logger.hasHandlers:
            logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s',
                                      datefmt='%Y %B %d %H:%M:%S'
                                      )
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def info(self, message):
        self.logger.info(message)
        
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)