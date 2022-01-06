import collections
import logging
import os

import tqdm

from config.hparams import PARAMS


def get_logger(log_name):
    """
    :param log_name: log name
    :return: logger
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.
        See Also:
            > https://stackoverflow.com/questions/38543506
        """

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # log path
    hparams = PARAMS
    hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)
    log_file = hparams.logFile
    log_dir = hparams.logPath
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] --%(name)s-- %(levelname)s: %(message)s',
                                       datefmt='%y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] --%(name)s-- %(levelname)s: %(message)s',
                                          datefmt='%y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger