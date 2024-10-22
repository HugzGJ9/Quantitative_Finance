import logging
from colorama import Fore, Style, init

# Initialize colorama for Windows compatibility (optional on Unix-based systems)
init(autoreset=True)

# Custom formatter class to add colors based on log levels
class ColoredFormatter(logging.Formatter):
    # Define colors for each log level
    COLORS = {
        logging.DEBUG: Fore.GREEN + Style.BRIGHT,  # Debug (allow access) in green
        logging.INFO: Fore.BLUE + Style.BRIGHT,    # Info in blue
        logging.WARNING: Fore.YELLOW + Style.BRIGHT, # Warning in orange (yellow)
        logging.ERROR: Fore.RED,    # Error in red
        logging.CRITICAL: Fore.RED + Style.BRIGHT  # Critical in bright red
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.WHITE)  # Default to white if level not found
        log_msg = super().format(record)
        return log_color + log_msg

class LOGGER():
    def __init__(self):
        # Create a logger
        self.logger = logging.getLogger('color_logger')
        self.logger.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Create a formatter with color support and custom date/time format
        formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')

        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(console_handler)

    def display_test(self):
        # Example log messages
        self.logger.debug('This is a debug message')  # Green
        self.logger.info('This is an info message')   # Blue
        self.logger.warning('This is a warning message')  # Yellow/Orange
        self.logger.error('This is an error message')  # Red
        self.logger.critical('This is a critical message')  # Bright Red

if __name__ == '__main__':
    new_logger = LOGGER()
    new_logger.display_test()
    new_logger.logger.info('This is an info message')