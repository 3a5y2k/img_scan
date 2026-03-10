import os
import logging
import logging.config


class LoggingConfig:
    def __init__(self, log_path=''):

        # define logging-configuration
        self.LOGGING_CONFIG = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
            },
            'handlers': {
                'file_handler': {
                    'class': 'logging.FileHandler',
                    'formatter': 'standard',
                    'filename': os.path.join(os.getcwd(), log_path+'my_app.log'),
                },
                'exception_handler': {
                    'class': 'logging.FileHandler',
                    'formatter': 'standard',
                    'filename': os.path.join(os.getcwd(), log_path+'exceptions.log'),
                },
            },
            'loggers': {
                '': {
                    'handlers': ['file_handler'],
                    'level': 'DEBUG',
                    'propagate': True
                },
                'exceptions': {
                    'handlers': ['exception_handler'],
                    'level': 'ERROR',
                    'propagate': False
                }
            }
        }

        # Konfiguriere Logging
        self.setup_logging()

    def setup_logging(self):
        """Konfiguriert den Logger basierend auf der `LOGGING_CONFIG`."""
        logging.config.dictConfig(self.LOGGING_CONFIG)

    def get_logging(self, name='__main__'):
        """Gibt einen Logger mit dem angegebenen Namen zurück."""
        return logging.getLogger(name)