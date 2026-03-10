import os
import yaml

# logging
import logging
from src.logging_config import LoggingConfig

# configuration of logging
logging_config = LoggingConfig()
logging_config.setup_logging()

class ConfigLoader:
    def __init__(self, config_path='../.env_app'):
                # Lade .env-Datei
        tmp_path = os.path.abspath(config_path)
        #check file exist and readable
        
        with open(tmp_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.config_path = tmp_path
        
        
    def get_db_config(self):
        """Gibt die Datenbankkonfiguration zurück."""
        if 'database' in self.config:
            return self.config['database']
        logging.warning("Keine Datenbankkonfiguration in der Konfigurationsdatei gefunden.")
        return {}


    def get_logging_config(self):
        """Gibt die Logging-Konfiguration zurück."""
        if 'logging' in self.config:
            return self.config['logging']
        logging.warning("Keine Logging-Konfiguration in der Konfigurationsdatei gefunden.")
        return {}