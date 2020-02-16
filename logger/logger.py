import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level = logging.INFO):
	"""
	Seup logging configurations
	"""

	log_config = Path(log_config)
	if log_config.is_file():
		config = read_json(log_config)
		#print(config)
		#print(config['handlers'])
		#print("---------------------------\n")
		#print(config['handlers'].items())
		
		#modify logging paths based on run config
		for _, handler in config['handlers'].items():
			#print(handler)
			#print(type(handler))
			if 'filename' in handler:
				handler['filename'] = str(save_dir / handler['filename'])


		logging.config.dictConfig(config)

	else:
		print("Warning: logging configuration file is not found in {}.".format(log_config))
		logging.basicConfig(level=default_level)

