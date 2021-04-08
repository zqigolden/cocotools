import sys
import os
from loguru import logger
logger.remove()
logger.add(sys.stderr, level=os.environ.get('LOGLEVEL', 'INFO').upper())
