import logging
from app.core.config import settings

# Centralized logging configuration foundation
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)
