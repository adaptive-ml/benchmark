import os
import sys
from loguru import logger


def load_config():
    config = {}
    try:
        for env in [
            "ARGOCD_URL",
            "ARGOCD_TOKEN",
            "ADAPTIVE_URL",
            "ADAPTIVE_API_KEY",
            "DOCKER_TAG_VERSION",
        ]:
            config[env] = os.environ[env]
    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        sys.exit(1)

    return config
