from argocd import ArgocdClient
from config import load_config
from adaptive_sdk import Adaptive
import yaml
from loguru import logger
import time


def main():
    cfg = load_config()
    create_app(cfg)
    check_adaptive(cfg)

    time.sleep(5)

    delete_app(cfg)


# PreChecks in order to fail as fast as possible
def preflight_checks():
    pass


# Create ArgoApp and validate that its up & running
def create_app(cfg):
    # Usage
    client = ArgocdClient(
        server_url=cfg["ARGOCD_URL"],
        token=cfg["ARGOCD_TOKEN"],
    )

    # Load YAML file
    with open("resources/application.yaml", "r") as f:
        app_manifest = yaml.safe_load(f)

    for key, tag in [
        ("installPostgres", "latest"),
        ("controlPlane", cfg["DOCKER_TAG_VERSION"]),
        ("harmony", cfg["DOCKER_TAG_VERSION"]),
    ]:
        app_manifest["spec"]["source"]["helm"]["valuesObject"][key]["image"]["tag"] = (
            tag
        )

    try:
        app = client.create_app_from_manifest(app_manifest)
        client.wait_for_app_ready(
            app_name=app["metadata"]["name"], timeout=300, poll_interval=5
        )
    except TimeoutError as e:
        logger.error(f"Application did not become ready in time: {e}")
        exit(1)
    except RuntimeError as e:
        logger.error(f"Error while creating or monitoring the application: {e}")
        exit(1)
    logger.info("Argo Application has started")


# Check that adaptive is ok and api key is working
def check_adaptive(cfg):
    adaptive = Adaptive(
        base_url=cfg["ADAPTIVE_URL"],
        api_key=cfg["ADAPTIVE_API_KEY"],
    )

    try:
        cp = adaptive.compute_pools.list()
    except Exception as e:
        logger.error(f"Error while connecting to Adaptive API: {e}")
        exit(1)

    logger.info("Adaptive Control Plane is reachable and API key is valid")
    logger.info(f"Compute Pools {cp}")


def delete_app(cfg):
    client = ArgocdClient(
        server_url=cfg["ARGOCD_URL"],
        token=cfg["ARGOCD_TOKEN"],
    )
    app_name = "adaptive-control-plane"
    try:
        client.delete_app(app_name=app_name, cascade=True)
        logger.info(f"Application {app_name} deleted successfully.")
    except RuntimeError as e:
        logger.error(f"Error while deleting the application: {e}")
        exit(1)


if __name__ == "__main__":
    main()
