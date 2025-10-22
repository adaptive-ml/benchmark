import requests
import time
from loguru import logger


class ArgocdClient:
    def __init__(self, server_url, username=None, password=None, token=None):
        """
        Initialize ArgoCD client.

        Args:
            server_url: ArgoCD server URL (e.g., 'https://argocd.example.com')
            username: ArgoCD username (if using password auth)
            password: ArgoCD password (if using password auth)
            token: ArgoCD auth token (if already have one)
        """
        self.server_url = server_url.rstrip("/")
        self.token = token

        if not token and username and password:
            self.token = self._get_token(username, password)

    def _get_token(self, username, password):
        """Get authentication token from ArgoCD."""
        url = f"{self.server_url}/api/v1/session"
        payload = {"username": username, "password": password}

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()["token"]

    def delete_app(self, app_name, cascade=True):
        """
        Delete ArgoCD application from name.

        Args:
            app_name: app name as a string
            cascade: boolean on whether to also delete k8s resources owned by the app
        """
        url = f"{self.server_url}/api/v1/applications/{app_name}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = requests.delete(
            url,
            params={
                "cascade": cascade,
            },
            headers=headers,
        )
        if response.status_code in [200]:
            logger.info(f"Application '{app_name}' deleted successfully")
            return response.json()
        else:
            logger.error(f"Error deleting application: {response.status_code}")
            response.raise_for_status()

    def create_app_from_manifest(self, app_manifest):
        """
        Create ArgoCD application from dict.

        Args:
            app_manifest: Dict from YAML
        """
        app_name = app_manifest["metadata"]["name"]
        app_spec = app_manifest["spec"]

        url = f"{self.server_url}/api/v1/applications"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        payload = {"metadata": app_manifest["metadata"], "spec": app_spec}

        response = requests.post(
            url,
            params={
                "upsert": True,
            },
            json=payload,
            headers=headers,
        )

        if response.status_code in [200, 201]:
            logger.info(f"Application '{app_name}' created successfully")
            return response.json()
        else:
            logger.error(f"Error creating application: {response.status_code}")
            response.raise_for_status()

    def get_app_status(self, app_name):
        """
        Get application status.

        Args:
            app_name: Name of the application
        """
        url = f"{self.server_url}/api/v1/applications/{app_name}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def wait_for_app_ready(
        self,
        app_name,
        timeout=300,
        poll_interval=5,
        required_health_status="Healthy",
        required_sync_status="Synced",
    ):
        """
        Wait for application to be ready.

        Args:
            app_name: Name of the application
            timeout: Maximum time to wait in seconds (default: 300)
            poll_interval: Time between status checks in seconds (default: 5)
            required_health_status: Expected health status (default: "Healthy")
            required_sync_status: Expected sync status (default: "Synced")

        Raises:
            TimeoutError: If application is not ready within timeout
            RuntimeError: If application enters a degraded state
        """
        start_time = time.time()

        logger.info(f"Waiting for application '{app_name}' to be ready...")

        while True:
            elapsed_time = time.time() - start_time

            if elapsed_time > timeout:
                raise TimeoutError(
                    f"Application '{app_name}' did not become ready within {timeout} seconds"
                )

            try:
                app_status = self.get_app_status(app_name)

                health_status = (
                    app_status.get("status", {})
                    .get("health", {})
                    .get("status", "Unknown")
                )
                sync_status = (
                    app_status.get("status", {})
                    .get("sync", {})
                    .get("status", "Unknown")
                )

                logger.info(
                    f"[{int(elapsed_time)}s] Health: {health_status}, Sync: {sync_status}"
                )

                # Check for degraded states
                if health_status == "Degraded":
                    raise RuntimeError(f"Application '{app_name}' is in Degraded state")

                if sync_status == "OutOfSync" and elapsed_time > 60:
                    # Allow some time for initial sync
                    logger.Warning(
                        f"Warning: Application has been OutOfSync for {int(elapsed_time)}s"
                    )

                # Check if ready
                if (
                    health_status == required_health_status
                    and sync_status == required_sync_status
                ):
                    logger.info(f"Application '{app_name}' is ready!")
                    return app_status

            except requests.exceptions.RequestException as e:
                logger.error(f"Error checking status: {e}")

            time.sleep(poll_interval)
