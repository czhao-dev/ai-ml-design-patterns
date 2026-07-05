"""Run the Gradio app."""

import gradio.networking as gradio_networking

from app.config import get_settings
from app.ui import build_interface


def _patch_gradio_container_url_check() -> None:
    original_url_ok = gradio_networking.url_ok

    def container_url_ok(url: str) -> bool:
        # Gradio binds to 0.0.0.0 for Cloud Run, but its self-check should
        # connect to a loopback address from inside the container.
        local_url = url.replace("://0.0.0.0:", "://127.0.0.1:", 1)
        return original_url_ok(local_url)

    gradio_networking.url_ok = container_url_ok


def main() -> None:
    settings = get_settings()
    _patch_gradio_container_url_check()
    app = build_interface()
    app.launch(
        server_name=settings.server_name,
        server_port=settings.server_port,
        share=False,
        show_error=True,
        show_api=False,
    )


if __name__ == "__main__":
    main()
