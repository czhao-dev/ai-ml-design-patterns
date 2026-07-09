"""Tests for app/main.py."""

import gradio.networking as gradio_networking

from app.main import _patch_gradio_container_url_check


def test_patch_rewrites_0_0_0_0_to_loopback_before_checking(monkeypatch):
    seen_urls = []

    def fake_url_ok(url: str) -> bool:
        seen_urls.append(url)
        return True

    monkeypatch.setattr(gradio_networking, "url_ok", fake_url_ok)

    _patch_gradio_container_url_check()
    result = gradio_networking.url_ok("http://0.0.0.0:7860")

    assert result is True
    assert seen_urls == ["http://127.0.0.1:7860"]


def test_patch_leaves_other_hosts_unchanged(monkeypatch):
    seen_urls = []

    def fake_url_ok(url: str) -> bool:
        seen_urls.append(url)
        return True

    monkeypatch.setattr(gradio_networking, "url_ok", fake_url_ok)

    _patch_gradio_container_url_check()
    gradio_networking.url_ok("http://example.com:7860")

    assert seen_urls == ["http://example.com:7860"]
