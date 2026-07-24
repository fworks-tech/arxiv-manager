"""Tests for config loading."""

import yaml
from pathlib import Path


def test_load_config_exists(tmp_path):
    """Existing YAML file loads correctly."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "domains.yaml").write_text("domains:\n  - name: Computer Science\n")
    from arxiv_manager.config import CONFIG_DIR, load_config
    import arxiv_manager.config as cfg_mod
    original = cfg_mod.CONFIG_DIR
    cfg_mod.CONFIG_DIR = cfg_dir
    try:
        result = load_config("domains")
        assert "domains" in result
        assert result["domains"][0]["name"] == "Computer Science"
    finally:
        cfg_mod.CONFIG_DIR = original


def test_load_config_missing(tmp_path):
    """Missing config file returns empty dict."""
    from arxiv_manager.config import load_config
    result = load_config("nonexistent")
    assert result == {}


def test_get_domains_empty(tmp_path):
    """get_domains returns empty list when domains config is missing."""
    from arxiv_manager.config import CONFIG_DIR, get_domains
    import arxiv_manager.config as cfg_mod
    original = cfg_mod.CONFIG_DIR
    cfg_mod.CONFIG_DIR = tmp_path
    try:
        result = get_domains()
        assert result == []
    finally:
        cfg_mod.CONFIG_DIR = original


def test_get_domains_with_data(tmp_path):
    """get_domains returns domain list when config exists."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "domains.yaml").write_text(
        "domains:\n  - name: Physics\n  - name: Chemistry\n"
    )
    from arxiv_manager.config import get_domains
    import arxiv_manager.config as cfg_mod
    original = cfg_mod.CONFIG_DIR
    cfg_mod.CONFIG_DIR = cfg_dir
    try:
        result = get_domains()
        assert len(result) == 2
        assert result[1]["name"] == "Chemistry"
    finally:
        cfg_mod.CONFIG_DIR = original
