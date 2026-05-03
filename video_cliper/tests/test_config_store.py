import config_store


def test_save_and_get_last_export_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config_store, "config_path", lambda: tmp_path / "cfg.json")
    config_store.save_last_export_dir(str(tmp_path / "outdir"))
    assert config_store.get_last_export_dir() == str(tmp_path / "outdir")


def test_get_last_export_dir_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config_store, "config_path", lambda: tmp_path / "nope.json")
    assert config_store.get_last_export_dir() is None
