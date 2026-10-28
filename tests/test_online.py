import numpy as np
from signalfeed.online_learning import online_matrix


def test_cold_user_features_are_finite_and_history_changes_interactions():
    vectors = np.eye(384, dtype="float32")[:3]
    cold = online_matrix(vectors, [0, 1], [])
    warm = online_matrix(vectors, [0, 1], [0])
    assert cold.shape == (2, 1158)
    assert np.isfinite(cold).all()
    assert cold[0, -6] == 0
    assert warm[0, -6] == 1
    assert warm[1, -6] == 0
    assert np.all(cold[:, -2:] == 0)


def test_bad_model_publication_falls_back_and_rollback_clears_model(
    tmp_path, monkeypatch
):
    import signalfeed.serving as serving

    monkeypatch.setattr(serving, "path", lambda name: tmp_path)
    service = serving.ContentService()
    marker = tmp_path / "active.json"
    marker.write_text("{invalid")
    assert service.active_ranker() is None
    previous = object()
    service.online_ranker = previous
    service.online_version = "old"
    assert service.active_ranker() is previous
    marker.unlink()
    assert service.active_ranker() is None
    assert service.online_version is None
