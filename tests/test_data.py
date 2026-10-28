from signalfeed.data import read_behaviors


def test_real_format_preserves_impression_and_temporal_order(tmp_path):
    p = tmp_path / "behaviors.tsv"
    p.write_text(
        "2\tU1\t11/15/2019 10:00:00 AM\tN1 N2\tN3-1 N4-0\n1\tU1\t11/14/2019 10:00:00 AM\tN1\tN2-0 N3-1\n"
    )
    rows = read_behaviors(p)
    assert [r["id"] for r in rows] == ["1", "2"]
    assert rows[0]["history"] == ["N1"]
    assert rows[0]["items"] == ["N2", "N3"]
    assert rows[0]["labels"] == [0, 1]
