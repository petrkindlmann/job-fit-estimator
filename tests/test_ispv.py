from job_fit.data.ispv import IspvIndex


def test_index_loads_and_finds_isco():
    idx = IspvIndex.load_default()
    assert len(idx.iscos()) > 50

    target = "2512" if "2512" in idx.iscos() else next(iter(idx.iscos()))
    rec = idx.lookup(target)
    assert rec is not None
    assert rec.median > 0
    assert rec.d1 < rec.median < rec.d9


def test_lookup_with_rollup():
    idx = IspvIndex.load_default()
    iscos = idx.iscos()
    target = next(iter(iscos))
    rec = idx.lookup_with_rollup(target)
    assert rec is not None
    assert rec.isco_level in (2, 3, 4)


def test_lookup_unknown_returns_none():
    idx = IspvIndex.load_default()
    rec = idx.lookup("9999")
    assert rec is None
