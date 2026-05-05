from job_fit.data.esco import EscoIndex


def test_index_loads():
    idx = EscoIndex.load_default()
    assert len(idx.candidates) >= 25


def test_search_developer_keywords():
    idx = EscoIndex.load_default()
    top = idx.search("senior python software engineer with 5 years backend", k=5)
    codes = [c.isco_code for c in top]
    assert "2512" in codes  # software developer should rank in top 5


def test_search_nurse_cs():
    idx = EscoIndex.load_default()
    top = idx.search("zdravotní sestra na pohotovosti", k=5)
    codes = [c.isco_code for c in top]
    assert "2221" in codes
