def test_imports():
    from app.config.settings import FEEDS, PATHS
    from app.rag.graph import app
    from app.bandit.linucb import LinUCB
    assert len(FEEDS) > 0
    assert app is not None
    assert LinUCB(3, 5).select(__import__("numpy").zeros((5, 1))) in [0, 1, 2]
