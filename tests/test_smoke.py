
def test_imports():
    import models.dataset as d
    import models.train as t
    import models.infer as i
    assert hasattr(d, "build_datasets")
