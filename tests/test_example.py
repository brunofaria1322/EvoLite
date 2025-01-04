from evolite import __about__

def test_example():
    assert 45**2 == 2025

def test_version():
    assert __about__.__version__ == "0.0.1"