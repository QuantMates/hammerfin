def test_import():
    try:
        import hammerfin  # noqa: F401
    except ImportError:
        raise AssertionError("The package cannot be imported")
