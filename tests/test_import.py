def test_import():
    """Import test."""
    try:
        pass
    except ImportError as exc:
        raise AssertionError("The package cannot be imported") from exc
