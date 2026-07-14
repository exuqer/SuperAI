from server.core.settings import Settings


def test_debug_accepts_deployment_labels():
    assert Settings(debug="release").debug is False
    assert Settings(debug="development").debug is True
