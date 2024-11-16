import pytest


# Add a custom command-line option
def pytest_addoption(parser):
    parser.addoption("--recording_directory",
                     action="store",
                     default="../test_recording",
                     help="Neon Native Recording Data to test against")


# Fixture that uses the custom command-line option
@pytest.fixture()
def recording_directory(request):
    return request.config.getoption("--recording_directory")
