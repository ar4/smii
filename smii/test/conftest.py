import json
import pytest

def pytest_addoption(parser):
    parser.addoption("--cpuprop_kwargs", action="store", default="{}",
                     help="options for CPU propagators as JSON")
    parser.addoption("--gpuprop_kwargs", action="store", default="{}",
                     help="options for GPU propagators as JSON")

@pytest.fixture
def cpuprop_kwargs(request):
    return json.loads(request.config.getoption("--cpuprop_kwargs"))

@pytest.fixture
def gpuprop_kwargs(request):
    return json.loads(request.config.getoption("--gpuprop_kwargs"))
