import pytest
import shutil


@pytest.fixture(scope="module")
def rm_strax_data():
    """Remove ./strax_test_data directory before and after initializing the
    TestCase."""
    shutil.rmtree("./strax_test_data", ignore_errors=True)
    yield
    shutil.rmtree("./strax_test_data", ignore_errors=True)
