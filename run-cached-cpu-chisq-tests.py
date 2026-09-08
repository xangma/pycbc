"""Run the existing chi-squared test using only cached GWTC-1 inputs."""
import sys

from astropy.utils.data import cache_contents
import pycbc.io
import pytest

cached = cache_contents()


def cached_get_file(url, **kwargs):
    if url not in cached:
        raise RuntimeError(f"Required URL is not in the local cache: {url}")
    return str(cached[url])


# These are test fixture adaptations, not changes to PyCBC source. The current
# catalog-directory URL is uncached; restrict lookup to the cached GWTC-1
# catalog and prevent every file request from making a network connection.
pycbc.io.get_file = cached_get_file
from pycbc.catalog import catalog
catalog.get_file = cached_get_file
catalog._catalogs = {"GWTC-1-confident": "LVK"}

sys.argv = ["pytest"]
raise SystemExit(pytest.main(["-q", "test/test_chisq.py", "--tb=short"]))
