from hict import HiCTClient, HiCTJVMClient


def test_hict_facade_aliases_primary_client():
    assert HiCTClient is HiCTJVMClient
