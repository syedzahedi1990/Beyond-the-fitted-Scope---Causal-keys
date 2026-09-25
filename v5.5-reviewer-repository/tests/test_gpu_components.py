import unittest


def load_tests(loader, tests, pattern):
    tests.addTests(loader.loadTestsFromName("gpu.tests.test_components"))
    tests.addTests(loader.loadTestsFromName("gpu.tests.test_key_scope"))
    return tests
