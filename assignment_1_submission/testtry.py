# run_tests.py
import unittest

if __name__ == "__main__":
    # Discover and run the test case from the tests/test_loading.py file
    loader = unittest.TestLoader()
    #suite = loader.loadTestsFromName('tests.test_loading')
    #suite=loader.loadTestsFromName('tests.test_activation')
    #suite=loader.loadTestsFromName('tests.test_network')
    suite=loader.loadTestsFromName('tests.test_training')
    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(suite)
