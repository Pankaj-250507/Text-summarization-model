import unittest

from .data_prep import make_tokenizer, worker


class TestDataPrep(unittest.TestCase):
    def test_tokenizer(self):
        # Test if the tokenizer is loaded correctly
        tokenizer = make_tokenizer()
        self.assertIsNotNone(tokenizer)
        self.assertTrue(hasattr(tokenizer, 'encode'))
        self.assertTrue(hasattr(tokenizer, 'from_pretrained'))

    def test_worker_function(self):
        #shld add after deciding dataset 
        ...
if __name__ == '__main__':
    unittest.main(verbosity = 2)