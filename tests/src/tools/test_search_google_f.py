import unittest
from unittest.mock import patch
from evoagentx.tools.search_google_f import SearchGoogleFree

class TestSearchGoogleFree(unittest.TestCase):
    def setUp(self):
        self.search_tool = SearchGoogleFree(num_search_pages=2, max_content_words=100)

    @patch('evoagentx.tools.search_google_f.google_f_search')
    def test_search_success(self, mock_search):
        mock_search.return_value = ['https://www.python.org/']

        result = self.search_tool.search('Python')
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    @patch('evoagentx.tools.search_google_f.google_f_search')
    def test_search_no_results(self, mock_search):
        mock_search.return_value = []
        result = self.search_tool.search('NonExistentQuery')
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No search results found.')

if __name__ == '__main__':
    unittest.main()
