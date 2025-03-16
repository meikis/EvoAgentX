import unittest
import os
from unittest.mock import patch
from evoagentx.tools.search_google import SearchGoogle

class TestSearchGoogle(unittest.TestCase):
    def setUp(self):
        self.search_tool = SearchGoogle(num_search_pages=2, max_content_words=100)
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

    @patch('evoagentx.tools.search_google.requests.get')
    @patch('evoagentx.tools.search_base.SearchBase._scrape_page')
    def test_search_success(self, mock_scrape_page, mock_get):
        mock_get.return_value.json.return_value = {
            'items': [
                {'title': 'Python', 'link': 'https://www.python.org/'}
            ]
        }
        mock_scrape_page.return_value = ('Python', 'Python is a programming language...')

        result = self.search_tool.search('Python', {'api_key': self.api_key, 'search_engine_id': self.search_engine_id})
        self.assertIn('results', result)
        self.assertIsNone(result['error'])
        self.assertEqual(len(result['results']), 1)
        self.assertIn('Python', result['results'][0]['title'])

    @patch('evoagentx.tools.search_google.requests.get')
    def test_search_no_results(self, mock_get):
        mock_get.return_value.json.return_value = {}
        result = self.search_tool.search('NonExistentQuery', {'api_key': self.api_key, 'search_engine_id': self.search_engine_id})
        self.assertIn('results', result)
        self.assertEqual(result['results'], [])
        self.assertEqual(result['error'], 'No search results found.')

    @patch('evoagentx.tools.search_google.requests.get')
    def test_search_api_error(self, mock_get):
        mock_get.side_effect = Exception("API Error")
        result = self.search_tool.search('Python', {'api_key': self.api_key, 'search_engine_id': self.search_engine_id})
        self.assertIn('results', result)
        self.assertEqual(result['results'], [])
        self.assertEqual(result['error'], 'API Error')

if __name__ == '__main__':
    unittest.main()

