from .search_base import SearchBase
from googlesearch import search as google_f_search

class SearchGoogleFree(SearchBase):

    def get_tool_info(self):
        return {
            "description": """The Free Google Search Tool enables querying Google without requiring an API key. It retrieves search results and extracts relevant content from linked pages. 
            This tool provides a lightweight alternative for search-based tasks but comes with limitations:
            - Since it does not use authentication, it may be subject to unknown rate limits.
            - Excessive or frequent queries may result in temporary access restrictions.
            - Search results are obtained through the `googlesearch` library and are dependent on public availability.""",
            
            "inputs": {
                "query": {
                    "type": "str",
                    "description": "The search query to be executed on Google.",
                    "required": True
                },
                "num_search_pages": {
                    "type": "int",
                    "description": "The number of search results to retrieve (default is 5).",
                    "required": False
                },
                "max_content_words": {
                    "type": "int",
                    "description": "The maximum number of words extracted from each search result page (default is 500).",
                    "required": False
                }
            },
            
            "outputs": {
                "results": {
                    "type": "list[dict]",
                    "description": "A list of search results, each containing 'title', 'content' (truncated), and 'url'."
                },
                "error": {
                    "type": "str",
                    "description": "An error message if the search fails (optional)."
                }
            },
            
            "functionality": """Methods and their functionality:
            - `search(query: str)`: Executes a Google search and retrieves structured results.
            - `_scrape_page(url: str)`: Extracts relevant content from a given search result page.
            - `__init__(num_search_pages: int, max_content_words: int)`: Initializes default search parameters, including the number of results and content length limits.""",
            
            "interface": "search(query: str) -> dict with key 'results' (list of dicts) or 'error' (str)"
        }

    num_search_pages:int = 5
    max_content_words:int = 500

    
    def search(self, query: str) -> list:
        """
        Searches Google for the given query and retrieves content from multiple pages.

        Args:
            query (str): The search query.

        Returns:
            dict: Contains a list of search results (title, truncated content, and source URL).
        """
        results = []
        try:
            # Step 1: Get top search result links
            search_results = list(google_f_search(query, num_results=self.num_search_pages))
            if not search_results:
                return {"error": "No search results found."}

            # Step 2: Fetch content from each page
            for url in search_results:
                try:
                    title, content = self._scrape_page(url)
                    if content:  # Ensure valid content exists
                        results.append({
                            "title": title,
                            "content": ' '.join(content.split()[:self.max_content_words]) + " ...",
                            "url": url,
                        })
                except Exception:
                    continue  # Skip pages that cannot be processed

            return results
        
        except Exception as e:
            return {"error": str(e)}