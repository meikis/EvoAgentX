import requests
from .search_base import SearchBase

class SearchGoogle(SearchBase):
    def get_tool_info(self):
        return {
            "description": """The Google Search Tool utilizes the Google Custom Search API to perform structured search queries. 
            Unlike free alternatives, this tool requires an API key but provides more reliable and unrestricted access. 
            It ensures efficient retrieval of search results, extracting and summarizing relevant content from top-ranked pages.

            The tool ensures:
            - Secure and authenticated search queries using Google Custom Search API.
            - Structured and precise retrieval of search results based on the provided query.
            - Summarization and truncation of retrieved content for better readability.
            - Optimized output formatting, returning structured lists containing titles, URLs, and extracted content.
            
            Google Custom Search API: https://developers.google.com/custom-search/v1/overview
            Google Custom Search Engine: https://cse.google.com/cse/create/new""",
            
            "inputs": {
                "query": {
                    "type": "str",
                    "description": "The search query to be executed on Google.",
                    "required": True
                },
                "search_params": {
                    "type": "dict",
                    "description": "A dictionary containing API authentication details ('api_key' and 'search_engine_id').",
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
            - `search(query: str, search_params: dict)`: Sends a query to Google Custom Search API and retrieves structured search results.
            - `_scrape_page(url: str)`: Extracts content from a given search result page for further processing.
            - `__init__(num_search_pages: int, max_content_words: int)`: Initializes the default search parameters, including result count and content length restrictions.""",
            
            "interface": "search(query: str, search_params: dict) -> dict with key 'results' (list of dicts) or 'error' (str)"
        }
    
    num_search_pages:int = 5
    max_content_words:int = 500

    
    def search(self, query: str, search_params:dict) -> dict:
        results = []
        api_key = search_params["api_key"]
        search_engine_id = search_params["search_engine_id"]
        
        try:
            # Step 1: Query Google Custom Search API
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": api_key,
                "cx": search_engine_id,
                "q": query,
                "num": self.num_search_pages,
            }
            response = requests.get(search_url, params=params)
            data = response.json()

            if "items" not in data:
                return {"results": [], "error": "No search results found."}

            search_results = data["items"]

            # Step 2: Fetch content from each valid search result
            for item in search_results:
                url = item["link"]
                title = item["title"]
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

            return {"results": results, "error": None}

        except Exception as e:
            return {"results": [], "error": str(e)}
