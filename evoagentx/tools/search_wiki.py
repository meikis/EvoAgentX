import wikipedia
from .search_base import SearchBase


class SearchWiki(SearchBase):

    def get_tool_info(self):
        return {
            "description": """The Wikipedia Search Tool enables querying Wikipedia to find relevant articles and extract key information. 
            It searches for Wikipedia pages based on a given query, retrieves article summaries, and extracts content in a structured format.
            
            The tool ensures:
            - Reliable Wikipedia search results based on the provided query.
            - Extraction of key details such as article title, summary, truncated full content, and the Wikipedia page link.
            - Filtering of ambiguous or non-existent pages to return the most relevant information.
            - Flexible result customization, allowing control over the number of search results, content length, and summary size.""",
            
            "inputs": {
                "query": {
                    "type": "str",
                    "description": "The search query to look up on Wikipedia.",
                    "required": True
                },
                "num_search_pages": {
                    "type": "int",
                    "description": "The number of Wikipedia search results to retrieve (default is 5).",
                    "required": False
                },
                "max_content_words": {
                    "type": "int",
                    "description": "The maximum number of words to retain from the full Wikipedia page content (default is 500).",
                    "required": False
                },
                "max_sentences": {
                    "type": "int",
                    "description": "The maximum number of sentences in the summary returned from Wikipedia (default is 15).",
                    "required": False
                }
            },
            
            "outputs": {
                "results": {
                    "type": "list[dict]",
                    "description": "A list of search results, each containing 'title', 'summary', 'content' (truncated), and 'url'."
                },
                "error": {
                    "type": "str",
                    "description": "An error message if the search fails (optional)."
                }
            },
            
            "functionality": """Methods and their functionality:
            - `search(query: str, max_sentences: int)`: Searches Wikipedia and retrieves structured results.
            - `__init__(num_search_pages: int, max_content_words: int)`: Initializes default search parameters, including the number of results and content length limits.""",
            
            "interface": "search(query: str, max_sentences: int) -> dict with key 'results' (list of dicts) or 'error' (str)"
        }

    num_search_pages:int = 5
    max_content_words:int = 500

    def search(self, query: str, max_sentences: int = 15) -> list:
        """
        Searches Wikipedia for the given query and returns the summary and truncated full content.

        Args:
            query (str): The search query.
            max_sentences (int): Maximum number of sentences in the summary.

        Returns:
            dict: A dictionary with the title, summary, truncated content, and Wikipedia page link.
        """
        try:
            # Search for top matching titles
            search_results = wikipedia.search(query, results=self.num_search_pages)
            if not search_results:
                return {"error": "No search results found."}

            # Try fetching the best available page
            results = []
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(title, sentences=max_sentences)

                    # Truncate the full content to the first max_content_words words
                    content = ' '.join(page.content.split()[:self.max_content_words])

                    results.append({
                        "title": page.title,
                        "summary": summary,
                        "content": content + " ...",
                        "url": page.url,
                    })
                except wikipedia.exceptions.DisambiguationError:
                    # Skip ambiguous results and try the next
                    continue
                except wikipedia.exceptions.PageError:
                    # Skip non-existing pages and try the next
                    continue

            return {"results": results}
        
        except Exception as e:
            return {"error": str(e)}
    