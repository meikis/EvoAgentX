import requests
from bs4 import BeautifulSoup
from .tool import Tool

class SearchBase(Tool):
    num_search_pages:int = 5
    max_content_words:int = 500

    def search(self, query: str) -> list:
        pass


    def _scrape_page(self, url: str) -> tuple:
        """
        Fetches the title and main text content from a web page.

        Args:
            url (str): The URL of the web page.

        Returns:
            tuple: (title, main textual content)
        """
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            return None, None

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = soup.title.string if soup.title else "No Title"

        # Extract text content (only from <p> tags)
        paragraphs = soup.find_all("p")
        text_content = " ".join([p.get_text(strip=True) for p in paragraphs])

        return title, text_content