import requests
from newspaper import Article

def fetch_article_from_url(url):
    print("Fetching article from URL...")

    try:
        # Method 1 - Use newspaper3k
        article = Article(url)
        article.download()
        article.parse()

        title = article.title
        text  = article.text

        if text and len(text) > 100:
            full_text = title + ' ' + text
            print("Article fetched successfully!")
            print("Title   :", title[:80])
            print("Length  :", len(text), "characters")
            return full_text

        # Method 2 - Use requests if newspaper fails
        print("Trying alternative method...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove scripts and styles
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()

            # Get paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])

            if len(text) > 100:
                print("Article fetched via alternative method!")
                print("Length:", len(text), "characters")
                return text

        print("Could not extract article text from URL.")
        return None

    except Exception as e:
        print("Error fetching URL:", str(e))
        return None