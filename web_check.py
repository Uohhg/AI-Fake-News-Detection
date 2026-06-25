import requests
import re

NEWS_API_KEY = "13e7dcca3a8e40c8aa56dffe3dff4e9e"

TRUSTED_SOURCES = [
    'bbc.co.uk', 'bbc.com', 'reuters.com', 'apnews.com',
    'aljazeera.com', 'theguardian.com', 'punchng.com',
    'channelstv.com', 'vanguardngr.com', 'nytimes.com',
    'cnn.com', 'nbcnews.com', 'forbes.com',
    'bloomberg.com', 'washingtonpost.com', 'thisdaylive.com',
    'dailytrust.com', 'premiumtimesng.com', 'thecable.ng',
    'independent.co.uk', 'telegraph.co.uk', 'sky.com',
    'france24.com', 'dw.com', 'npr.org', 'cbsnews.com',
    'abcnews.go.com', 'time.com', 'economist.com',
    'ft.com', 'wsj.com', 'usatoday.com', 'politico.com',
    'thehill.com', 'axios.com', 'businessinsider.com',
    'haaretz.com', 'timesofisrael.com', 'jpost.com',
    'middleeasteye.net', 'arabnews.com'
]

FAKE_SOURCES = [
    'infowars.com',
    'beforeitsnews.com',
    'worldnewsdailyreport.com',
    'empirenews.net',
    'yournewswire.com'
]

def extract_keywords(text):
    noise_words = set([
        'the', 'a', 'an', 'is', 'are', 'was', 'were',
        'has', 'have', 'had', 'be', 'been', 'being',
        'in', 'on', 'at', 'to', 'for', 'of', 'and',
        'or', 'but', 'that', 'this', 'with', 'from',
        'said', 'says', 'will', 'would', 'could', 'should',
        'their', 'they', 'them', 'there', 'these', 'those',
        'which', 'when', 'where', 'who', 'how', 'what',
        'about', 'after', 'before', 'during', 'while',
        'into', 'onto', 'upon', 'over', 'under', 'between'
    ])
    words = re.sub(r'[^a-zA-Z\s]', '', text).lower().split()
    keywords = [w for w in words if w not in noise_words and len(w) > 3]
    return ' '.join(keywords[:5])

def search_newsapi(query):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'pageSize': 5,
            'sortBy': 'relevancy'
        }
        response = requests.get(url, params=params)
        data = response.json()
        if data.get('status') == 'ok':
            return data.get('articles', [])
        else:
            print("NewsAPI message:", data.get('message', ''))
            return []
    except Exception as e:
        print("Search error:", str(e))
        return []

def check_online(article_text):
    trusted_found = []
    fake_found = []
    all_results = []

    print("Extracting keywords and searching online...")

    keywords = extract_keywords(article_text)
    queries = [
        keywords,
        article_text[:80].strip(),
        article_text.split('.')[0]
    ]

    print("Search query used: " + keywords)

    for query in queries:
        if not query or len(query) < 5:
            continue

        articles = search_newsapi(query)

        if articles:
            for article in articles:
                source_url  = article.get('url', '')
                source_name = article.get('source', {}).get('name', '')
                title       = article.get('title', '')
                description = article.get('description', '')

                if source_url and source_url not in all_results:
                    all_results.append(source_url)

                for source in TRUSTED_SOURCES:
                    if source in source_url:
                        if source_url not in [t['url'] for t in trusted_found]:
                            trusted_found.append({
                                'url'         : source_url,
                                'title'       : title,
                                'source'      : source_name,
                                'description' : description
                            })

                for source in FAKE_SOURCES:
                    if source in source_url:
                        if source_url not in [f['url'] for f in fake_found]:
                            fake_found.append({
                                'url'         : source_url,
                                'title'       : title,
                                'source'      : source_name,
                                'description' : description
                            })

            if trusted_found:
                print("Found " + str(len(trusted_found)) + " related trusted article(s) online!")
                break

    if not all_results:
        print("No related articles found online for this topic.")

    return trusted_found, fake_found, all_results