import feedparser
import joblib
import re
import nltk
import os
import csv
from datetime import datetime
from nltk.corpus import stopwords
from nigerian_check import check_nigerian_context

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ===== NEWS FEEDS =====
NEWS_FEEDS = {
    'BBC News'        : 'http://feeds.bbci.co.uk/news/rss.xml',
    'Reuters'         : 'https://feeds.reuters.com/reuters/topNews',
    'Punch Nigeria'   : 'https://punchng.com/feed/',
    'Vanguard Nigeria': 'https://www.vanguardngr.com/feed/',
    'Channels TV'     : 'https://www.channelstv.com/feed/',
    'Premium Times'   : 'https://www.premiumtimesng.com/feed',
    'The Cable'       : 'https://www.thecable.ng/feed',
    'Al Jazeera'      : 'https://www.aljazeera.com/xml/rss/all.xml',
    'AP News'         : 'https://apnews.com/rss',
    'CNN'             : 'http://rss.cnn.com/rss/edition.rss',
    'Guardian'        : 'https://www.theguardian.com/world/rss',
    'This Day'        : 'https://www.thisdaylive.com/feed/',
    'Daily Trust'     : 'https://dailytrust.com/feed',
}

# ===== LOAD MODEL =====
model      = None
vectorizer = None

def load_model():
    global model, vectorizer
    try:
        model      = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        print("Monitor: Model loaded!")
        return True
    except Exception as e:
        print("Monitor: Could not load model:", str(e))
        return False

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def get_credibility_score(text, prediction, ml_confidence):
    emotional_words = [
        'shocking', 'unbelievable', 'secret', 'hidden', 'exposed',
        'banned', 'censored', 'urgent', 'breaking', 'alert',
        'warning', 'danger', 'crisis', 'conspiracy', 'hoax',
        'share', 'viral', 'truth', 'lies', 'corrupt', 'evil',
        'disgusting', 'outrage', 'horrifying', 'terrifying'
    ]
    clickbait_patterns = [
        'you wont believe', 'share before', 'they dont want',
        'wake up', 'open your eyes', 'deleted', 'banned',
        'they are hiding', 'mainstream media', 'share now'
    ]
    words         = text.lower().split()
    emotional     = sum(1 for w in words if w in emotional_words)
    clickbait     = sum(1 for p in clickbait_patterns if p in text.lower())
    emotional_sc  = max(0, 100 - (emotional * 15))
    clickbait_sc  = max(0, 100 - (clickbait * 25))
    ai_score      = ml_confidence if prediction == 1 else (100 - ml_confidence)

    score = (
        emotional_sc * 0.20 +
        clickbait_sc * 0.20 +
        ai_score     * 0.60
    )
    return round(score, 2)

def analyse_article(title, text, source, url):
    global model, vectorizer

    if model is None or vectorizer is None:
        if not load_model():
            return None

    try:
        full_text = str(title) + ' ' + str(text)
        cleaned   = clean_text(full_text)

        if len(cleaned.strip()) < 20:
            return None

        vec           = vectorizer.transform([cleaned])
        prediction    = model.predict(vec)[0]
        probability   = model.predict_proba(vec)[0]
        ml_confidence = probability[prediction] * 100
        nigerian      = check_nigerian_context(full_text)
        score         = get_credibility_score(full_text, prediction, ml_confidence)

        return {
            'title'      : str(title)[:200],
            'source'     : source,
            'url'        : str(url),
            'prediction' : 'FAKE' if prediction == 0 else 'REAL',
            'confidence' : round(ml_confidence, 2),
            'score'      : score,
            'nigerian'   : nigerian['is_nigerian_content'],
            'timestamp'  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print("Analysis error:", str(e))
        return None

def save_to_csv(article):
    filepath    = 'flagged_articles.csv'
    file_exists = os.path.exists(filepath)
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'title', 'source',
                'prediction', 'confidence', 'score', 'url'
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp' : article['timestamp'],
                'title'     : article['title'][:100],
                'source'    : article['source'],
                'prediction': article['prediction'],
                'confidence': article['confidence'],
                'score'     : article['score'],
                'url'       : article['url']
            })
    except Exception as e:
        print("CSV save error:", str(e))

def scan_feeds():
    print("\n========================================")
    print("SCANNING NEWS FEEDS")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("========================================")

    flagged    = []
    real_count = 0
    fake_count = 0
    total      = 0
    errors     = 0

    for source_name, feed_url in NEWS_FEEDS.items():
        try:
            print("\nScanning:", source_name, "...")
            feed    = feedparser.parse(feed_url)
            entries = feed.entries[:5]

            if not entries:
                print("  No entries found for", source_name)
                continue

            for entry in entries:
                title = entry.get('title', '')
                text  = (
                    entry.get('summary', '') or
                    entry.get('description', '') or
                    entry.get('content', [{}])[0].get('value', '')
                )
                url   = entry.get('link', '')

                if not title:
                    continue

                result = analyse_article(title, text, source_name, url)

                if result:
                    total += 1
                    if result['prediction'] == 'FAKE':
                        fake_count += 1
                        flagged.append(result)
                        save_to_csv(result)
                        print("  FLAGGED :", result['title'][:60])
                        print("  Score   :", result['score'], "| Confidence:", result['confidence'], "%")
                    else:
                        real_count += 1

        except Exception as e:
            errors += 1
            print("  Error scanning", source_name, ":", str(e))
            continue

    # ===== SUMMARY REPORT =====
    print("\n========================================")
    print("SCAN COMPLETE")
    print("========================================")
    print("Total scanned      :", total)
    print("Real news          :", real_count)
    print("Suspicious flagged :", fake_count)
    print("Feed errors        :", errors)
    print("")

    if flagged:
        print("FLAGGED ARTICLES:")
        print("----------------------------------------")
        for i, article in enumerate(flagged):
            print(str(i + 1) + ". " + article['title'][:70])
            print("   Source     : " + article['source'])
            print("   Confidence : " + str(article['confidence']) + "%")
            print("   Score      : " + str(article['score']) + "/100")
            print("   URL        : " + article['url'])
            print("")
    else:
        print("No suspicious articles detected.")

    print("Results saved to: flagged_articles.csv")
    return flagged

def scan_feeds_background():
    try:
        from database import save_monitored
        print("\nBackground scan started...")

        for source_name, feed_url in NEWS_FEEDS.items():
            try:
                feed    = feedparser.parse(feed_url)
                entries = feed.entries[:5]

                for entry in entries:
                    title = entry.get('title', '')
                    text  = (
                        entry.get('summary', '') or
                        entry.get('description', '') or ''
                    )
                    url   = entry.get('link', '')

                    if not title:
                        continue

                    result = analyse_article(title, text, source_name, url)

                    if result and result['prediction'] == 'FAKE':
                        save_monitored(result)
                        print("Flagged:", result['title'][:60])

            except Exception as e:
                print("Feed error:", source_name, str(e))
                continue

        print("Background scan complete!")

    except Exception as e:
        print("Background scan error:", str(e))

if __name__ == "__main__":
    print("========================================")
    print("   REAL TIME NEWS FEED MONITOR")
    print("========================================")
    print("")
    print("Monitoring", len(NEWS_FEEDS), "news sources:")
    for name in NEWS_FEEDS:
        print("  - " + name)
    print("")

    if not load_model():
        print("Please run train_model.py first!")
        exit()

    print("Press Ctrl+C to stop")
    print("")

    try:
        import schedule
        import time
        scan_feeds()
        schedule.every(30).minutes.do(scan_feeds)
        while True:
            schedule.run_pending()
            time.sleep(60)
    except ImportError:
        print("Schedule library not found. Running single scan...")
        scan_feeds()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")