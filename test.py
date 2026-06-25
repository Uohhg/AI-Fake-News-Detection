import re
import joblib
import nltk
from nltk.corpus import stopwords
from web_check import check_online
from url_fetcher import fetch_article_from_url
from nigerian_check import check_nigerian_context

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def get_credibility_details(text, prediction, ml_confidence):
    details = {}

    emotional_words = [
        'shocking', 'unbelievable', 'secret', 'hidden', 'exposed',
        'banned', 'censored', 'urgent', 'breaking', 'alert',
        'warning', 'danger', 'crisis', 'conspiracy', 'hoax',
        'share', 'viral', 'truth', 'lies', 'corrupt', 'evil',
        'disgusting', 'outrage', 'horrifying', 'terrifying'
    ]
    words = text.lower().split()
    emotional_count = sum(1 for w in words if w in emotional_words)
    emotional_score = max(0, 100 - (emotional_count * 15))
    details['emotional'] = {
        'score': emotional_score,
        'count': emotional_count,
        'label': 'Low emotional language' if emotional_score > 60 else 'High emotional language'
    }

    sentences = text.split('.')
    avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    if 15 <= avg_len <= 35:
        writing_score = 80
        writing_label = 'Professional writing style'
    elif avg_len < 10:
        writing_score = 40
        writing_label = 'Very short sentences - possible clickbait'
    else:
        writing_score = 60
        writing_label = 'Moderate writing style'
    details['writing'] = {
        'score': writing_score,
        'label': writing_label
    }

    clickbait_patterns = [
        'you wont believe', 'this is why', 'what happened next',
        'share before', 'they dont want', 'the truth about',
        'wake up', 'open your eyes', 'mainstream media',
        'deleted', 'banned', 'they are hiding'
    ]
    text_lower = text.lower()
    clickbait_count = sum(1 for p in clickbait_patterns if p in text_lower)
    clickbait_score = max(0, 100 - (clickbait_count * 25))
    details['clickbait'] = {
        'score': clickbait_score,
        'count': clickbait_count,
        'label': 'No clickbait detected' if clickbait_score > 70 else 'Clickbait patterns detected'
    }

    ai_score = ml_confidence if prediction == 1 else (100 - ml_confidence)
    details['ai'] = {
        'score': round(ai_score, 2),
        'label': 'AI pattern analysis'
    }

    return details

def get_explanation(details, trusted_found, fake_found, nigerian_result):
    reasons = []

    if details['emotional']['count'] > 0:
        reasons.append(
            "Contains " + str(details['emotional']['count']) +
            " emotional trigger word(s) commonly found in fake news"
        )

    if details['clickbait']['count'] > 0:
        reasons.append(
            "Contains " + str(details['clickbait']['count']) +
            " clickbait phrase(s) such as share before deleted"
        )

    if details['writing']['score'] < 60:
        reasons.append(
            "Writing style is unusual - very short sentences common in misleading content"
        )

    if nigerian_result['is_nigerian_content']:
        for exp in nigerian_result['explanation']:
            reasons.append(exp)

    if trusted_found:
        reasons.append(
            "Found on " + str(len(trusted_found)) +
            " trusted source(s) including " + trusted_found[0]['source']
        )

    if fake_found:
        reasons.append(
            "Found on " + str(len(fake_found)) +
            " known fake news source(s) including " + fake_found[0]['source']
        )

    if not trusted_found and not fake_found:
        reasons.append("Could not verify topic on any known trusted news source")

    if details['ai']['score'] > 70:
        reasons.append(
            "AI pattern analysis strongly indicates " +
            ("real" if details['ai']['score'] > 60 else "fake") +
            " news writing style"
        )

    return reasons

# ===== LOAD MODEL =====
print("Loading model...")
model      = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
print("Model loaded!")
print("")

# ===== MAIN INTERFACE =====
print("========================================")
print("   AI-BASED FAKE NEWS DETECTOR")
print("   With Nigerian Language Support")
print("========================================")
print("")
print("How do you want to check the news?")
print("  1 = Paste article text")
print("  2 = Enter a URL link")
print("")
choice = input("Enter 1 or 2: ").strip()

article = ""

if choice == "2":
    print("")
    url = input("Paste the URL here: ").strip()
    article = fetch_article_from_url(url)
    if not article:
        print("Could not fetch article. Switching to text input.")
        choice = "1"

if choice == "1":
    print("")
    print("Paste your article below.")
    print("Press Enter twice when done.")
    print("")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    article = ' '.join(lines)

if not article or article.strip() == "":
    print("No text to analyse.")

else:
    # ===== PHASE 1 - AI PREDICTION =====
    cleaned       = clean_text(article)
    vec           = vectorizer.transform([cleaned])
    prediction    = model.predict(vec)[0]
    probability   = model.predict_proba(vec)[0]
    ml_confidence = probability[prediction] * 100

    print("")
    print("========================================")
    print("PHASE 1: AI PATTERN ANALYSIS")
    print("========================================")
    if prediction == 0:
        print("Pattern Result : FAKE NEWS")
    else:
        print("Pattern Result : REAL NEWS")
    print("Confidence     : " + str(round(ml_confidence, 2)) + "%")

    # ===== PHASE 2 - CREDIBILITY BREAKDOWN =====
    details = get_credibility_details(article, prediction, ml_confidence)

    print("")
    print("========================================")
    print("PHASE 2: CREDIBILITY BREAKDOWN")
    print("========================================")
    print("Emotional Language  : " + str(details['emotional']['score']) + "/100 - " + details['emotional']['label'])
    print("Writing Quality     : " + str(details['writing']['score'])   + "/100 - " + details['writing']['label'])
    print("Clickbait Detection : " + str(details['clickbait']['score']) + "/100 - " + details['clickbait']['label'])
    print("AI Pattern Score    : " + str(details['ai']['score'])        + "/100 - " + details['ai']['label'])

    # ===== PHASE 2b - NIGERIAN CONTEXT CHECK =====
    nigerian_result = check_nigerian_context(article)

    print("")
    print("========================================")
    print("PHASE 2b: NIGERIAN CONTEXT CHECK")
    print("========================================")
    if nigerian_result['is_nigerian_content']:
        print("Nigerian Content  : Detected!")
        print("Language Detected : " + nigerian_result['language_detected'])
        print("Nigerian Score    : " + str(nigerian_result['nigerian_score']) + "/100")
        for exp in nigerian_result['explanation']:
            print("  - " + exp)
    else:
        print("Nigerian Content  : Not detected")
        print("Standard analysis : Applied")

    # ===== PHASE 3 - ONLINE VERIFICATION =====
    trusted_found, fake_found, all_results = check_online(article)

    print("")
    print("========================================")
    print("PHASE 3: ONLINE SOURCE VERIFICATION")
    print("========================================")

    if trusted_found:
        print("Found on " + str(len(trusted_found)) + " trusted source(s):")
        for item in trusted_found:
            print("")
            print("  Source : " + item['source'])
            print("  Title  : " + item['title'])
            print("  URL    : " + item['url'])
    elif fake_found:
        print("Found on " + str(len(fake_found)) + " fake source(s):")
        for item in fake_found:
            print("")
            print("  Source : " + item['source'])
            print("  Title  : " + item['title'])
            print("  URL    : " + item['url'])
    else:
        print("Topic not found on any known news source.")
        if all_results:
            print("Related links found:")
            for url in all_results[:3]:
                print("  ? " + url)

    # ===== PHASE 4 - CREDIBILITY SCORE CALCULATION =====
    ai_score = ml_confidence if prediction == 1 else (100 - ml_confidence)

    if nigerian_result['is_nigerian_content']:
        credibility_score = (
            details['emotional']['score']     * 0.15 +
            details['writing']['score']       * 0.10 +
            details['clickbait']['score']     * 0.15 +
            ai_score                          * 0.40 +
            nigerian_result['nigerian_score'] * 0.20
        )
        print("\nNigerian context weighting applied")
    else:
        credibility_score = (
            details['emotional']['score'] * 0.20 +
            details['writing']['score']   * 0.15 +
            details['clickbait']['score'] * 0.20 +
            ai_score                      * 0.45
        )

    if trusted_found and fake_found:
        boost = len(trusted_found) * 20
        credibility_score = min(100, credibility_score + boost)
    elif trusted_found:
        boost = len(trusted_found) * 20
        credibility_score = min(100, credibility_score + boost)
    elif fake_found:
        penalty = len(fake_found) * 15
        credibility_score = max(0, credibility_score - penalty)

    # ===== PHASE 4 - EXPLANATION =====
    print("")
    print("========================================")
    print("PHASE 4: EXPLANATION")
    print("========================================")
    reasons = get_explanation(details, trusted_found, fake_found, nigerian_result)
    print("Why the system made this decision:")
    for i, reason in enumerate(reasons):
        print(str(i + 1) + ". " + reason)

    # ===== FINAL CREDIBILITY REPORT =====
    print("")
    print("========================================")
    print("FINAL CREDIBILITY REPORT")
    print("========================================")
    print("Emotional Language  : " + str(details['emotional']['score']) + "/100")
    print("Writing Quality     : " + str(details['writing']['score'])   + "/100")
    print("Clickbait Detection : " + str(details['clickbait']['score']) + "/100")
    print("AI Pattern Score    : " + str(round(ai_score, 2))            + "/100")
    if nigerian_result['is_nigerian_content']:
        print("Nigerian Score      : " + str(nigerian_result['nigerian_score']) + "/100")
    print("Online Verification : " + (
        "Trusted source found" if trusted_found else
        "Fake source found"   if fake_found    else
        "Not verified"
    ))
    print("----------------------------------------")
    print("CREDIBILITY SCORE   : " + str(round(credibility_score, 2)) + "/100")
    print("")

    if credibility_score >= 70:
        print("VERDICT : REAL NEWS")
    elif credibility_score >= 45:
        print("VERDICT : UNCERTAIN - needs manual verification")
    else:
        print("VERDICT : FAKE NEWS")

    print("========================================")

    # ===== PHASE 5 - FEEDBACK AND LEARNING =====
    print("")
    print("Was this prediction correct?")
    print("  y = Yes")
    print("  n = No")
    print("  s = Skip")

    feedback = input("\nEnter y, n or s: ").strip().lower()

    if feedback == 'y':
        try:
            from sklearn.linear_model import SGDClassifier
            from sklearn.feature_extraction.text import HashingVectorizer
            h_vec = HashingVectorizer(n_features=2**18, alternate_sign=False)
            try:
                online_model = joblib.load('online_model.pkl')
            except:
                online_model = SGDClassifier(loss='log_loss', random_state=42)
            vec2 = h_vec.transform([cleaned])
            online_model.partial_fit(vec2, [prediction], classes=[0, 1])
            joblib.dump(online_model, 'online_model.pkl')
            print("Thanks! Model reinforced.")
        except Exception as e:
            print("Could not update model:", str(e))

    elif feedback == 'n':
        try:
            correct_label = 1 - prediction
            from sklearn.linear_model import SGDClassifier
            from sklearn.feature_extraction.text import HashingVectorizer
            h_vec = HashingVectorizer(n_features=2**18, alternate_sign=False)
            try:
                online_model = joblib.load('online_model.pkl')
            except:
                online_model = SGDClassifier(loss='log_loss', random_state=42)
            vec2 = h_vec.transform([cleaned])
            online_model.partial_fit(vec2, [correct_label], classes=[0, 1])
            joblib.dump(online_model, 'online_model.pkl')
            print("Thanks! Model corrected and updated.")
        except Exception as e:
            print("Could not update model:", str(e))

    else:
        print("Feedback skipped.")

    print("")
    print("Detection complete!")