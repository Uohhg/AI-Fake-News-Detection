import joblib
import re
import nltk
from nltk.corpus import stopwords
from web_check import check_online
from nigerian_check import check_nigerian_context

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

model      = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

def get_credibility_details(text, prediction, ml_confidence):
    emotional_words = [
        'shocking', 'unbelievable', 'secret', 'hidden', 'exposed',
        'banned', 'censored', 'urgent', 'breaking', 'alert',
        'warning', 'danger', 'crisis', 'conspiracy', 'hoax',
        'share', 'viral', 'truth', 'lies', 'corrupt', 'evil'
    ]
    clickbait_patterns = [
        'you wont believe', 'share before', 'they dont want',
        'wake up', 'open your eyes', 'deleted', 'banned',
        'they are hiding', 'mainstream media'
    ]
    words           = text.lower().split()
    emotional_count = sum(1 for w in words if w in emotional_words)
    emotional_score = max(0, 100 - (emotional_count * 15))
    sentences       = text.split('.')
    avg_len         = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

    if 15 <= avg_len <= 35:
        writing_score = 80
    elif avg_len < 10:
        writing_score = 40
    else:
        writing_score = 60

    clickbait_count = sum(1 for p in clickbait_patterns if p in text.lower())
    clickbait_score = max(0, 100 - (clickbait_count * 25))
    ai_score        = ml_confidence if prediction == 1 else (100 - ml_confidence)

    return {
        'emotional' : emotional_score,
        'writing'   : writing_score,
        'clickbait' : clickbait_score,
        'ai'        : round(ai_score, 2),
        'avg_sentence_len': round(avg_len, 1),
        'emotional_count': emotional_count,
        'clickbait_count': clickbait_count
    }

def diagnose(label, article):
    cleaned       = clean_text(article)
    vec           = vectorizer.transform([cleaned])
    prediction    = model.predict(vec)[0]
    probability   = model.predict_proba(vec)[0]
    ml_confidence = probability[prediction] * 100

    d = get_credibility_details(article, prediction, ml_confidence)
    nigerian_result = check_nigerian_context(article)
    trusted_found, fake_found, all_results = check_online(article)

    if nigerian_result['is_nigerian_content']:
        credibility = (
            d['emotional'] * 0.15 +
            d['writing']   * 0.10 +
            d['clickbait'] * 0.15 +
            d['ai']        * 0.40 +
            nigerian_result['nigerian_score'] * 0.20
        )
    else:
        credibility = (
            d['emotional'] * 0.20 +
            d['writing']   * 0.15 +
            d['clickbait'] * 0.20 +
            d['ai']        * 0.45
        )

    base_credibility = credibility

    if trusted_found and fake_found:
        credibility = min(100, credibility + len(trusted_found) * 20)
    elif trusted_found:
        credibility = min(100, credibility + len(trusted_found) * 20)
    elif fake_found:
        credibility = max(0, credibility - len(fake_found) * 15)

    if credibility >= 70:
        verdict = 'REAL NEWS'
    elif credibility >= 45:
        verdict = 'UNCERTAIN'
    else:
        verdict = 'FAKE NEWS'

    print("=" * 70)
    print("TEST CASE:", label)
    print("-" * 70)
    print(f"AI score          : {d['ai']}  (raw ML confidence: {round(ml_confidence,2)}%, predicted {'REAL' if prediction==1 else 'FAKE'})")
    print(f"Emotional score   : {d['emotional']}  ({d['emotional_count']} trigger word(s) found)")
    print(f"Writing score     : {d['writing']}  (avg sentence length: {d['avg_sentence_len']} words)")
    print(f"Clickbait score   : {d['clickbait']}  ({d['clickbait_count']} phrase(s) found)")
    print(f"Nigerian content  : {nigerian_result['is_nigerian_content']}")
    print(f"Trusted sources   : {len(trusted_found)}  | Fake sources: {len(fake_found)}")
    print(f"Base score (before online adjustment) : {round(base_credibility,2)}")
    print(f"Final score (after online adjustment) : {round(credibility,2)}")
    print(f">>> VERDICT: {verdict}")
    print()

# ===== TEST ARTICLES =====

diagnose("Reuters - interest rate news", """
Reuters reported that the United States Federal Reserve raised interest rates 
by 25 basis points on Wednesday. Federal Reserve Chair Jerome Powell confirmed 
the decision was unanimous among committee members and was aimed at bringing 
inflation back to the 2 percent target.
""")

diagnose("Obvious political fake news", """
SHOCKING: Government officials have confirmed that elections will be cancelled 
due to security concerns. A reliable source inside the presidency revealed the 
decision was made secretly last night. Share this before they delete it. The 
mainstream media is hiding this from you.
""")

diagnose("Borderline / ambiguous article", """
Officials say the new policy could affect thousands of workers nationwide. 
Critics argue the move was rushed, while supporters say it was necessary. 
The debate is expected to continue in coming weeks as both sides present 
their case to lawmakers.
""")

diagnose("Short factual statement", """
The company announced quarterly earnings today, beating analyst expectations 
by a wide margin according to the official report.
""")

diagnose("Local Nigerian story, no major coverage", """
Residents of a quiet neighborhood in Lagos report water shortages that have 
lasted for three days. Officials have not yet responded to inquiries about 
when supply will be restored.
""")