from flask import Flask, render_template, request, redirect, url_for, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from web_check import check_online
from url_fetcher import fetch_article_from_url
from nigerian_check import check_nigerian_context
from database import init_db, save_prediction, update_feedback, get_all_predictions, get_stats, get_monitored_articles, delete_prediction, delete_all_predictions
import threading
from news_monitor import scan_feeds_background

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load model
print("Loading model...")
model      = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
print("Model loaded!")

# Initialise database
init_db()

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
    words          = text.lower().split()
    emotional_count = sum(1 for w in words if w in emotional_words)
    emotional_score = max(0, 100 - (emotional_count * 15))
    sentences       = text.split('.')
    avg_len         = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

    if 15 <= avg_len <= 35:
        writing_score = 80
        writing_label = 'Professional writing style'
    elif avg_len < 10:
        writing_score = 40
        writing_label = 'Very short sentences'
    else:
        writing_score = 60
        writing_label = 'Moderate writing style'

    clickbait_count = sum(1 for p in clickbait_patterns if p in text.lower())
    clickbait_score = max(0, 100 - (clickbait_count * 25))
    ai_score        = ml_confidence if prediction == 1 else (100 - ml_confidence)

    return {
        'emotional' : { 'score': emotional_score, 'count': emotional_count },
        'writing'   : { 'score': writing_score, 'label': writing_label },
        'clickbait' : { 'score': clickbait_score, 'count': clickbait_count },
        'ai'        : { 'score': round(ai_score, 2) }
    }

def analyse(article, input_type='text', url=''):
    cleaned       = clean_text(article)
    vec           = vectorizer.transform([cleaned])
    prediction    = model.predict(vec)[0]
    probability   = model.predict_proba(vec)[0]
    ml_confidence = probability[prediction] * 100

    details         = get_credibility_details(article, prediction, ml_confidence)
    nigerian_result = check_nigerian_context(article)
    trusted_found, fake_found, all_results = check_online(article)

    ai_score = ml_confidence if prediction == 1 else (100 - ml_confidence)

    if nigerian_result['is_nigerian_content']:
        credibility = (
            details['emotional']['score']     * 0.15 +
            details['writing']['score']       * 0.10 +
            details['clickbait']['score']     * 0.15 +
            ai_score                          * 0.40 +
            nigerian_result['nigerian_score'] * 0.20
        )
    else:
        credibility = (
            details['emotional']['score'] * 0.20 +
            details['writing']['score']   * 0.15 +
            details['clickbait']['score'] * 0.20 +
            ai_score                      * 0.45
        )

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

    reasons = []
    if details['emotional']['count'] > 0:
        reasons.append("Contains " + str(details['emotional']['count']) + " emotional trigger word(s)")
    if details['clickbait']['count'] > 0:
        reasons.append("Contains " + str(details['clickbait']['count']) + " clickbait phrase(s)")
    if trusted_found:
        reasons.append("Found on " + str(len(trusted_found)) + " trusted source(s)")
    if fake_found:
        reasons.append("Found on " + str(len(fake_found)) + " known fake source(s)")
    if not trusted_found and not fake_found:
        reasons.append("Could not verify on any known news source")
    if nigerian_result['is_nigerian_content']:
        reasons.append("Nigerian content detected: " + nigerian_result['language_detected'])

    online_result = 'trusted' if trusted_found else 'fake' if fake_found else 'unverified'

    prediction_id = save_prediction({
        'input_type'     : input_type,
        'article_text'   : article,
        'url'            : url,
        'ai_prediction'  : 'FAKE' if prediction == 0 else 'REAL',
        'ai_confidence'  : round(ml_confidence, 2),
        'emotional_score': details['emotional']['score'],
        'writing_score'  : details['writing']['score'],
        'clickbait_score': details['clickbait']['score'],
        'nigerian_flag'  : 1 if nigerian_result['is_nigerian_content'] else 0,
        'online_result'  : online_result,
        'credibility'    : round(credibility, 2),
        'final_verdict'  : verdict
    })

    return {
        'prediction_id'  : prediction_id,
        'ai_prediction'  : 'FAKE NEWS' if prediction == 0 else 'REAL NEWS',
        'ai_confidence'  : round(ml_confidence, 2),
        'emotional_score': details['emotional']['score'],
        'writing_score'  : details['writing']['score'],
        'clickbait_score': details['clickbait']['score'],
        'ai_score'       : details['ai']['score'],
        'nigerian'       : nigerian_result,
        'trusted_found'  : trusted_found,
        'fake_found'     : fake_found,
        'all_results'    : all_results[:3],
        'credibility'    : round(credibility, 2),
        'verdict'        : verdict,
        'reasons'        : reasons,
        'article_preview': article[:300]
    }

@app.route('/', methods=['GET'])
def index():
    stats = get_stats()
    return render_template('index.html', stats=stats)

@app.route('/analyse', methods=['POST'])
def analyse_route():
    input_type = request.form.get('input_type', 'text')
    article    = ''
    url        = ''

    if input_type == 'url':
        url     = request.form.get('url', '').strip()
        article = fetch_article_from_url(url)
        if not article:
            return render_template('index.html',
                error="Could not fetch article from URL. Please try pasting the text instead.",
                stats=get_stats()
            )
    else:
        article = request.form.get('article', '').strip()

    if not article:
        return render_template('index.html',
            error="Please enter some text or a valid URL.",
            stats=get_stats()
        )

    result = analyse(article, input_type, url)
    return render_template('result.html', result=result)

@app.route('/feedback/<int:prediction_id>/<feedback>')
def feedback(prediction_id, feedback):
    update_feedback(prediction_id, feedback)

    if feedback == 'correct':
        label = 1
    else:
        label = 0

    try:
        from sklearn.linear_model import SGDClassifier
        from sklearn.feature_extraction.text import HashingVectorizer
        h_vec = HashingVectorizer(n_features=2**18, alternate_sign=False)
        try:
            import joblib as jl
            online_model = jl.load('online_model.pkl')
        except:
            online_model = SGDClassifier(loss='log_loss', random_state=42)

        import sqlite3
        conn = sqlite3.connect('fakenews.db')
        c    = conn.cursor()
        c.execute('SELECT article_text FROM predictions WHERE id = ?', (prediction_id,))
        row  = c.fetchone()
        conn.close()

        if row:
            cleaned = clean_text(row[0])
            vec2    = h_vec.transform([cleaned])
            online_model.partial_fit(vec2, [label], classes=[0, 1])
            jl.dump(online_model, 'online_model.pkl')
    except Exception as e:
        print("Feedback error:", str(e))

    return redirect(url_for('history'))

@app.route('/history')
def history():
    rows  = get_all_predictions()
    return render_template('history.html', predictions=rows)

@app.route('/monitor')
def monitor():
    articles = get_monitored_articles()
    return render_template('monitor.html', articles=articles)

@app.route('/scan', methods=['POST'])
def scan():
    thread = threading.Thread(target=scan_feeds_background)
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'Scan started!'})

@app.route('/delete/<int:prediction_id>')
def delete_single(prediction_id):
    delete_prediction(prediction_id)
    return redirect(url_for('history'))

@app.route('/delete-all', methods=['POST'])
def delete_all():
    delete_all_predictions()
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True)
