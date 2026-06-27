import sqlite3
from datetime import datetime

DB_PATH = 'fakenews.db'


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            input_type      TEXT,
            article_text    TEXT,
            url             TEXT,
            ai_prediction   TEXT,
            ai_confidence   REAL,
            emotional_score REAL,
            writing_score   REAL,
            clickbait_score REAL,
            nigerian_flag   INTEGER,
            online_result   TEXT,
            credibility     REAL,
            final_verdict   TEXT,
            feedback        TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS monitored_articles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            title       TEXT,
            source      TEXT,
            url         TEXT,
            prediction  TEXT,
            confidence  REAL,
            score       REAL
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialised!")


def save_prediction(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (
            timestamp, input_type, article_text, url,
            ai_prediction, ai_confidence,
            emotional_score, writing_score, clickbait_score,
            nigerian_flag, online_result,
            credibility, final_verdict, feedback
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data.get('input_type', 'text'),
        data.get('article_text', '')[:500],
        data.get('url', ''),
        data.get('ai_prediction', ''),
        data.get('ai_confidence', 0),
        data.get('emotional_score', 0),
        data.get('writing_score', 0),
        data.get('clickbait_score', 0),
        data.get('nigerian_flag', 0),
        data.get('online_result', ''),
        data.get('credibility', 0),
        data.get('final_verdict', ''),
        data.get('feedback', '')
    ))
    last_id = c.lastrowid
    conn.commit()
    conn.close()
    return last_id


def update_feedback(prediction_id, feedback):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'UPDATE predictions SET feedback = ? WHERE id = ?',
        (feedback, prediction_id)
    )
    conn.commit()
    conn.close()


def get_all_predictions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    return rows


def delete_prediction(prediction_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
    conn.commit()
    conn.close()


def delete_all_predictions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()


def save_monitored(article):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO monitored_articles (
            timestamp, title, source, url,
            prediction, confidence, score
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        article['timestamp'],
        article['title'][:200],
        article['source'],
        article['url'],
        article['prediction'],
        article['confidence'],
        article['score']
    ))
    conn.commit()
    conn.close()


def get_monitored_articles():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT * FROM monitored_articles ORDER BY id DESC LIMIT 50'
    )
    rows = c.fetchall()
    conn.close()
    return rows


def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM predictions')
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE final_verdict = 'FAKE NEWS'")
    fake = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE final_verdict = 'REAL NEWS'")
    real = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM predictions WHERE final_verdict = 'UNCERTAIN'")
    uncertain = c.fetchone()[0]
    c.execute('SELECT AVG(credibility) FROM predictions')
    avg_score = c.fetchone()[0] or 0
    conn.close()
    return {
        'total'    : total,
        'fake'     : fake,
        'real'     : real,
        'uncertain': uncertain,
        'avg_score': round(avg_score, 2)
    }