import pandas as pd
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

all_data = []

# ===== LOAD ISOT =====
print("Loading ISOT dataset...")
try:
    fake = pd.read_csv('data/Fake.csv')
    true = pd.read_csv('data/True.csv')
    fake['label'] = 0
    true['label'] = 1
    fake['combined'] = fake['title'].astype(str) + ' ' + fake['text'].astype(str)
    true['combined'] = true['title'].astype(str) + ' ' + true['text'].astype(str)
    isot_df = pd.concat([fake[['combined', 'label']], true[['combined', 'label']]])
    isot_df.columns = ['text', 'label']
    all_data.append(isot_df)
    print("ISOT loaded:", len(isot_df), "articles")
except Exception as e:
    print("ISOT error:", str(e))

# ===== LOAD FAKENEWSNET =====
print("Loading FakeNewsNet dataset...")
try:
    fn_files = {
        'data/politifact_fake.csv': 0,
        'data/politifact_real.csv': 1,
        'data/gossipcop_fake.csv': 0,
        'data/gossipcop_real.csv': 1
    }
    fn_frames = []
    for filepath, label in fn_files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            text_col = None
            for col in ['title', 'text', 'content', 'news_url']:
                if col in df.columns:
                    text_col = col
                    break
            if text_col:
                temp = df[[text_col]].copy()
                temp.columns = ['text']
                temp['label'] = label
                temp = temp[temp['text'].str.len() > 20]
                fn_frames.append(temp)
    if fn_frames:
        fn_df = pd.concat(fn_frames, ignore_index=True)
        all_data.append(fn_df)
        print("FakeNewsNet loaded:", len(fn_df), "articles")
except Exception as e:
    print("FakeNewsNet error:", str(e))

# ===== LOAD LIAR =====
print("Loading LIAR dataset...")
try:
    liar_frames = []
    fake_labels = ['pants-fire', 'false', 'barely-true']
    real_labels = ['half-true', 'mostly-true', 'true']

    for filepath in ['data/train.tsv', 'data/test.tsv', 'data/valid.tsv']:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep='\t', header=None)
            df = df[[1, 2]].copy()
            df.columns = ['label_text', 'text']
            df = df[df['label_text'].isin(fake_labels + real_labels)]
            df['label'] = df['label_text'].apply(
                lambda x: 0 if x in fake_labels else 1
            )
            # Duplicate LIAR entries since they are short statements
            # This gives them more weight in training
            df = pd.concat([df] * 3, ignore_index=True)
            liar_frames.append(df[['text', 'label']])

    if liar_frames:
        liar_df = pd.concat(liar_frames, ignore_index=True)
        all_data.append(liar_df)
        print("LIAR loaded:", len(liar_df), "articles")
except Exception as e:
    print("LIAR error:", str(e))

# ===== COMBINE =====
print("\nCombining datasets...")
df = pd.concat(all_data, ignore_index=True)
df = df[['text', 'label']].dropna()
df = df[df['text'].str.strip().str.len() > 20]

print("Total before balancing:", len(df))
print("Fake:", len(df[df['label'] == 0]))
print("Real:", len(df[df['label'] == 1]))

# ===== BALANCE =====
fake_df = df[df['label'] == 0]
real_df = df[df['label'] == 1]
min_size = min(len(fake_df), len(real_df))
fake_df = resample(fake_df, replace=False, n_samples=min_size, random_state=42)
real_df = resample(real_df, replace=False, n_samples=min_size, random_state=42)
df = pd.concat([fake_df, real_df])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter balancing:")
print("Fake:", len(df[df['label'] == 0]))
print("Real:", len(df[df['label'] == 1]))
print("Total:", len(df))

# ===== CLEAN =====
print("\nCleaning text...")
df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.len() > 10]

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print("Training:", len(X_train))
print("Testing :", len(X_test))

# ===== FEATURES =====
print("\nExtracting features...")
vectorizer = TfidfVectorizer(
    max_features=150000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.90,
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{2,}'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ===== TRAIN ALL MODELS AND PICK BEST =====
print("\nTraining models...")

models = {
    'SVM C=1.0': CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.0)),
    'SVM C=2.0': CalibratedClassifierCV(LinearSVC(max_iter=3000, C=2.0)),
    'SVM C=3.0': CalibratedClassifierCV(LinearSVC(max_iter=3000, C=3.0)),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced')
}

best_model = None
best_acc   = 0
best_name  = ""

for name, m in models.items():
    print("\nTraining", name, "...")
    m.fit(X_train_vec, y_train)
    preds = m.predict(X_test_vec)
    acc   = accuracy_score(y_test, preds)
    print(name, "accuracy:", round(acc * 100, 2), "%")

    if acc > best_acc:
        best_acc   = acc
        best_model = m
        best_name  = name

print("\n==========================")
print("BEST MODEL :", best_name)
print("ACCURACY   :", round(best_acc * 100, 2), "%")
print("==========================")
print(classification_report(
    y_test,
    best_model.predict(X_test_vec),
    target_names=['Fake', 'Real']
))

# ===== SAVE =====
joblib.dump(best_model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model saved!")
print("Vectorizer saved!")