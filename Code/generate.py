import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay
)
from sklearn.utils import resample

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

os.makedirs('results', exist_ok=True)

def clean_text(text):
    text  = str(text).lower()
    text  = re.sub(r'http\S+|www\S+', '', text)
    text  = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# ===== LOAD DATA =====
print("Loading datasets...")
all_data = []

try:
    fake = pd.read_csv('data/Fake.csv')
    true = pd.read_csv('data/True.csv')
    fake['label'] = 0
    true['label'] = 1
    fake['combined'] = fake['title'].astype(str) + ' ' + fake['text'].astype(str)
    true['combined'] = true['title'].astype(str) + ' ' + true['text'].astype(str)
    isot = pd.concat([fake[['combined','label']], true[['combined','label']]])
    isot.columns = ['text','label']
    all_data.append(isot)
    print("ISOT loaded:", len(isot))
except Exception as e:
    print("ISOT error:", e)

try:
    fn_files = {
        'data/politifact_fake.csv': 0,
        'data/politifact_real.csv': 1,
        'data/gossipcop_fake.csv' : 0,
        'data/gossipcop_real.csv' : 1
    }
    fn_frames = []
    for fp, label in fn_files.items():
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            for col in ['title','text','content','news_url']:
                if col in df.columns:
                    temp = df[[col]].copy()
                    temp.columns = ['text']
                    temp['label'] = label
                    temp = temp[temp['text'].str.len() > 20]
                    fn_frames.append(temp)
                    break
    if fn_frames:
        fn_df = pd.concat(fn_frames, ignore_index=True)
        all_data.append(fn_df)
        print("FakeNewsNet loaded:", len(fn_df))
except Exception as e:
    print("FakeNewsNet error:", e)

try:
    liar_frames = []
    fake_labels = ['pants-fire','false','barely-true']
    real_labels = ['half-true','mostly-true','true']
    for fp in ['data/train.tsv','data/test.tsv','data/valid.tsv']:
        if os.path.exists(fp):
            df = pd.read_csv(fp, sep='\t', header=None)
            df = df[[1,2]].copy()
            df.columns = ['label_text','text']
            df = df[df['label_text'].isin(fake_labels+real_labels)]
            df['label'] = df['label_text'].apply(
                lambda x: 0 if x in fake_labels else 1)
            df = pd.concat([df]*3, ignore_index=True)
            liar_frames.append(df[['text','label']])
    if liar_frames:
        liar_df = pd.concat(liar_frames, ignore_index=True)
        all_data.append(liar_df)
        print("LIAR loaded:", len(liar_df))
except Exception as e:
    print("LIAR error:", e)

df = pd.concat(all_data, ignore_index=True)
df = df[['text','label']].dropna()
df = df[df['text'].str.strip().str.len() > 20]

fake_df = df[df['label']==0]
real_df = df[df['label']==1]
min_size = min(len(fake_df), len(real_df))
fake_df  = resample(fake_df, replace=False, n_samples=min_size, random_state=42)
real_df  = resample(real_df, replace=False, n_samples=min_size, random_state=42)
df = pd.concat([fake_df, real_df]).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nCleaning text...")
df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.len() > 10]

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label'])

print("Training:", len(X_train), "| Testing:", len(X_test))

vectorizer = TfidfVectorizer(
    max_features=150000, ngram_range=(1,3),
    min_df=2, max_df=0.90,
    sublinear_tf=True, strip_accents='unicode',
    analyzer='word', token_pattern=r'\w{2,}')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ===== TRAIN ALL THREE MODELS =====
print("\nTraining all models...")

models = {
    'Naive Bayes'        : MultinomialNB(alpha=0.1),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=5.0, class_weight='balanced'),
    'SVM'                : CalibratedClassifierCV(LinearSVC(max_iter=2000, C=2.0))
}

results     = {}
all_metrics = {}

for name, model in models.items():
    print("Training", name, "...")
    model.fit(X_train_vec, y_train)
    preds  = model.predict(X_test_vec)
    probs  = model.predict_proba(X_test_vec)[:,1]
    acc    = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds,
                target_names=['Fake','Real'], output_dict=True)
    results[name]     = {'model':model,'preds':preds,'probs':probs,'acc':acc}
    all_metrics[name] = {
        'Accuracy' : round(acc*100, 2),
        'Precision': round(report['weighted avg']['precision']*100, 2),
        'Recall'   : round(report['weighted avg']['recall']*100, 2),
        'F1-Score' : round(report['weighted avg']['f1-score']*100, 2)
    }
    print(name, "accuracy:", round(acc*100,2), "%")

# ===== FIGURE 1: MODEL COMPARISON BAR CHART =====
print("\nGenerating Figure 1: Model Comparison Bar Chart...")
metrics_df = pd.DataFrame(all_metrics).T
fig, ax = plt.subplots(figsize=(10, 6))
x     = np.arange(len(metrics_df.index))
width = 0.2
colors = ['#1a1a2e','#4a4a7e','#8a8abe','#c0c0de']

for i, col in enumerate(metrics_df.columns):
    bars = ax.bar(x + i*width, metrics_df[col],
                  width, label=col, color=colors[i], alpha=0.9)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%',
            xy=(bar.get_x()+bar.get_width()/2, height),
            xytext=(0,3), textcoords="offset points",
            ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 1: Model Performance Comparison\n(Naive Bayes vs Logistic Regression vs SVM)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(metrics_df.index, fontsize=11)
ax.set_ylim(0, 115)
ax.legend(loc='upper left', fontsize=10)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('results/figure1_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figure1_model_comparison.png")

# ===== FIGURE 2: ACCURACY COMPARISON LINE CHART =====
print("Generating Figure 2: Accuracy Line Chart...")
fig, ax = plt.subplots(figsize=(8, 5))
model_names = list(all_metrics.keys())
accuracies  = [all_metrics[m]['Accuracy'] for m in model_names]
ax.plot(model_names, accuracies, 'o-',
        color='#1a1a2e', linewidth=2.5, markersize=10,
        markerfacecolor='white', markeredgewidth=2.5)
for i, (name, acc) in enumerate(zip(model_names, accuracies)):
    ax.annotate(f'{acc}%',
        xy=(i, acc), xytext=(0, 12),
        textcoords="offset points",
        ha='center', fontsize=11, fontweight='bold', color='#1a1a2e')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Figure 2: Accuracy Comparison Across Models',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(80, 105)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('results/figure2_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figure2_accuracy_comparison.png")

# ===== FIGURE 3: CONFUSION MATRIX FOR SVM =====
print("Generating Figure 3: SVM Confusion Matrix...")
svm_preds = results['SVM']['preds']
cm        = confusion_matrix(y_test, svm_preds)
fig, ax   = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Fake','Real'])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Figure 3: Confusion Matrix — SVM Model',
             fontsize=13, fontweight='bold', pad=15)
tn,fp,fn,tp = cm.ravel()
ax.set_xlabel(f'Predicted Label\n\nTP={tp}  TN={tn}  FP={fp}  FN={fn}',
              fontsize=11)
plt.tight_layout()
plt.savefig('results/figure3_confusion_matrix_svm.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figure3_confusion_matrix_svm.png")

# ===== FIGURE 4: ROC CURVES FOR ALL MODELS =====
print("Generating Figure 4: ROC Curves...")
fig, ax = plt.subplots(figsize=(8, 6))
line_styles = ['-','--','-.']
roc_colors  = ['#1a1a2e','#e74c3c','#27ae60']

for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['probs'])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, linestyle=line_styles[i],
            color=roc_colors[i], linewidth=2.5,
            label=f'{name} (AUC = {roc_auc:.4f})')

ax.plot([0,1],[0,1],'k--',linewidth=1,alpha=0.5,label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('Figure 4: ROC Curves — All Models',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.yaxis.grid(True, alpha=0.3)
ax.xaxis.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figure4_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figure4_roc_curves.png")

# ===== FIGURE 5: PRECISION RECALL F1 GROUPED =====
print("Generating Figure 5: Precision Recall F1 Chart...")
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
metric_names = ['Precision','Recall','F1-Score']
bar_colors   = ['#1a1a2e','#4a4a7e','#8a8abe']

for i, metric in enumerate(metric_names):
    vals  = [all_metrics[m][metric] for m in model_names]
    bars  = axes[i].bar(model_names, vals,
                         color=bar_colors, alpha=0.9, width=0.5)
    for bar in bars:
        height = bar.get_height()
        axes[i].annotate(f'{height}%',
            xy=(bar.get_x()+bar.get_width()/2, height),
            xytext=(0,3), textcoords="offset points",
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[i].set_title(metric, fontsize=12, fontweight='bold')
    axes[i].set_ylim(0, 115)
    axes[i].yaxis.grid(True, alpha=0.3)
    axes[i].set_axisbelow(True)
    axes[i].tick_params(axis='x', labelsize=9)

fig.suptitle('Figure 5: Precision, Recall and F1-Score Comparison',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figure5_precision_recall_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figure5_precision_recall_f1.png")

# ===== PRINT FULL COMPARISON TABLE =====
print("\n========================================")
print("FULL MODEL COMPARISON TABLE")
print("========================================")
print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-"*65)
for name, m in all_metrics.items():
    print(f"{name:<22} {str(m['Accuracy'])+'%':>10} {str(m['Precision'])+'%':>10} {str(m['Recall'])+'%':>10} {str(m['F1-Score'])+'%':>10}")
print("========================================")
print("\nAll figures saved to results/ folder!")
print("Use these in Chapter 4 of your project report.")