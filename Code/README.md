# AI-Fake-News-Detection

A Python-based fake news detection system with credibility scoring, online verification, and Nigerian content support.

## Project Overview

This repository contains a fake news detection project built using machine learning and natural language processing. It combines model prediction, credibility analysis, online source verification, and Nigerian content detection to help determine whether news articles are likely real or fake.

## Key Features

- AI-based fake news classification using a trained machine learning model
- Credibility breakdown including emotional language, writing quality, and clickbait detection
- Nigerian content detection with language-specific context checks
- Online source verification against trusted and fake news sources
- Command-line interface for text and URL analysis
- Training and model serialization with `train_model.py`

## Live Demo

Try the application here: https://ai-fake-news-detection-2-mlxj.onrender.com

## Files and Structure

- `app.py` - Main application interface for running the news checker
- `train_model.py` - Script to train the fake news classification model
- `url_fetcher.py` - Fetches article content from URLs
- `web_check.py` - Verifies news sources online with NewsAPI and source lists
- `data/` - Includes datasets used for training and testing
- `fake_news_model.pkl`, `vectorizer.pkl` - Trained model and vectorizer files
- `online_model.pkl` - Optional online learning model updated with user feedback
- `templates/`, `static/`, `results/` - Supporting files for project reports and UI

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Uohhg/AI-Fake-News-Detection.git
   cd AI-Fake-News-Detection
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

> If `requirements.txt` is not present, install the main packages manually:
> `pandas`, `scikit-learn`, `joblib`, `nltk`, `newspaper3k`, `requests`, `beautifulsoup4`.

## Usage

Run the main application:
```bash
python app.py
```

Follow the prompts to analyze either pasted article text or a URL.

## Notes

- `data/Fake.csv` and `data/True.csv` are large dataset files; GitHub may warn about file size limits.
- If you want a cleaner repository, consider using Git LFS for large datasets and model files.
- The GitHub repo description itself can be edited from the repository settings page on GitHub.

## Recommended GitHub Repository Description

Use this description on GitHub:

> Fake news detection system with ML-based classification, credibility scoring, online verification, and Nigerian content support.

## License

Add a license file if needed.

