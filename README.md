# Amazon Reviews Sentiment Analysis

A natural language processing project that classifies Amazon product reviews as positive or negative. It compares a rule-based baseline (VADER) with machine learning models trained on TF-IDF text features, and shows how much the trained models improve on it.

## Results

| Approach | Accuracy (same 2,000-review test set) |
| :--- | :--- |
| VADER, rule-based, no training | 70.7% |
| Naive Bayes + TF-IDF | 84.3% |
| Linear SVM + TF-IDF | 85.5% |
| Logistic Regression + TF-IDF | 85.6% |
| **Logistic Regression + TF-IDF, tuned with GridSearchCV** | **86.1%** |

The tuned model has balanced precision and recall on both classes (F1 of about 0.86 each).

## What it does

- **Explores the data.** 10,000 labelled reviews, almost perfectly balanced between positive and negative.
- **Turns text into features.** TF-IDF over unigrams and bigrams, with stop-word removal. The vectorizer is fit on training data only, so there is no leakage into the test set.
- **Compares models.** Logistic Regression, Naive Bayes and a linear SVM.
- **Tunes the best model.** GridSearchCV with 5-fold cross-validation over n-gram range, minimum document frequency and regularization strength.
- **Explains the model.** Lists the words that push a review towards positive or negative, such as "great", "excellent" and "love" versus "boring", "waste" and "worst".
- **Benchmarks against VADER.** The original rule-based approach, scored on the same test set.

## Keywords

NLP · Sentiment Analysis · Text Classification · TF-IDF · Logistic Regression · Naive Bayes · SVM · GridSearchCV · VADER · Machine Learning · Data Science · Python

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (TfidfVectorizer, Pipeline, Logistic Regression, Naive Bayes, LinearSVC, GridSearchCV)
- VADER (vaderSentiment) for the baseline
- matplotlib and seaborn for charts
- Jupyter Notebook

## Project Structure

| File | Purpose |
| :--- | :--- |
| `Amazon_Reviews_Sentiment_ML.ipynb` | Main notebook: TF-IDF, model comparison, tuning, evaluation, VADER benchmark |
| `Amazon_Sentiment_Analysis.ipynb` | Earlier VADER-only baseline |
| `amazonreviews.tsv` | Dataset of 10,000 labelled Amazon reviews |
| `requirements.txt` | Python dependencies |

## Author

**Amar Chaurasiya**, Data Scientist working as an AI Engineer. Open to full-time roles and freelance work.

- Email: [amarchaurasiya124@gmail.com](mailto:amarchaurasiya124@gmail.com)
- LinkedIn: [linkedin.com/in/amar-chaurasiya-163788262](https://www.linkedin.com/in/amar-chaurasiya-163788262)
- Portfolio: [amarchaurasiya.vercel.app](https://amarchaurasiya.vercel.app)
