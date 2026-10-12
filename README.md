# Indonesian News Classification

Multi-class classification of Indonesian news articles into 9 categories using TF-IDF features and three classifiers: Multinomial Naive Bayes, Decision Tree, and AdaBoost. Models are compared with stratified 5-fold cross-validation on accuracy, precision, recall, F1-score, and training time.

## Highlights

- Built a full text-classification workflow for Indonesian: cleaning, stemming and stop-word removal with Sastrawi, TF-IDF, model benchmarking, error analysis, and inference on new text.
- **Naive Bayes** was the best model: **77.5% accuracy** and **75.1% weighted F1**, and it trained about **22x faster** than AdaBoost (2.7 s vs. 60.5 s per fold).
- Error analysis shows that smaller categories (Regional, Global, Health) are often misclassified as the broad "News" category.

## Dataset

| Item | Detail |
|---|---|
| Source | Liputan6 news articles (`liputan6.com`) |
| Raw size | 4,581 articles, 8 columns (`title`, `source`, `date`, `time`, `category`, `sub-category`, `content`, `url`) |
| Average length | about 308 words per article |
| After cleaning | 4,491 articles, 9 categories |

**Cleaning steps**
1. Dropped rows with missing or empty `title`, `content`, or `category` (4,581 to 4,574 rows) and removed duplicates (none found).
2. Removed categories with fewer than 100 samples (`Lifestyle`: 82, `Citizen6`: 1), leaving 4,491 articles.

**Classes (9):** Bisnis, Bola, Global, Health, News, Otomotif, Regional, ShowBiz, Tekno

| Class | Articles | Share |
|---|---|---|
| News | 1,183 | 26.3% |
| Bola | 1,075 | 23.9% |
| Bisnis | 553 | 12.3% |
| ShowBiz | 430 | 9.6% |
| Regional | 318 | 7.1% |
| Global | 297 | 6.6% |
| Tekno | 240 | 5.3% |
| Health | 209 | 4.7% |
| Otomotif | 186 | 4.1% |

> The dataset is not included in this repository. Place your copy of the CSV in the project folder and update the file path in the notebook.

## Methodology

**1. Text preprocessing** (title and content are concatenated)
- Lowercasing
- Removal of URLs, numbers, punctuation, and non-alphabet characters
- Stop-word removal and stemming with [PySastrawi](https://github.com/har07/PySastrawi)

**2. Feature extraction**
- `TfidfVectorizer`: `max_features=10000`, `ngram_range=(1, 2)`, `min_df=2`, `sublinear_tf=True`
- Fitted inside each cross-validation fold through a scikit-learn `Pipeline`, so no information leaks from the test fold.

**3. Models**

| Model | Main settings |
|---|---|
| Multinomial Naive Bayes | `alpha=1.0` |
| Decision Tree | `criterion="entropy"`, `min_samples_split=5`, `min_samples_leaf=2` |
| AdaBoost | 100 estimators, `SAMME`, base tree `max_depth=3`, entropy criterion |

**4. Evaluation**
- Stratified 5-fold cross-validation (`random_state=42`)
- Weighted accuracy, precision, recall, and F1-score, plus training time per fold
- Aggregated confusion matrices, per-class TP/FP/FN/TN, and most frequent misclassifications

## Results (5-fold cross-validation, mean ± std)

| Model | Accuracy | Precision | Recall | F1-Score | Train time / fold |
|---|---|---|---|---|---|
| **Naive Bayes** | **0.7751** ± 0.0129 | **0.8218** | **0.7751** | **0.7508** ± 0.0145 | **2.7 s** |
| AdaBoost | 0.7288 ± 0.0199 | 0.7511 | 0.7288 | 0.7269 ± 0.0195 | 60.5 s |
| Decision Tree | 0.6843 ± 0.0121 | 0.6831 | 0.6843 | 0.6816 ± 0.0129 | 5.4 s |

### Per-class performance (Naive Bayes, aggregated over 5 folds)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Bola | 0.970 | 0.957 | 0.964 |
| ShowBiz | 0.888 | 0.847 | 0.867 |
| Bisnis | 0.733 | 0.895 | 0.806 |
| News | 0.613 | 0.954 | 0.747 |
| Tekno | 0.986 | 0.592 | 0.740 |
| Global | 0.900 | 0.455 | 0.604 |
| Otomotif | 0.987 | 0.414 | 0.583 |
| Health | 0.872 | 0.325 | 0.474 |
| Regional | 0.778 | 0.132 | 0.226 |

### Key observations

- **Strong categories:** `Bola` (sports) and `ShowBiz` have distinctive vocabulary and are classified reliably.
- **"News" acts as a catch-all.** Naive Bayes predicts `News` for many articles from other classes. The most common errors are `Regional` to `News` (244), `Global` to `News` (119), and `Health` to `News` (91).
- **Weighted vs. macro metrics.** Weighted scores look healthy, but recall for `Regional` (13%) and `Health` (33%) is low. The model favors the larger classes.
- **Speed vs. accuracy.** AdaBoost improves on a single Decision Tree but costs roughly 11x more training time, and it still does not beat Naive Bayes on this feature set.

## Inference demo

The notebook includes a `predict_news(text)` function that applies the same preprocessing and returns the predicted category, a confidence score, and the top-3 class probabilities. On 6 hand-written example articles, the model classified 5 correctly. The one miss was a health article predicted as `News`, which matches the weakness described above. This is a small qualitative demo, not a performance estimate.

## Limitations and possible next steps

- Class imbalance and overlapping categories limit performance on smaller classes. Possible improvements: class weighting, `ComplementNB`, linear models (Logistic Regression, Linear SVM), or resampling.
- Hyperparameters were set manually and not tuned; a grid or random search could be added.
- Transformer-based Indonesian models (e.g., IndoBERT) could be compared against the TF-IDF baseline.
- Sub-category labels (`sub-category`) are available in the data but were not used.

## Tech stack

Python, Pandas, NumPy, scikit-learn, PySastrawi, Matplotlib, Seaborn, Jupyter / Google Colab

## How to run

```bash
git clone https://github.com/davinapsh04/indonesian-news-classification.git
cd indonesian-news-classification
pip install pandas numpy matplotlib seaborn scikit-learn PySastrawi jupyter
jupyter notebook
```

Open the notebook, set the dataset path, and run all cells. Stemming with Sastrawi is the slowest step.

## Repository structure

```
.
├── indonesian-news-classification.ipynb   # full workflow with outputs
└── README.md
```

Davina Posh, [GitHub](https://github.com/davinapsh04)
