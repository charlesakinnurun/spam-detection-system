
<!--![sms-header](/assets/sms_header_2.png)-->

<!-- ![](/assets/spam_header.png) -->


![header](/assets/image_header.webp)

<h1 align="center">SMS Spam Detection System</h1>








<!-- ![sms-header](/assets/sms_header.jpg) -->



<!--This project implements an SMS spam detection system using machine learning techniques. The core logic and experiments are documented in `model.ipynb`.-->
<!--
## Overview

Engineered a machine learning workflow to detect and classify sms messages as ham or spam. The workflow includes data loading, preprocessing, feature extraction, model training, evaluation, and prediction.-->






<!--


<p>What I learned</p> -->

<!-- - How to build reproducible NLP pipelines that turn raw text into actionable insights using feature engineering (TF‑IDF), end-to-end ML workflows, model evaluation (precision, recall, F1, confusion matrices), and best practices for real-world text classification tasks.-->
<!--
- How natural language processing classifies data into ham or spam messages.

-->

<!--
- NLP can turn raw text into actionable decisions through cleaning, tokenisation, and representation.

- Feature engineering (e.g., TF‑IDF vectorisation) converts words into numerical features that capture importance.
- Learned the full ML workflow: data loading, splitting, training, persisting, and re-using models.

- Interpreting results using precision, recall, F1-score, and confusion matrices to understand and improve classifiers.

- Writing structured, reproducible code by separating concerns (data_loader, preprocessor, trainer).

- Gained skills applicable to real-world text classification tasks like filtering unwanted messages.-->

<p> What I did</p>

- Engineered a modular Python SMS classification pipeline handling data loading, cleaning, vectorising, model training, evaluation, and prediction, including label encoding and saving the best model with TF‑IDF vectoriser using joblib.

- Created visualizations (message lengths, class balance, model performance) and documented the workflow in README.md and a Jupyter notebook for reproducibility and future use.


<!--
- Engineered a modular Python pipeline for SMS data: loading, cleaning, vectorising, training, evaluating, and predicting.

- Processed the dataset (spam.csv), handled inconsistencies, and encoded labels.

- Trained and tested models (e.g., logistic regression) and saved the best classifier with TF‑IDF vectoriser and label encoder using joblib.

- Created visualisations: message length distributions, class balance, and model performance plots via visualiser.py.

- Documented workflow in README.md and an exploratory Jupyter notebook (model.ipynb) for reproducibility and future use.

-->


















<h2 align="center">Workflow</h2>

<!--
-   Import Libraries
    - pandas
    - scikit-learn
    - seaborn
    - matplotlib
    - numpy
-->

- <h4><a href="/src/data_loader.py">Data Loading</a></h4>
<!--- Data Loading-->

| v1  | v2 |
|-----|-----|
| ham | Go until jurong point, crazy.. Available only ... |
| ham | Ok lar... Joking wif u oni... |
| spam | Free entry in 2 a wkly comp to win FA Cup fina... |
| ham | U dun say so early hor... U c already then say... |
| ham | Nah I don't think he goes to usf, he lives aro... |

<a href="/data/spam.csv">Check out dataset</a>




- <h4><a href="/src/preprocessor.py">Data Preprocessing</a></h4>

    - Rename columns for clarity
    - Check for missing values
    - Check for duplicated rows
    - Initialize LabelEncoder to categorical text labels to numerical values
    


- <h4><a href="/src/preprocessor.py">Feature Engineering</a></h4>

    - Feature (X): ``df["message"] ``

    - Target (y): ``df["label_encoded"] ``


- <h4><a href="/src/vectoriser.py">Data Splitting</a></h4>

    - 80% (training), 20% (testing)

- <h4><a href="/assets/distribution.png">Pre-Training Visualization</a></h4>

    - <a href="/assets/distribution.png">Distribution of Ham and Spam Messages</a>

    ![Distribution of Ham and Spam Messages](/assets/distribution.png)

- <h4><a href="/src/vectoriser.py">Text Vectorization</a></h4>

    - Initialize the TF-IDF Vectorizer

- <h4><a href="/src/trainer.py">Model Training and Comparison</a></h4>

    - Multinominal Naive Bayes
    - Random Forest
    - Logistic Regression
    - Support Vector Machine


- <h4><a href="/evaluation/evaluation_scores.csv">Model Evaluation</a></h4>


    | Model                         | Accuracy |
    |--------------------------------|----------|
    | Multinomial Naive Bayes        | 0.9722   |
    | Logistic Regression            | 0.9704   |
    | Support Vector Machine (SVC)   | 0.9785   |
    | Random Forest                  | 0.9758   |

- <h4><a href="/src/visualiser.py">Post-Training Visualization</a></h4>

    -   <a href="/assets/evaluation.png">Classification Model Accuracy</a>

    ![Classification Model Accuracy](/assets/evaluation.png)





<!--
- <a href="/src/predictor.py">Input Prediction</a>

    ![Input Prediction](/assets/prediction_input.png) -->
































<!--
- Import Libraries

-   Data Loading

- The dataset is loaded (typically a CSV file with columns like `label` and `message`).
- Labels are mapped to binary values (e.g., "spam" = 1, "ham" = 0).

#### 2. Data Preprocessing

- Text cleaning: Lowercasing, removing punctuation, and stopwords.
- Tokenization: Splitting messages into words.
- Optional: Stemming or lemmatization to reduce words to their base forms.

#### 3. Feature Extraction

- Text data is converted into numerical features using techniques like:
    - Bag of Words (CountVectorizer)
    - TF-IDF (TfidfVectorizer)

#### 4. Model Training

- The dataset is split into training and testing sets.
- Machine learning models are trained, such as:
    - Naive Bayes (MultinomialNB)
    - Logistic Regression
    - Support Vector Machine (SVM)
- Hyperparameters may be tuned for better performance.

#### 5. Evaluation

- Models are evaluated using metrics like:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion matrix

#### 6. Prediction

- The trained model predicts whether new messages are spam or ham.


<!--
## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/sms-spam.git
cd sms-spam
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
- Environment
    - Jupyter Notebook
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```


## Project Structure
```
sms-spam/
│
├── model.ipynb  
|── model.py    
|── spam.csv  
├── requirements.txt 
├── LICENSE
├── iamge.jpg    
├── Screenshot (244).png
├── Screenshot (245).png
├── Screenshot (246).png
├── Screenshot (247).png
├── Screenshot (248).png  
└── README.md          
```-->
























































