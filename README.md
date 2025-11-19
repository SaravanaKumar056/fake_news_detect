# AI Fake News Detection System (Production-Ready API)

> A robust Machine Learning pipeline and REST API designed to classify news articles as "Real" or "Fake" in real-time using Natural Language Processing (NLP).

---

##  About The Project
Misinformation spreads 6x faster than truth. This project is an engineering solution to that problem. 
Unlike standard data science notebooks, this project is architected as a **deployable software application**. It decouples the model training pipeline from the inference engine, allowing for scalable, real-time predictions via a Flask API.

###  Key Features
* **Real-Time Inference:** Exposed a lightweight REST API using Flask to serve predictions instantly.
* **Modular Architecture:** Refactored monolithic Jupyter Notebooks into production-ready Python scripts (`train.py` for training, `app.py` for the server).
* **Advanced NLP:** Implemented a custom text preprocessing pipeline (Regex cleaning, Stopword removal) and TF-IDF vectorization.
* **High Accuracy:** Achieved **~99% accuracy** on the testing dataset using Logistic Regression.
* **Model Persistence:** Integrated `joblib`/`pickle` for serializing the model and vectorizer, eliminating the need to retrain for every request.

---

##  Tech Stack
* **Language:** Python 3.x
* **Backend:** Flask (Web Framework)
* **Machine Learning:** Scikit-Learn (Logistic Regression, Decision Tree)
* **Data Processing:** Pandas, NumPy, Regular Expressions (Re)
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
* **Version Control:** Git

---

## System Architecture
The project follows a clear separation of concerns:

1.  **`train.py` (The Pipeline):** * Ingests raw data (`Fake.csv`, `True.csv`).
    * Cleans and normalizes text using the `wordopt` function.
    * Trains the Logistic Regression model.
    * Serializes (saves) the trained model (`model.pkl`) and vectorizer (`vectorizer.pkl`) to disk.

2.  **`app.py` (The Application):** * Initializes a Flask web server.
    * Loads the pre-trained "brain" (`.pkl` files) into memory.
    * Exposes a `/predict` endpoint to handle user inputs and return classification results.

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/SaravanaKumar056/fake_news_detect.git](https://github.com/SaravanaKumar056/fake_news_detect.git)
cd fake_news_detect


2. Install Dependencies
Bash

pip install -r requirements.txt
3. Train the Model (Optional)
Note: The repo comes with a pre-trained model, but you can retrain it using the dataset.

Bash

python train.py
# Output: "Success! You can now run app.py."
4. Start the Application
Bash

python app.py
# Output: * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
5. Verify
Open your browser and go to http://127.0.0.1:5000. Enter a news headline to see the prediction!

 Performance Metrics
During the training phase, the model was evaluated on a split test set:

Logistic Regression: 99% Accuracy

Decision Tree: 99% Accuracy

Gradient Boosting: 99% Accuracy

Random Forest: 99% Accuracy

The production app currently uses Logistic Regression for its balance of speed and accuracy.

Future Improvements
Dockerization: Containerize the application for cloud deployment (AWS/Azure).

Deep Learning: Implement an LSTM or BERT-based model for better context understanding.

CI/CD: Set up GitHub Actions for automated testing and deployment.
