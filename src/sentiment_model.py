import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
data = pd.read_csv("../Data/sample_reviews.csv")

X = data["review"]
y = data["sentiment"]

# Vectorization
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Predict example
sample = ["The service was very bad"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("Prediction:", prediction[0])
