import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("spam.csv")

data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data.message)

y = data.label_num

model = MultinomialNB()
model.fit(X, y)

msg = input("Enter email message: ")
msg_vec = vectorizer.transform([msg])

prediction = model.predict(msg_vec)

if prediction[0] == 1:
    print("🚨 Spam Email Detected")
else:
    print("✅ Not a Spam Email")
