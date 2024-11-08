from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

# File paths
filedir = os.getcwd()
embeddings = os.path.join(filedir, "output", "embeddings.pickle")
out_model = os.path.join(filedir, "output", "recognizer.pickle")
out_label_encoder = os.path.join(filedir, "output", "label_encoder.pickle")

# Load embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open(embeddings, "rb").read())

# Encode labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# Train the model
print("[INFO] training model...")
recognizer = SVC(C=2, kernel="rbf", probability=True)
recognizer.fit(data["embeddings"], labels)

# Save the model and label encoder
with open(out_model, "wb") as f:
    f.write(pickle.dumps(recognizer))
with open(out_label_encoder, "wb") as f:
    f.write(pickle.dumps(le))
