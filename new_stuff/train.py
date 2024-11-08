import os
import pandas as pd
import piexif
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def extract_metadata_features(file_path):
    try:
        img = Image.open(file_path)
        exif_data = img._getexif()
        metadata = {}
        
        if exif_data is not None:
            for tag, value in exif_data.items():
                metadata[tag] = value
        else:
            metadata = {"No_Metadata": True}
            
    except Exception as e:
        print(f"Error extracting metadata for {file_path}: {e}")
        metadata = {"Error": True}
        
    return metadata


def load_metadata_dataset(base_dir):
    data = []
    labels = []
    for label, folder in [("authentic", "au"), ("tampered", "tp")]:
        folder_path = os.path.join(base_dir, folder)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            metadata_features = extract_metadata_features(file_path)
            
            # Add label and filename for reference
            metadata_features["label"] = 1 if label == "tampered" else 0
            metadata_features["filename"] = file_name
            data.append(metadata_features)
    
    return pd.DataFrame(data)

def train_and_evaluate(base_dir):
    metadata_df = load_metadata_dataset(base_dir)
    metadata_df = metadata_df.dropna(axis=1, how='any')
    X = metadata_df.drop(["label", "filename"], axis=1, errors='ignore')
    y = metadata_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

base_dir = "/path/to/CASIA2"
train_and_evaluate(base_dir)
