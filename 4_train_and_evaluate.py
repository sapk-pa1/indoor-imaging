import os
from collections import Counter
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import utils
import config

def extract_semantic_objects(candidate_objects, pattern_dict):
    """Extracts semantically related objects."""
    semantic_objects = set()
    candidate_set = set(candidate_objects)
    
    for candidate in candidate_set:
        for pair, _ in pattern_dict.items():
            if candidate in pair:
                other_object = pair[0] if pair[1] == candidate else pair[1]
                if other_object not in candidate_set:
                    semantic_objects.add(other_object)
    return list(semantic_objects)

def calculate_feature_vector(raw_objects, dictionaries):
    """Calculates the final feature vector for a single image."""
    
    # 1. Select candidate objects (most frequent ones)
    candidate_objects = [obj for obj, _ in Counter(raw_objects).most_common(5)]

    # 2. Extract semantic objects by checking against each category's pattern dict
    all_semantic_objects = set()
    for pattern_dict in dictionaries['pattern'].values():
        all_semantic_objects.update(
            extract_semantic_objects(candidate_objects, pattern_dict)
        )
    
    # 3. Calculate probability and delta parameter for the feature vector
    feature_vector = []
    category_names = sorted(dictionaries['raw'].keys())

    for category in category_names:
        raw_dict = dictionaries['raw'][category]
        total_objects = len(raw_dict)
        raw_dict_counts = Counter(raw_dict)
        
        # Create a sub-vector for the current category
        category_feature = 0.0
        if total_objects > 0:
            for sobj in all_semantic_objects:
                # [cite_start]Normal Delta Parameter as per the paper [cite: 240, 283]
                p_obj = raw_dict_counts[sobj] / total_objects
                delta = p_obj
                category_feature += p_obj * delta
        
        feature_vector.append(category_feature)
        
    return feature_vector

if __name__ == '__main__':
    # 1. Load all necessary data
    print("Loading data...")
    raw_data = utils.load_data(config.RAW_OBJECTS_FILE)
    dictionaries = utils.load_data(config.DICTIONARIES_FILE)

    # 2. Prepare data for scikit-learn
    print("Preparing feature vectors...")
    X = []
    y = []
    
    for path, objects in raw_data.items():
        category = os.path.basename(os.path.dirname(path))
        feature_vec = calculate_feature_vector(objects, dictionaries)
        X.append(feature_vec)
        y.append(category)

    X = np.array(X)
    y = np.array(y)

    # 3. Split data and scale features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train SVM and evaluate
    print("Training SVM classifier...")
    svm = SVC(kernel=config.SVM_KERNEL, probability=True)
    svm.fit(X_train_scaled, y_train)
    
    print("\nEvaluating classifier...")
    y_pred = svm.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))