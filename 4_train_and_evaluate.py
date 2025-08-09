import os
from collections import Counter
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import utils
import config
from collections import Counter
from collections import Counter, defaultdict

# ===== Delta parameter functions =====
def p_obj_given_D(obj, raw_counter, total):
    return raw_counter.get(obj, 0) / float(total) if total > 0 else 0.0

def delta_normal(obj, raw_counter, total):
    return p_obj_given_D(obj, raw_counter, total)

def delta_avg(obj, raw_counter, total):
    p = p_obj_given_D(obj, raw_counter, total)
    return p / sum(raw_counter.values()) if sum(raw_counter.values()) > 0 else 0.0

def delta_normalized(obj, raw_counter, total):
    p = p_obj_given_D(obj, raw_counter, total)
    return p / (p ** 0.25) if p > 0 else 0.0

def delta_multi(obj, raw_counter, total):
    p = p_obj_given_D(obj, raw_counter, total)
    return p * raw_counter.get(obj, 0)

def delta_root(obj, raw_counter, total):
    p = p_obj_given_D(obj, raw_counter, total)
    return p ** 0.5

def delta_divide(obj, raw_counter, total, k=1):
    p = p_obj_given_D(obj, raw_counter, total)
    return p / (10 ** k)

delta_functions = {
    "normal": delta_normal,
    "avg": delta_avg,
    "normalized": delta_normalized,
    "multi": delta_multi,
    "root": delta_root,
    "divide": delta_divide
}

# ===== Precompute neighbor maps from pattern dictionary =====
def build_neighbor_maps(pattern_dict):
    """
    Converts a pattern dictionary Counter into forward & backward neighbor maps + all objects set.
    """
    forward = defaultdict(set)
    backward = defaultdict(set)
    all_objs = set()

    for (a, b), _ in pattern_dict.items():
        forward[a].add(b)
        backward[b].add(a)
        all_objs.add(a)
        all_objs.add(b)

    return forward, backward, all_objs

# ===== Optimized semantic extraction with Propositions 1–4 =====
def extract_semantic_objects_all_props_optimized(candidate_objects, neighbor_maps):
    """
    Optimized semantic object extraction using set lookups.
    neighbor_maps = (forward_map, backward_map, all_objects_set)
    """
    forward_map, backward_map, all_objects = neighbor_maps
    candidate_set = set(candidate_objects)
    semantic_objects = set()

    # Prop 1: direct neighbors
    for c in candidate_set:
        semantic_objects.update(forward_map.get(c, set()))
        semantic_objects.update(backward_map.get(c, set()))
    semantic_objects -= candidate_set

    # Prop 2: neighbors-of-neighbors
    neighbors_lvl1 = set(semantic_objects)
    for n in neighbors_lvl1:
        semantic_objects.update(forward_map.get(n, set()))
        semantic_objects.update(backward_map.get(n, set()))
    semantic_objects -= candidate_set

    # Prop 3: shared neighbor between candidates
    for obj in all_objects:
        connected_candidates = sum(1 for c in candidate_set if obj in forward_map.get(c, set()) or obj in backward_map.get(c, set()))
        if connected_candidates >= 2 and obj not in candidate_set:
            semantic_objects.add(obj)

    # Prop 4: closure until no new objects
    prev_size = -1
    while len(semantic_objects) != prev_size:
        prev_size = len(semantic_objects)
        new_objs = set()
        for s in semantic_objects:
            new_objs.update(forward_map.get(s, set()))
            new_objs.update(backward_map.get(s, set()))
        semantic_objects.update(new_objs - candidate_set)

    return list(semantic_objects)

# ===== Feature vector calculation =====
def calculate_feature_vector_optimized(raw_objects, dictionaries, neighbor_maps_all, delta_type="normal", divide_k=1):
    """
    raw_objects: detected objects from one image
    dictionaries: {'raw': {category: list}, 'pattern': {category: Counter}}
    neighbor_maps_all: {category: (forward_map, backward_map, all_objects_set)}
    """
    candidate_objects = [obj for obj, _ in Counter(raw_objects).most_common(5)]

    feature_vector = []
    category_names = sorted(dictionaries['raw'].keys())

    for category in category_names:
        raw_dict = dictionaries['raw'][category]
        total_objects = len(raw_dict)
        raw_counter = Counter(raw_dict)
        neighbor_maps = neighbor_maps_all[category]

        # Per-category semantic extraction
        semantic_objects = extract_semantic_objects_all_props_optimized(candidate_objects, neighbor_maps)

        category_feature = 0.0
        if total_objects > 0:
            for sobj in semantic_objects:
                p_obj = p_obj_given_D(sobj, raw_counter, total_objects)
                if delta_type == "divide":
                    delta = delta_divide(sobj, raw_counter, total_objects, k=divide_k)
                else:
                    delta = delta_functions[delta_type](sobj, raw_counter, total_objects)
                category_feature += p_obj * delta

        feature_vector.append(category_feature)

    return feature_vector


if __name__ == '__main__':
    # 1. Load all necessary data
    print("Loading data...")
    raw_data = utils.load_data(config.RAW_OBJECTS_FILE)
    dictionaries = utils.load_data(config.DICTIONARIES_FILE)

    # 2. Precompute neighbor maps for each category (optimization)
    print("Precomputing neighbor maps...")
    neighbor_maps_all = {
        category: build_neighbor_maps(pattern_dict)
        for category, pattern_dict in dictionaries['pattern'].items()
    }

    # 3. Prepare feature vectors
    print("Preparing feature vectors...")
    X, y = [], []
    for path, objects in raw_data.items():
        category = os.path.basename(os.path.dirname(path))
        feature_vec = calculate_feature_vector_optimized(
            raw_objects=objects,
            dictionaries=dictionaries,
            neighbor_maps_all=neighbor_maps_all,
            delta_type=config.DELTA_TYPE,        # e.g., "normal", "avg", "multi", ...
            divide_k=getattr(config, "DIVIDE_K", 1)
        )
        X.append(feature_vec)
        y.append(category)

    X = np.array(X)
    y = np.array(y)

    # 4. Split data & scale features
    print("Splitting data and scaling features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train SVM
    print("Training SVM classifier...")
    svm = SVC(
        kernel=config.SVM_KERNEL,
        probability=True,
        verbose=True
    )
    svm.fit(X_train_scaled, y_train)

    # 6. Evaluate
    print("\nEvaluating classifier...")
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))
