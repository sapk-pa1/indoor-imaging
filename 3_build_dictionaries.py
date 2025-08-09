import os
import time
from collections import Counter, defaultdict
import utils
import config

def build_all_dictionaries(raw_objects_data):
    """Builds raw and pattern dictionaries for all categories."""
    
    # 1. Group objects by category
    category_objects = defaultdict(list)
    for path, objects in raw_objects_data.items():
        # Assumes path format like '.../category/image.jpg'
        category = os.path.basename(os.path.dirname(path))
        category_objects[category].extend(objects)

    # 2. Build dictionaries for each category
    raw_dictionaries = {}
    pattern_dictionaries = {}

    for category, objects in category_objects.items():
        # Raw dictionary is the list of all objects
        raw_dictionaries[category] = objects
        
        # Pattern dictionary counts co-occurring pairs
        object_pairs = [
            tuple((objects[i], objects[i+1])) # Preserving the discovery stage 
            # Proposition rely on the directed adjacency 
            for i in range(len(objects) - 1)
        ]
        pattern_dictionaries[category] = Counter(object_pairs)
        
    return {'raw': raw_dictionaries, 'pattern': pattern_dictionaries}

if __name__ == '__main__':
    print("Loading raw object data...")
    raw_data = utils.load_data(config.RAW_OBJECTS_FILE)
    
    print("Building dictionaries...")
    start_time = time.time()
    
    all_dicts = build_all_dictionaries(raw_data)
    
    utils.save_data(all_dicts, config.DICTIONARIES_FILE)
    
    end_time = time.time()
    print(f"Finished building dictionaries in {end_time - start_time:.2f} seconds.")
    print(f"Found {len(all_dicts['raw'])} categories.")