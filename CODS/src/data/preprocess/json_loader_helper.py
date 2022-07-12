import json

def load_json(json_path):
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)
    
    return json_data


def write_json(json_path, data):
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)