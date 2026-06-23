import os
import json
import csv

def build_dataset(json_dir, output_csv):
    rows = []
    
    # iterate over all .json files in the directory
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # each file has "intents" list
                for intent in data.get("intents", []):
                    tag = intent.get("tag", "")
                    patterns = intent.get("patterns", [])
                    
                    # add each pattern with its tag
                    for p in patterns:
                        rows.append([p, tag])
    
    # write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patterns", "tag"])  # header
        writer.writerows(rows)

# Example usage:
# put your folder path here
json_folder = "intents"
output_file = "intentDataset.csv"
build_dataset(json_folder, output_file)
print(f"Dataset saved to {output_file}")
