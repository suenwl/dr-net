import os
import json

DIR = "/Users/suenwailun/Desktop/"
os.chdir(DIR)
for filename in os.listdir():
    if filename.endswith(".json"):
        print("processing", filename)
        new_categories = []

        with open(DIR + "/" + filename, "r") as save_file:
            categories = json.load(save_file)
            for category in categories:
                if category["category"] == "Amount (including GST)":
                    category["category"] = "Total amount"
                    new_categories.append(category)
                elif category["category"] != "Amount (excluding GST)":
                    new_categories.append(category)

        with open(DIR + "/" + filename, "w") as save_file:
            json.dump(new_categories, save_file)
