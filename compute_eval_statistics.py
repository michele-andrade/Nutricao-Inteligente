r"""Script to compute statistics on nutrition predictions.

This script takes in a csv of nutrition predictions and computes absolute and
percentage mean average error values comparable to the metrics used to eval
models in the Nutrition5k paper. The input csv file of nutrition predictions
should be in the form of:
dish_id, calories, mass, carbs, protein
And the groundtruth values will be pulled from the metadata csv file provided
in the Nutrition5k dataset release where the first 5 fields are also:
dish_id, calories, mass, carbs, protein

Example Usage:

python3 compute_eval_statistics.py /media/work/datasets/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv /media/work/datasets/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv output/predictions_inception.csv output/output_statistics.json
"""
import json
import os
import statistics
import sys
import pandas as pd

import os
import pandas as pd

DATA_FIELDNAMES = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']

def process_dish_metadata(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines[1:]:
        fields = line.split(',')
        dish_id = fields[0]
        total_calories = float(fields[1])
        total_mass = float(fields[2])
        total_fat = float(fields[3])
        total_carb = float(fields[4])
        total_protein = float(fields[5])
        data.append([dish_id, total_calories, total_mass, total_fat, total_carb, total_protein])

    return {item[0]: item[1:] for item in data}

if len(sys.argv) != 5:
    raise Exception("Invalid number of arguments\n\n%s" % __doc__)

groundtruth_csv_path1 = sys.argv[1]
groundtruth_csv_path2 = sys.argv[2]
predictions_csv_path = sys.argv[3]
output_path = sys.argv[4]

groundtruth_data1 = process_dish_metadata(groundtruth_csv_path1)
groundtruth_data2 = process_dish_metadata(groundtruth_csv_path2)
groundtruth_data = {**groundtruth_data1, **groundtruth_data2}
prediction_data = process_dish_metadata(predictions_csv_path)

groundtruth_values = {}
err_values = {}
output_stats = {}

for field in DATA_FIELDNAMES[1:]:
    groundtruth_values[field] = []
    err_values[field] = []

for dish_id in prediction_data:
    if dish_id in groundtruth_data:
        for i in range(len(DATA_FIELDNAMES[1:])):
            groundtruth_values[DATA_FIELDNAMES[i+1]].append(
                groundtruth_data[dish_id][i])
            err_values[DATA_FIELDNAMES[i+1]].append(abs(
                float(prediction_data[dish_id][i])
                - groundtruth_data[dish_id][i]))

for field in DATA_FIELDNAMES[1:]:
    output_stats[field + "_MAE"] = statistics.mean(err_values[field])
    output_stats[field + "_MAE_%"] = (100 * statistics.mean(err_values[field]) /
                                    statistics.mean(groundtruth_values[field]))

with open(output_path, "w") as f_out:
    f_out.write(json.dumps(output_stats))




