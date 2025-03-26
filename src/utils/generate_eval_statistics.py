#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script: compute_eval_statistics.py
----------------------------------
Compute regression error statistics (MAE and %MAE) for Nutrition5k predictions.

This script loads two ground-truth CSV files (e.g., from cafe1 and cafe2),
merges them, then compares them to a CSV of predictions. It calculates both
mean absolute error (MAE) and relative MAE (%) for each nutritional field:
- total_calories
- total_mass
- total_fat
- total_carb
- total_protein

Inputs:
    1) groundtruth_csv_path1 (str)
    2) groundtruth_csv_path2 (str)
    3) predictions_csv_path  (str)
    4) output_json_path      (str)

The 'predictions_csv_path' should contain lines of the form:
    dish_id, total_calories, total_mass, total_fat, total_carb, total_protein

The 'groundtruth_csv_path1' and 'groundtruth_csv_path2' have the same columns.
We will read them, parse them into a dictionary dish_id -> [cal, mass, fat, carb, protein],
and compute errors only for dish_ids present in both the groundtruth and the predictions.

Output:
    A JSON file with keys like:
        "total_calories_MAE", "total_calories_MAE_%"
        "total_mass_MAE", "total_mass_MAE_%"
        "total_fat_MAE", "total_fat_MAE_%"
        "total_carb_MAE", "total_carb_MAE_%"
        "total_protein_MAE", "total_protein_MAE_%"

Example (Command-line):
    python compute_eval_statistics.py \
        /path/to/cafe1.csv \
        /path/to/cafe2.csv \
        /path/to/predictions.csv \
        /path/to/output.json
"""

import sys
import json
import statistics  # For Python <3.4, install via 'pip install statistics'
import pandas as pd
import os

# CSV fields in groundtruth/predictions
DATA_FIELDNAMES = [
    'dish_id',
    'total_calories',
    'total_mass',
    'total_fat',
    'total_carb',
    'total_protein'
]

def process_dish_metadata(csv_path):
    """
    Reads a CSV for Nutrition5k dish metadata or predictions (6 columns):
      dish_id, total_calories, total_mass, total_fat, total_carb, total_protein

    Returns a dict: { dish_id (str) : [cal, mass, fat, carb, protein] }
    """
    data_dict = {}
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        fields = line.strip().split(',')
        if len(fields) < 6:
            # Malformed line
            continue

        dish_id = fields[0]
        try:
            total_calories = float(fields[1])
            total_mass     = float(fields[2])
            total_fat      = float(fields[3])
            total_carb     = float(fields[4])
            total_protein  = float(fields[5])
        except ValueError:
            # Skip lines that can't parse as float
            continue

        data_dict[dish_id] = [
            total_calories,
            total_mass,
            total_fat,
            total_carb,
            total_protein
        ]

    return data_dict

def compute_eval_statistics(groundtruth_csv_path1,
                            groundtruth_csv_path2,
                            predictions_csv_path,
                            output_json_path):
    """
    Compute and save evaluation statistics (MAE, %MAE) for Nutrition5k.

    Args:
        groundtruth_csv_path1 (str): CSV path for cafe1 metadata
        groundtruth_csv_path2 (str): CSV path for cafe2 metadata
        predictions_csv_path  (str): CSV path for model predictions
        output_json_path      (str): JSON path for output stats

    Returns:
        dict: A dictionary with the computed stats, e.g.:
          {
            "total_calories_MAE": float,
            "total_calories_MAE_%": float,
            ...,
          }
    """
    # 1) Load groundtruth
    groundtruth_data1 = process_dish_metadata(groundtruth_csv_path1)
    groundtruth_data2 = process_dish_metadata(groundtruth_csv_path2)

    # Merge dicts with .update() for Python <3.5 compatibility
    groundtruth_data = {}
    groundtruth_data.update(groundtruth_data1)
    groundtruth_data.update(groundtruth_data2)

    # 2) Load predictions
    prediction_data = process_dish_metadata(predictions_csv_path)

    # 3) Prepare structures
    groundtruth_values = {}
    abs_errors = {}
    for field in DATA_FIELDNAMES[1:]:
        groundtruth_values[field] = []
        abs_errors[field] = []

    # 4) Compute errors for each dish_id
    for dish_id, pred_vals in prediction_data.items():
        if dish_id not in groundtruth_data:
            continue
        gt_vals = groundtruth_data[dish_id]

        # Need 5 numeric fields
        if len(pred_vals) < 5 or len(gt_vals) < 5:
            continue

        # Indices: 0->cal, 1->mass, 2->fat, 3->carb, 4->protein
        for i in range(5):
            field_name = DATA_FIELDNAMES[i+1]
            gt_val   = gt_vals[i]
            pred_val = pred_vals[i]
            error_val = abs(pred_val - gt_val)

            groundtruth_values[field_name].append(gt_val)
            abs_errors[field_name].append(error_val)

    # 5) Calculate MAE and %MAE
    output_stats = {}
    for field in DATA_FIELDNAMES[1:]:
        if len(groundtruth_values[field]) == 0:
            # No samples for that field
            output_stats["{}_MAE".format(field)]   = None
            output_stats["{}_MAE_%".format(field)] = None
            continue

        mean_abs_err = statistics.mean(abs_errors[field])
        mean_gt_val  = statistics.mean(groundtruth_values[field])
        if mean_gt_val == 0:
            mae_percent = None
        else:
            mae_percent = 100.0 * mean_abs_err / mean_gt_val

        output_stats["{}_MAE".format(field)]   = mean_abs_err
        output_stats["{}_MAE_%".format(field)] = mae_percent

    # 6) Write results to JSON
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_json_path, "w") as f_out:
        json.dump(output_stats, f_out, indent=2)

    # Print summary
    print("[INFO] Evaluation statistics saved to: {}".format(output_json_path))
    for k, v in output_stats.items():
        print("  {} : {}".format(k, v))

    return output_stats

def main():
    # If running from command line, parse args
    if len(sys.argv) != 5:
        print("Usage: python compute_eval_statistics.py <gt_csv1> <gt_csv2> <pred_csv> <output_json>")
        sys.exit(1)

    groundtruth_csv_path1 = sys.argv[1]
    groundtruth_csv_path2 = sys.argv[2]
    predictions_csv_path  = sys.argv[3]
    output_json_path      = sys.argv[4]

    compute_eval_statistics(groundtruth_csv_path1,
                            groundtruth_csv_path2,
                            predictions_csv_path,
                            output_json_path)

if __name__ == "__main__":
    main()
