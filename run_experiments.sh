#!/usr/bin/env bash
# --------------------------------------------------------------
# Bash script to run the full Nutrition5k experiment pipeline:
#
# 1) Train both models (ImageNet and COYO) with main.py
# 2) Generate predictions for each
# 3) Compute evaluation statistics for each set of predictions
# --------------------------------------------------------------

# 0) (Opcional) Limpar logs ou resultados antigos
# rm -rf output/*.pth output/*.csv output/*.json

echo "[STEP 1] Training both models (ImageNet + COYO)..."
python3 main.py

# Se quiser rodar em segundo plano, use:
# nohup python main.py > train.log 2>&1 &

echo "[STEP 2] Generating predictions for ImageNet-based model..."
python3 generate_predictions_ViT_imagenet.py

echo "[STEP 3] Generating predictions for COYO-based model..."
python3 generate_predictions_ViT_COYO.py

echo "[STEP 4] Computing evaluation statistics for ImageNet predictions..."
# Supondo que "output/predictions_vit_imagenet.csv" seja o CSV de predições gerado
# Ajuste os caminhos de metadata e do CSV conforme necessário
python3 generate_eval_statistics.py \
    /media/work/datasets/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv \
    /media/work/datasets/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv \
    output/predictions_vit_imagenet.csv \
    output/output_statistics_vit_imagenet.json

echo "[STEP 5] Computing evaluation statistics for COYO predictions..."
# Supondo que "output/predictions_vit_COYO.csv" seja o CSV de predições do modelo COYO
python3 generate_eval_statistics.py \
    /media/work/datasets/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv \
    /media/work/datasets/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv \
    output/predictions_vit_COYO.csv \
    output/output_statistics_vit_COYO.json

echo "[DONE] Pipeline complete! Check the 'output/' folder for logs, predictions, and stats."
