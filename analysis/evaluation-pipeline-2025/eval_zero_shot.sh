#!/bin/bash

MODEL_PATH=$1
BACKEND=$2
EVAL_DIR=${3:-"evaluation_data/full_eval"}

python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/blimp_filtered" --save_predictions
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/supplement_filtered" --save_predictions
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task ewok --data_path "${EVAL_DIR}/ewok_filtered" --save_predictions
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task entity_tracking --data_path "${EVAL_DIR}/entity_tracking" --save_predictions
python -m evaluation_pipeline.sentence_zero_shot.run --model_path_or_name $MODEL_PATH --backend $BACKEND --task wug --data_path "${EVAL_DIR}/wug_adj_nominalization" --save_predictions
python -m evaluation_pipeline.reading.run --model_path_or_name $MODEL_PATH --backend $BACKEND --data_path "${EVAL_DIR}/reading/reading_data.csv"