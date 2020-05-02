@echo off
call venv\Scripts\activate.bat
set QDMR_DIR=C:\Users\Osher\Desktop\oren\break-processed-data

set OUTPUT_DIR=C:\Users\Osher\Desktop\oren\oren-qdmr-output

set BERT_BASE_DIR=C:\Users\Osher\Desktop\oren\cased_L-12_H-768_A-12



::# If you train multiple models on the same data, change this label.
set EXPERIMENT=qdmr_experiment
::# To quickly test that model training works, set the number of epochs to a
::# smaller value (e.g. 0.01).
::set NUM_EPOCHS=3.0
set NUM_EPOCHS=2.25
::set BATCH_SIZE=64
set BATCH_SIZE=32
set PHRASE_VOCAB_SIZE=500
set MAX_INPUT_EXAMPLES=1000000
set SAVE_CHECKPOINT_STEPS=500

:: ###########################

::goto NEXT_1
goto NEXT_p1
::goto NEXT_p2
::goto NEXT_3_1
::goto NEXT_4
::goto NEXT_5
::goto NEXT_6

:: 1. Phrase Vocabulary Optimization

:NEXT_1
python phrase_vocabulary_optimization.py^
    --input_file=%QDMR_DIR%\train1.tsv^
    --input_format=wikisplit^
    --vocabulary_size=%PHRASE_VOCAB_SIZE%^
    --max_input_examples=%MAX_INPUT_EXAMPLES%^
    --output_file=%OUTPUT_DIR%\label_map.txt
goto END
::### 2. Converting Target Texts to Tags

:NEXT_p1
python preprocess_main.py^
    --input_file=%QDMR_DIR%\tune1.tsv^
    --input_format=wikisplit^
    --output_tfrecord=%OUTPUT_DIR%\tune1.tf_record^
    --label_map_file=%OUTPUT_DIR%\label_map.txt^
    --vocab_file=%BERT_BASE_DIR%\vocab.txt^
    --output_arbitrary_targets_for_infeasible_examples=true
goto END

:NEXT_p2
python preprocess_main.py^
    --input_file=%QDMR_DIR%\train1.tsv^
    --input_format=wikisplit^
    --output_tfrecord=%OUTPUT_DIR%\train1.tf_record^
    --label_map_file=%OUTPUT_DIR%\label_map.txt^
    --vocab_file=%BERT_BASE_DIR%\vocab.txt^
    --output_arbitrary_targets_for_infeasible_examples=false
goto END

::### 3. Model Training

:NEXT_3_1
::NUM_TRAIN_EXAMPLES=$(cat "${OUTPUT_DIR}/train.tf_record.num_examples.txt")
::NUM_EVAL_EXAMPLES=$(cat "${OUTPUT_DIR}/tune.tf_record.num_examples.txt")
set NUM_TRAIN_EXAMPLES=40317
set NUM_EVAL_EXAMPLES=2123
set CONFIG_FILE=C:\Users\Osher\Desktop\oren\lasertagger-master\configs\lasertagger_config.json

python run_lasertagger.py^
  --training_file=%OUTPUT_DIR%\train1.tf_record^
  --eval_file=%OUTPUT_DIR%\tune1.tf_record^
  --label_map_file=%OUTPUT_DIR%\label_map.txt^
  --model_config_file=%CONFIG_FILE%^
  --output_dir=%OUTPUT_DIR%\models\%EXPERIMENT%^
  --init_checkpoint=%BERT_BASE_DIR%\bert_model.ckpt^
  --do_train=true^
  --do_eval=true^
  --train_batch_size=%BATCH_SIZE%^
  --save_checkpoints_steps=%SAVE_CHECKPOINT_STEPS%^
  --num_train_epochs=%NUM_EPOCHS%^
  --num_train_examples=%NUM_TRAIN_EXAMPLES%^
  --num_eval_examples=%NUM_EVAL_EXAMPLES%

goto END

::### 4. Prediction

::# Export the model.
:NEXT_4

python run_lasertagger.py^
  --label_map_file=%OUTPUT_DIR%\label_map.txt^
  --model_config_file=%CONFIG_FILE%^
  --output_dir=%OUTPUT_DIR%\models\%EXPERIMENT%^
  --do_export=true^
  --export_path=%OUTPUT_DIR%\models\%EXPERIMENT%\export
goto END

:NEXT_5

::# Get the most recently exported model directory.
::TIMESTAMP=$(ls "${OUTPUT_DIR}/models/${EXPERIMENT}/export/" | \
::            grep -v "temp-" | sort -r | head -1)
set SAVED_MODEL_DIR=%OUTPUT_DIR%\models\%EXPERIMENT%\export\1588256843
set PREDICTION_FILE=%OUTPUT_DIR%\models\%EXPERIMENT%\pred.tsv

python predict_main.py^
  --input_file=%QDMR_DIR%\dev.tsv^
  --input_format=wikisplit^
  --output_file=%PREDICTION_FILE%^
  --label_map_file=%OUTPUT_DIR%\label_map.txt^
  --vocab_file=%BERT_BASE_DIR%\vocab.txt^
  --saved_model=%SAVED_MODEL_DIR%
goto END

::### 5. Evaluation
:NEXT_6
python score_main.py --prediction_file=%PREDICTION_FILE%

:END