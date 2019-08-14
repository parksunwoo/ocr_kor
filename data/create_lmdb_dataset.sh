#!/usr/bin/env bash
# train_basic 9000
python3 create_lmdb_dataset.py --inputPath ocr_kor/data/generator/TextRecognitionDataGenerator --gtFile data/gt_basic.txt --outputPath ocr_kor/data/data_lmdb_release/training/basic/;

# train_skew 9000
python3 create_lmdb_dataset.py --inputPath ocr_kor/data/generator/TextRecognitionDataGenerator --gtFile data/gt_skew.txt --outputPath ocr_kor/data/data_lmdb_release/training/skew/;

# val_distortion 3000
python3 create_lmdb_dataset.py --inputPath ocr_kor/data/generator/TextRecognitionDataGenerator --gtFile data/gt_dist.txt --outputPath ocr_kor/data/data_lmdb_release/validation/dist/;

# val_blur 3000
python3 create_lmdb_dataset.py --inputPath ocr_kor/data/generator/TextRecognitionDataGenerator --gtFile data/gt_blur.txt --outputPath ocr_kor/data/data_lmdb_release/validation/blur/;

# val_background 3000
python3 create_lmdb_dataset.py --inputPath ocr_kor/data/generator/TextRecognitionDataGenerator --gtFile data/gt_back.txt --outputPath ocr_kor/data/data_lmdb_release/validation/back/;