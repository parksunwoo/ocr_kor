# 딥러닝을 활용한 한글문서 OCR 연구
| [paper](document/HCLT2019_deeplearningOCR.pdf) | [poster](document/Poster_HCLT2019_deeplearningOCR.pdf)

## Introduction
한글에 대한 OCR 연구는 공식 데이터가 없고 딥러닝을 사용한 시도가 많지 않았다. 본 논문은 폰트와 사전데이터를 사용해 딥러닝 모델 학습을 위한 한글 문장 이미지 데이터를 직접 생성해보고 이를 활용해서 한글 문장의 OCR 성능을 높일 다양한 모델 조합들에 대한 실험을 진행했다. 딥러닝을 활용한 OCR 실험 결과 문자 영역을 찾는 부분보다 문자인식 부분에서 개선할 부분이 있음을 찾고 다양한 모델조합의 성능을 비교하였다. 한글은 영문과 비교해 분류해야 할 글자 수가 많아 정확도를 높이기가 어렵고 기존 한글 OCR 연구 또한 글자 단위, 단일모델로만 진행되어 실제 서비스에 적용하기에 한계를 보였다. 반면 해당 논문에서는 OCR의 범위를 문장 단위로 확장하였고 실제 문서 이미지에서 자주 발견되는 유형의 데이터를 사용해 애플리케이션 적용 가능성을 높이고자 한 부분에 의의가 있다.

## Updates
2019-08-11 훈련 및 검증 데이터셋 생성과정 설명추가  
2019-10-05 모델훈련 및 1차실험결과 추가  
2019-10-13 논문 및 포스터, 모델파일 추가  
2019-10-15 데모 스크립트 및 파일정리    
2021-08-16 requirements.txt, 폰트파일 추가 및 minor fix

## Getting started
### Generate train/ validation data

1. fonts/ko 에 폰트를 추가한다 .ttf만 가능
    - [네이버 나눔글꼴](https://hangeul.naver.com/2017/nanum)(23종) 
    - [네이버 나눔손글씨](https://clova.ai/handwriting/list.html)(109종)  
2. dicts 에 한글 단어사전을 추가한다
    - [국립국어원 한국어 학습용 어휘목록](https://www.korean.go.kr/)  
3. 원하는 유형에 맞추어 데이터를 생성한다
    - [5개 유형 데이터 생성 bash](/data/generator/TextRecognitionDataGenerator/generate_data_5type.sh)  
    - basic  
        <img src="./data/generator/TextRecognitionDataGenerator/out/basic/가스 까맣다 나빠지다 생선 출판 흐르다 비롯되다 인격 자랑스럽다 저렇게_1.jpg" width="1000" title="basic1">
        <img src="./data/generator/TextRecognitionDataGenerator/out/basic/국제화 아쉬움 넘치다 뜨다 낡다_1.jpg" width="1000" title="basic2">
        <img src="./data/generator/TextRecognitionDataGenerator/out/basic/농민 특성 지우개 철도 전설 벽 향하다 아스팔트 모두 존중하다_0.jpg" width="1000" title="basic3">
    - skew  
        <img src="./data/generator/TextRecognitionDataGenerator/out/skew/싶어지다 담배 들여다보다 외치다 달다_1.jpg" width="1000" title="skew1">
        <img src="./data/generator/TextRecognitionDataGenerator/out/skew/장모 무리하다 항상 목적 높아지다_2.jpg" width="1000" title="skew2">
        <img src="./data/generator/TextRecognitionDataGenerator/out/skew/커튼 실시 사계절 접근하다 듣다_0.jpg" width="1000" title="skew3">        
    - distortion  
        <img src="./data/generator/TextRecognitionDataGenerator/out/dist/그리 물 태권도 덜 지급_2.jpg" width="1000" title="dist1">
        <img src="./data/generator/TextRecognitionDataGenerator/out/dist/남대문 시대적 먹이 놀이 석_2.jpg" width="1000" title="dist2">
        <img src="./data/generator/TextRecognitionDataGenerator/out/dist/인사 밤중 자극 쥐 마음씨_2.jpg" width="1000" title="dist3">
    - blurring  
        <img src="./data/generator/TextRecognitionDataGenerator/out/blur/결정 안심하다 한복 재산 감상_1.jpg" width="1000" title="blur1">
        <img src="./data/generator/TextRecognitionDataGenerator/out/blur/민주주의 열정 시중 백두산 앨범_4.jpg" width="1000" title="blur2">
        <img src="./data/generator/TextRecognitionDataGenerator/out/blur/집단 원피스 현지 갈비탕 교환하다_3.jpg" width="1000" title="blur3">
    - background  
        <img src="./data/generator/TextRecognitionDataGenerator/out/back/가능 즐겁다 너머 최선 기타_1.jpg" width="1000" title="back1">
        <img src="./data/generator/TextRecognitionDataGenerator/out/back/결혼 연세 전개되다 찌다 싸움_0.jpg" width="1000" title="back2">
        <img src="./data/generator/TextRecognitionDataGenerator/out/back/곁 호주 꾸미다 너무 산부인과_0.jpg" width="1000" title="back3">

4. 생성된 데이터에 맞는 gt 파일을 생성한다
```shell
$ cd data
$ ./create_gt_file.sh basic | tee -a gt_basic.txt
```    

### Create lmdb dataset
```shell
$ python3 data/create_lmdb_dataset.py --inputPath data/generator/TextRecognitionDataGenerator/ \ 
                        --gtFile data/gt_basic.txt \
                        --outputPath data/data_lmdb_release/training;
```
  
### Train / Test 
ex) TPS-VGG-None-Attn  
```shell
$ CUDA_VISIBLE_DEVICES=0 python3 deep-text-recognition-benchmark/train.py \ 
                                          --train_data data/data_lmdb_release/training \
                                          --valid_data data/data_lmdb_release/validation \
                                          --select_data basic-skew --batch_ratio 0.5-0.5 \
                                          --Transformation TPS \
                                          --FeatureExtraction VGG \
                                          --SequenceModeling None \
                                          --Prediction Attn \
                                          --data_filtering_off \ 
                                          --batch_max_length 50 \
                                          --workers 4;

$ CUDA_VISIBLE_DEVICES=0 python3 deep-text-recognition-benchmark/test.py \
                        --eval_data data/data_lmdb_release/evaluation \ 
                        --benchmark_all_eval \
                        --Transformation TPS \
                        --FeatureExtraction VGG \ 
                        --SequenceModeling None \
                        --Prediction Attn \
                        --saved_model saved_models/TPS-VGG-None-Attn-Seed1111/best_accuracy.pth \
                        --data_filtering_off \
                        --workers 4;
```

### Demo
```shell script
$ CUDA_VISIBLE_DEVICES=0 python3 deep-text-recognition-benchmark/demo.py \
                        --Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM --Prediction Attn \
                        --image_folder data/demo_image/ \
                        --saved_model deep-text-recognition-benchmark/saved_models/TPS-VGG-BiLSTM-Attn-Seed9998/best_accuracy.pth;        
```

### Exprements results
 
| v1    | 변환      | 추출      | 시퀀스     | 예측      | 정확도%    | 시간ms    | 파라미터*10^6| 
| ----- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | 
| T1    | None     | RCNN     | None     | CTC      | 0.19     | 0.27     | 2.35     | 
| T2    | TPS      | RCNN     | None     | CTC      | 10.3     | 0.40     | 4.05     | 
| T3    | TPS      | VGG      | None     | Attn     | 25.8     | 1.04     | 9.46     |
| T4    | TPS      | ResNet   | BiLSTM   | Attn     | 29.28    | 1.69     | 50.75    |
| T5    | TPS      | VGG      | BiLSTM   | Attn     | 29.34    | 3.68     | 12.04    |

| v2    | 변환      | 추출      | 시퀀스     | 예측      | 정확도%    | 시간ms    | 파라미터*10^6| 
| ----- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | 
| T1    | None     | VGG      | None     | CTC      | 62.72    | 1.27     | 6.04     | 
| T2    | None     | VGG      | BiLSTM   | CTC      | 65.91    | 1.31     | 8.69     | 
| T3    | TPS      | VGG      | BiLSTM   | CTC      | 87.79    | 1.79     | 10.38    |
| T4    | TPS      | VGG      | BiLSTM   | Attn     | 88.24    | 2.95     | 12.04    |
| T5    | TPS      | ResNet   | BiLSTM   | Attn     | 87.43    | 6.84     | 50.75    |

### Download models (best accuracy)
[TPS-VGG-BiLSTM-Attn](https://drive.google.com/open?id=12gyTwu5tM_9MthsXI9Ja2yCOXpvDEIao)  

### Acknowledgements
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark), [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)













   
   
   


