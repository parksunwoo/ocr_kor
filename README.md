# ocr_kor
딥러닝을 활용한 한글문서 OCR 연구

## 1.Introduction
광학 문자 인식(Optical character recognition; OCR)은 사람이 쓰거나 기계로 인쇄한 문자의 영상을 이미지 스캐너로 획득하여 기계가 읽을 수 있는 문자로 변환하는 것이다
이미지 스캔으로 얻을 수 있는 문서의 활자 영상을 컴퓨터가 편집 가능한 문자코드 등의 형식으로 변환하는 소프트웨어로써 일반적으로 OCR이라고 하며, OCR은 인공지능이나 기계 시각(machine vision)의 연구분야로 시작되었다.
OCR 기술은 크게 text detection(이미지 상에서 글자가 있는 영역을 찾는 걸 말함) 과 text recognition(찾은 영역을 바탕으로 글자를 분별해 내는걸 말함) 으로 구성되어 있고 OCR 기술이 어려운 점은 크게 3 가지이다. 
첫째로, 문서이미지 속 글자들은 정형화되어 있는 반면 보통 이미지 글자들은 비정형화 되어 있어 분별이 어렵다. 둘째로, 간판이나 벽돌 유리처럼 배경이 복잡해서 구분이 힘들다. 그리고 마지막으로 다양한 간섭요소, 예를 들면 노이즈, 왜곡, 글자들의 밀도, 저해상도, 로 인해 어려움이 있다. 
본 논문에서는 한글데이터셋을 구축, text recognition부분에 한글에 적합한 여러 기술들을 비교하여 한글문서의 OCR 성능을 높이기 위한 연구를 진행하였다.

## 2.Recent Advances in Scene Text Detection and Recognition
### 2.1 text detection
기존의 scene text detection은 글자/단어 후보생성-> 후보 필터링 -> 그룹화와 같은 여러 단계로 나누어져 있어서, 모델 학습 중의 튜닝이 매우 어려웠고, 완성된 모델의 속도도 매우 느려서 실시간 감지에 적용하기 힘들었다. Textboxes는 물체인식에서 큰 성능 향상을 보여준 SSD 논문을 본따 만들었으며, 단일 네트워크로 구성되어 있기 때문에 기존의 모델들에 비해 현저히 빠른 성능을 보여주고, 정확도 또한 기존 모델들보다 크게 향상되었다. Textboxes는 컨볼루션과  풀링으로만 이루어진 완전 컨볼루션 네트워크이다. 대부분의 경우에 잘 작동하나 과다노출, 큰 문자간격에서는 취약하다. Textboxes는 문자와 비문자 그리고 bounding 박스에 대한 regression based 방식이다. 문자와 비문자의 구별은 때로 각 이미지에 매우 가깝에 위치해서 구별하기가 힘들다. 이를 개선하기 위해 나온 방법이 semantic segmentation 방법이다. 분할의 기본 단위를 클래스로 하여, 동일한 클래스에 해당하는 사물을 예측, 마스크 상에 동일한 색상으로 표시한다. PixelLink에서 사용한 Instance Segmentation은 분할의 기본 단위를 사물로 하여, 동일한 클래스에 해당하더라도 서로 다른 사물에 해당하면 이들을 예측 마스크 상에 다른 색상으로 표시한다. PixelLink는 location regression 없이도 텍스트 박스에서 직접적으로 segmentation 결과를 추출해 더 적은 수용영역이 필요하다. 이점은 학습을 더 쉽게 만든다.
end-to-end 접근방식은 감지모듈과 인식모듈이 동시에 훈련되어 인식결과가 향상되고 다시 detection의 정확도를 높이는 효과를 가져온다. end-to-end 방식에는 Fast Oriented Text Spotting (FOTS) 가 있다

### 2.2 text recognition
글자인식 부분에서는 feature를 추출하는 CNN과 시계열 모델인 RNN을 통합하여 하나의 통일된 네트워크 구조의 CRNN이 제안되었다. CRNN은 먼저 CNN을 통해 입력 이미지로부터 feature sequence를 추출하고 이 feature sequence들을 RNN의 입력값으로 하여 이미지의 텍스트 시퀀스를 예측한다. 예측된 텍스트 시퀀스를 텍스트로 변환하다. 이 모델은 미리 정해진 어휘에 제한되지 않고, 임의 길이의 시계열 데이터를 다룰 수 있다. Recurrent Convolution Layer에 있는 context 모듈을 제어하기 위해 RCNN에 gate가 더해졌다. gate는 context 모듈제어 뿐만 아니라 CNN의 정보와 RNN의 정보를 균형있게 만든다. 문자인식 부분에는 2가지 task가 있는데 constrained는 문자의 길이가 정해져있고 추론을 하는동안 사전활용이 가능하다. unconstrained 문자인식에서 각 단어는 사전없이 인식되어진다. 

### 2.3 end to end text recognition


## 3. datasets
### 3.2 Evaluation Protocols



