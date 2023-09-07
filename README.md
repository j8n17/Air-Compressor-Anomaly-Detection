# 제4회 2023 연구개발특구 AI SPARK 챌린지

## 개요
범천(주)은 ESG 가치를 담아 산업용 공기압축기를 개발하는 대덕연구개발특구 소재 기업으로 이 챌린지는 산업기기 피로도 예측하는 챌린지이다.

산업용 공기압축기 및 회전기기에서 모터 및 심부 온도, 진동, 노이즈 등은 기기 피로도에 영향을 주는 요소이며, 피로도 증가는 장비가 고장에 이르는 원인이 된다.
따라서 데이터를 통해 산업기기 이상 전조증상을 예측하여 기기 고장을 예방하고 그로 인한 사고를 예방하는 모델을 개발하는 것이 목표이다.

## 모델 조건

1. 비지도 학습 모델
2. 0 또는 1로 구분하는 이진 분류 모델
3. 시간 단위로 생성되는 입력 데이터에 대해 판정할 수 있는 모델
4. 신규 데이터로 학습 가능한 모델
5. 총 8개의 대상 설비에 대해 모두 동일한 아키텍처를 사용, 별도의 모델로 학습은 허용

## 데이터 셋
비지도 학습이고 이상치 탐지 모델이므로 train data에는 정상치 데이터만, test data에는 정상치와 이상치 데이터로 구성되어 있다.
<p align="center"><img width="514" alt="image" src="https://github.com/j8n17/Air-Compressor-Anomaly-Detection/assets/85532197/70c0c1d1-bdc9-47eb-af0e-8f1664e5eb7a"></p>


```
- air_inflow: 공기 흡입 유량 (^3/min)
- air_end_temp: 공기 말단 온도 (°C) - 공기를 압축하는 부분 == air_end
- out_pressure: 토출 압력 (Mpa)
- motor_current: 모터 전류 (A)
- motor_rpm: 모터 회전수 (rpm)
- motor_temp: 모터 온도 (°C)
- motor_vibe: 모터 진동 (mm/s)
- type: 설비 번호

설비 번호별 마력
- 설비 번호 [0, 4, 5, 6, 7]: 30HP(마력)
- 설비 번호 1: 20HP
- 설비 번호 2: 10HP
- 설비 번호 3: 50HP
```

## 모델링

### 전처리

![image](https://github.com/j8n17/Air-Compressor-Anomaly-Detection/assets/85532197/5b933fac-7b0e-485a-8536-91188251ffdc) | ![image](https://github.com/j8n17/Air-Compressor-Anomaly-Detection/assets/85532197/4bdcf5f3-f53d-44db-becd-060039e8143d)
--- | --- |

이상치 탐지 모델이므로 train 데이터로 정상치 데이터만 존재해야 하는데, type 1만 train 데이터에 이상치가 존재함을 발견해 이를 삭제후 학습을 진행했다.

또한 MinMax, Robust 등 스케일러 사용 후 최종적으로 Robust 스케일러로 채택했다.

### 모델

`AutoEncoder` (latent dimension : 8)

Encoder를 통해 데이터를 압축한 후 압축된 데이터를 Decoder를 통해 복원하는 AutoEncoder를 통해 원본 데이터와 복원 데이터의 오차(Reconstruction Error)가 크면 이상치라고 판단한다.

또한 AutoEncoder를 학습시키기 위해 MSE Loss, Adam Optimizer를 사용했다.

### 이상치 판단

학습된 AutoEncoder에 Train data로 예측을 진행하고 이를 통해 구한 최대 Reconstruction Error(MSE)를 Threshold로 설정한다.
그 후 Test data의 Reconstruction Erorr가 Threshold 보다 큰 것을 이상치라고 판단한다.

## 결과


<p align="center"><img width="543" alt="image" src="https://github.com/j8n17/Air-Compressor-Anomaly-Detection/assets/85532197/fa6eb76e-12fc-4b3e-89cb-2d08b5c881f1"></p>

```
정상치/이상치 개수
0    7006
1     383

type별 이상치 개수
0    146
6    101
2     55
5     30
3     27
4     13
1      7
7      4
```

## 시행 착오
### AutoEncoder latent 차원 크기 설정
column의 수가 8개밖에 되지 않기 때문에 이를 압축한 후 다시 복원하면 반드시 손실이 될 것이라 생각했다.
그래서 기존의 AutoEncoder와 반대로 차원을 키우고 다시 압축하는 방식으로 모델링을 했었고 성능이 잘 나오는 것을 확인했다. (`AutoDecoder.ipynb`)

하지만 선형대수학의 관점으로 봤을 때 column이 8이어서 rank는 최대 8이 되므로 어떤 짓을 해도 rank가 8 이상으로 올라가지 않는다.
고로 내가 `AutoDecoder.ipynb` 또는 `AutoDecoder_64.ipynb`에서 사용한 차원을 늘렸다 줄이는 방식은 결국 8차원으로 표현할 수 있는 정보를 다차원으로 표현했다가 줄이는 것이므로 의미가 없다.

그래서 결국 `AutoEncoder_8.ipynb`처럼 8차원 데이터를 Encoder를 통해 8차원으로 재구성하고 Decoder를 통해 8차원으로 복원하는 방식으로 더 좋은 모델을 구현했다.

