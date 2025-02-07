# **FSR 센서 기반 수면 자세 분류**

---

## **1. 프로젝트 개요**  
- **주제:** CNN을 이용한 FSR 센서 기반 수면 자세 분류  
- **목적:** 비구속적 센서를 활용하여 수면 자세를 실시간으로 분류하고, 자세 변화를 탐지함으로써 건강 관리 및 수면 질 향상을 목표로 함  
- **사용 데이터:** 28개의 FSR 센서 데이터를 활용하여 딥러닝 모델 구현  

---

## **2. 데이터 구성**  

### **2.1. 데이터 수집**  
- **장비:** FSR 4 channel (7개의 센서)
- **샘플링 주파수:** 10 Hz (1초당 10 프레임 수집)  
- **실험 대상:** 60명 
- **수집 자세:**  
   - **정자세 (Supine)**  
   - **왼쪽으로 누운 자세 (Left Lateral)**  
   - **오른쪽으로 누운 자세 (Right Lateral)**  
   - **엎드린 자세 (Prone)**
- **실험 시간:** 자세 당 15분, 총 60분
- **데이터 크기:**  
   - 각 자세당 **9000 프레임**
   - 4개의 센서 x 채널 당 7개의 센서 → 총 28 센서 데이터  

### **2.2. 데이터 형태**  
- 4개의 채널별 **4x7** 크기의 행렬   

---

## **3. 데이터 전처리**  
### **3.1. 데이터 재구성**  
- **데이터 재구성:**
- 각 자세당 양끝 50초씩 자름
- ex) reorganized_data_1s(i, j, 1:800) = current_array(j, 51:850);
  
### **3.2. 데이터 분할** 
- **윈도우 사이즈:**
  - 1초 윈도우 자세당 800 프레임
  - 5초 윈도우 자세당 160 프레임
  - 10초 윈도우 자세당 80 프레임
  - 10개씩 묶어서 평균냄 -> 초당 사진을 활용함


### **3.3. 데이터 정규화**  
- **정규화 기법:** Z-score Normalization

   -  $z = \frac{x - \mu}{\sigma}$  
   - 피험자 간의 신체 체중 차이를 최소화  

---

## **4. 모델링**  
### **4.1. CNN 모델 3 layer**  
- **입력 데이터:** (batch, 1, 4, 7)  
- **구조:**
   - **Convolution block**  
        - **Conv2D Layer**: 각 블록에서 입력 데이터를 기준으로 특징 추출
        - **BatchNormalization**: batch간 normalization을 적용하여 피험자간의 생길 수 있는 차이를 보완
        - **Relu**: activation function
        - **Pooling Layer**: Max Pooling으로 다운샘플링
   - **Flatten**
   - **Dense block**: 최종 출력층에서 4개 및 3개의 수면자세를 분류  
- **성과:**
   - 3 class 테스트 정확도
      - 1초 윈도우: 약 **84%**
      - 5초 윈도우: 약 **86%**
      - 10초 윈도우: 약 **86%**
   - 4 class 테스트 정확도
      - 1초 윈도우: 약 **80%**
      - 5초 윈도우: 약 **82%**
      - 10초 윈도우: 약 **82%** 

### **4.2. CNN 모델 2 layer**  
- **입력 데이터:** (batch, 1, 4, 7)  
- **구조:**
   - **Convolution block**  
        - **Conv2D Layer**: 각 블록에서 입력 데이터를 기준으로 특징 추출
        - **BatchNormalization**: batch간 normalization을 적용하여 피험자간의 생길 수 있는 차이를 보완
        - **Relu**: activation function
        - **Pooling Layer**: Max Pooling으로 다운샘플링
   - **Flatten**
   - **Dense block**: 최종 출력층에서 4개 및 3개의 수면자세를 분류  
- **성과:**
   - 3 class 테스트 정확도
      - 1초 윈도우: 약 **85%**
      - 5초 윈도우: 약 **85%**
      - 10초 윈도우: 약 **85%**
   - 4 class 테스트 정확도
      - 1초 윈도우: 약 **79%**
      - 5초 윈도우: 약 **80%**
      - 10초 윈도우: 약 **81%**

### **4.3. CNN 모델 1 layer** 
- **입력 데이터:** (batch, 1, 4, 7)  
- **구조:**
   - **Convolution block**  
        - **Conv2D Layer**: 각 블록에서 입력 데이터를 기준으로 특징 추출
        - **BatchNormalization**: batch간 normalization을 적용하여 피험자간의 생길 수 있는 차이를 보완
        - **Relu**: activation function
        - **Pooling Layer**: Max Pooling으로 다운샘플링
   - **Flatten**
   - **Dense block**: 최종 출력층에서 4개 및 3개의 수면자세를 분류  
- **성과:**
   - 3 class 테스트 정확도
      - 1초 윈도우: 약 **84%**
      - 5초 윈도우: 약 **85%**
      - 10초 윈도우: 약 **85%**
   - 4 class 테스트 정확도
      - 1초 윈도우: 약 **80%**
      - 5초 윈도우: 약 **79%**
      - 10초 윈도우: 약 **80%**
        
### **4.4.  Vision Transformer 모델**  
- **데이터 준비:** pretrain dataset의 특성에 맞게 기존 데이터를 RGB 형태로 변환  
- **모델:** Pretrained Vision Transformer (ViT) Fine-tuning
     - huggingface 라이브러리를 이용하여 cifar10으로 pretrained ViT를 사용
     - vit_model_selection.py 파일을 이용하여 데이터셋에 맞는 모델을 탐색 후 사용.
  
- **최종 모델**: vit_tiny_patch16_224

---

## **5. 프로젝트 결과 요약**  
- **최종 성능**  
   - CNN 모델 3 layer, 3 class
     
   | True\Predicted   | supine & prone | left       | right      |
   |------------------|----------------|------------|------------|
   | supine & prone   | 92.32          | 3.00       | 4.66       |
   | left             | 15.54          | 79.69      | 4.75       |
   | right            | 14.60          | 5.04       | 80.34      |

   -  Vision Transformer 모델

   | True\Predicted  | supine   | left     | right    | prone    |
   |-----------------|----------|----------|----------|----------|
   | supine          | 100.00   | 0.00     | 0.00     | 0.00     |
   | left            | 0.00     | 86.25    | 0.00     | 13.75    |
   | right           | 0.00     | 0.00     | 96.75    | 3.25     |
   | prone           | 0.00     | 0.00     | 6.50     | 93.50    |

  
---

## **6. 추가 개선 사항**
1. **Experiment**
   - 팔과 다리 각도 조정(30°, 45°, 60°), 수면 자세 세분화를 통해 다양한 수면 자세를 분류해보며 프로젝트를 확장해보면 좋을 것 같음.
   - 기존 수면 자세 분류의 경우, 침대 전체에 FSR 센서를 설치하여 실험이 진행됨. 본 프로젝트에서는 단 28개의 센서만을 사용하여 높은 성능을 달성함. 이를 실제 가정환경에 도입 하여 욕창과 같은 질병을 예방하는 자동화 시스템으로 확장하면 좋을 것 같음.

2. **Preprocessing**
   - 특정 자세에서 자세를 잘못 예측하는 경우가 있음(예: left, prone). 데이터의 특징을 분석해보며 잘못 분류된 샘플에서 공통적인 특징이나 특성, 예를 들면 모호한 자세가 많은지, 신체 부위가 겹치는지 등을 찾아보며 데이터의 수집을 조정하는 것이 필요해보임.
