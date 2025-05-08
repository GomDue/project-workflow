# 📘 MIDAS 모델 학습 및 데이터 파이프라인 구축

## 1. 프로젝트 개요

* **프로젝트명**: MIDAS 모델 학습 및 데이터 파이프라인 구축
* **프로젝트 기간**: 2024.03 ~ 2024.11
* **목적 및 배경**: 분리배출 안내 서비스 `분리위키`의 AI 기능을 위해,

  * **혐오 표현 분류 (KcBERT)**
  * **이미지 기반 재활용 품목 분류 (YOLOv8)**
    모델 학습 자동화 및 배포 파이프라인을 구축하였습니다.
    주기적인 학습/저장/적재를 통해 **지속적으로 갱신 가능한 AI 모델 관리 시스템**을 설계하고 구현하였습니다.


## 2. 기술 스택

| 구분                         | 기술                                                    |
| -------------------------- | ----------------------------------------------------- |
| **Language**               | Python 3.10                                           |
| **Orchestration**          | Apache Airflow                                        |
| **Distributed Processing** | Apache Spark                                          |
| **모델 학습**                  | HuggingFace Transformers (KcBERT), Ultralytics YOLOv8 |
| **Database**               | AWS RDS (MySQL), PostgreSQL                           |
| **Storage**                | AWS S3                                                |
| **Deployment**             | Docker, Docker Compose                                |
| **Monitoring**             | TensorBoard                                           |
| **ETL & 기타**               | Pandas, SQLAlchemy, loguru, boto3                     |


## 3. 주요 기능 및 흐름

* Apache Airflow를 이용한 **DAG 기반 자동화 학습 파이프라인 구축**

  * Google Sheet 및 AWS RDS 기반 데이터 수집 → Spark 처리 → PostgreSQL 저장
  * `Unsmile Dataset` 기반 KcBERT 혐오 발언 분류 모델 학습
  * AI-Hub 기반 재활용품 이미지 데이터셋 전처리 후 YOLOv8 모델 학습
* 학습된 모델은 `.pt` 형태로 저장되며, 버전별로 AWS S3에 업로드
* `latest_model.yaml`을 기준으로 FastAPI 서버와 연동 가능하도록 모델 버전 정보 관리
* TensorBoard를 연동하여 학습 로그 및 PR Curve 시각화


## 4. 시스템 아키텍처
### 구성 설명

* **데이터 수집 DAG (`midas_dag`)**

  * 팀원들이 Google Sheet에 기록한 지역별 분리배출 정책을 매일 자정에 수집
  * PostgreSQL에 저장된 기존 데이터와 비교하여 **신규 항목만 필터링** 및 저장

* **혐오 표현 필터링 DAG (`midas_hate_speech_dag`)**

  * 초기에 `Unsmile Dataset`을 전처리하여 `comment` 테이블에 저장
  * 이후에는 RDS에서 새로운 댓글만 수집하여 정제 후 저장
  * 수집이 완료되면 KcBERT 학습 DAG(`midas_training_dag`)을 트리거

* **KcBERT 학습 DAG (`midas_training_dag`)**

  * `params.yaml` 기반으로 학습 설정
  * 학습된 state\_dict 파일은 S3에 저장되고 `latest_model.yaml` 업데이트 수행

* **YOLOv8 학습 DAG (`midas_yolo_dag`)**

  * 전처리된 이미지 데이터셋을 기반으로 YOLO 모델 학습
  * `recycle.yaml` 기반 class 정의 및 데이터 경로 설정
  * 학습된 모델은 S3에 저장되며 `latest_model.yaml` 자동 갱신

* **모델 성능 시각화**

  * TensorBoard 로그를 자동 저장하여 모델 학습 상태 및 성능 확인 가능



## 5. 문제 해결 경험

* **EC2 프리티어 환경의 성능 문제**   
  KcBERT와 YOLOv8 모델을 EC2 프리티어 환경에서 서빙했을 때 **응답 속도가 수 분 이상 소요되는 문제**가 발생함.   
  → 시연 환경에서는 `ngrok`으로 임시 대응

* **모델 버전 관리 및 비용 이슈**   
  MLflow를 도입하려 하였으나, 구성 이해 부족 및 AWS 환경 설정 미숙으로 도입에 실패함. 이후 `.pt`, `params.yaml`, `latest_model.yaml`을 직접 저장하는 방식으로 대체함.   
  이 과정에서 **모델 용량 관리 미흡으로 AWS S3 과금 이슈가 발생**하였으며, AWS 지원팀에 문의하여 비용 면제를 받은 경험이 있음.

* **데이터 정제 및 클래스 통합 설계**   
  다양한 어노테이션 형식(BOX, POLYGON 등)을 구분하고, 학습에 적합한 BOX 타입만 필터링하여 사용함.   
  또한, **비슷한 클래스를 통합하여 총 9개의 클래스로 구성**함으로써 학습 효율성과 성능 향상을 도모함.
  


## 6. 프로젝트 회고 및 기술적 인사이트

* **모델 버전 및 실험 관리의 중요성 인식**   
  MLflow 도입 시도를 통해 실험과 버전 관리 체계의 필요성을 체감하였으며, 이후 수동 관리 방식의 한계를 경험함.
  → 추후에는 MLflow와 같은 툴을 적극 도입하여 실험 결과와 모델 버전을 통합 관리하는 방향의 필요성을 인식함.

* **로그 및 성능 시각화 구조의 명확성 필요**   
  TensorBoard를 활용한 시각화 도중, 모델명이나 디렉터리 구조가 불명확해 로그 해석이 어려웠던 경험이 있었음.
  → 명확한 로그 디렉터리 구조와 모델 이름 규칙을 설계하는 것이 중요하다는 점을 배움.

* **데이터 파이프라인의 디렉터리 구성 경험 부족**   
  설정 파일, DAG, Spark Job 등 파일 구조가 산발적으로 배치되어 관리에 비효율이 있었음.
  → 역할별로 명확하게 디렉터리를 분리하고 관리 체계를 표준화할 필요성을 절감함.

* **전이 학습 및 점진적 학습 설계의 필요성 인식**   
  기존 학습 모델을 기반으로 새로운 데이터를 반영하는 구조가 마련되어 있지 않았음.
  → 단순 재학습이 아닌 fine-tuning과 증분 학습의 흐름을 사전에 고려해야 한다는 점을 학습함.
