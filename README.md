# MIDAS 모델 학습 및 데이터 파이프라인 구축

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

* **모델 버전 관리 및 비용 이슈**   
  MLflow를 도입하려 하였으나, 구성 이해 부족 및 AWS 환경 설정 미숙으로 도입에 실패함. 이후 `.pt`, `params.yaml`, `latest_model.yaml`을 직접 저장하는 방식으로 대체함.   
  이 과정에서 **모델 용량 관리 미흡으로 AWS S3 과금 이슈가 발생**하였으며, AWS 지원팀에 문의하여 비용 면제를 받은 경험이 있음.

* **데이터 정제 및 클래스 통합 설계**   
  노트북 환경의 저장 공간 제약으로 인해 전체 데이터셋을 사용할 수 없었고, 학습에 적합한 BOX 타입 어노테이션만 선별하여 사용함.   
  또한, 분리배출 안내 서비스에서 실제로 분류해야 할 대상은 9종에 한정되어 있었기 때문에,   
  유사한 클래스를 통합하여 총 9개의 클래스로 재구성함.
  


## 6. 프로젝트 회고 및 기술적 인사이트

* **모델 버전 및 실험 관리의 중요성 인식**   
  MLflow를 제대로 이해하지 못한 채 도입을 시도하면서 구조와 설정에 대한 부족한 이해로 인해 실패를 겪음.    
  이를 통해 버전 관리 및 실험 기록의 중요성을 실감하게 되었고, 다음 프로젝트에서는 MLflow를 충분히 학습한 후 체계적으로 재도입할 계획.

* **데이터 파이프라인의 디렉터리 구성 경험 부족**   
  설정 파일, DAG, Spark Job 등 다양한 파일들이 산발적으로 배치되어 있어 관리 및 협업에 비효율이 발생함.   
  이를 통해 Airflow 프로젝트 구조 전반에 대한 이해 부족을 느꼈으며,   
  특히 Spark와 Airflow를 함께 사용하는 경우 어떤 디렉터리 구조가 효과적인지 체계적으로 공부할 필요성을 절감함.

* **전이 학습 및 점진적 학습 설계의 필요성 인식**   
  기존 학습 모델을 기반으로 새로운 데이터를 반영하는 구조가 마련되어 있지 않았음.   
  추후에는 단순 재학습이 아닌 fine-tuning과 전이 학습도 가능하게 구현.

* **과도한 범위 설정으로 인한 파이프라인 미완성의 아쉬움**   
  새로운 기술 학습(Airflow, Spark 등)과 ML 모델 구현, ML 서빙 서버 구축, MLOps 파이프라인 구성 등 방대한 범위의 업무를 동시에 수행하다 보니,   
  일부 파이프라인은 계획대로 완성하지 못한 점이 아쉬움으로 남음.   
  이번 경험을 통해 자신의 역량과 학습 속도에 맞는 프로젝트 규모를 설정하는 것의 중요성을 깨달았고,   
  다음 프로젝트에서는 더 명확한 범위 설정과 우선순위 조정을 통해 완성도를 높이고자 함.
