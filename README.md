# 📘 MIDAS 모델 학습 및 데이터 파이프라인 구축

## 1. 프로젝트 개요

* **프로젝트명**: MIDAS 모델 학습 및 데이터 파이프라인 구축
* **프로젝트 기간**: 2024.03 \~ 2024.05
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

* **모델 로딩 방식의 구조적 한계**
  FastAPI 서버 시작 시 모델을 S3에서 로드하는 구조로 구현하였으나, 새로운 모델 생성 후 서버를 재기동해야 하는 문제가 있었습니다.
  → 추후 개선을 위해 `load_model` API를 도입하고, 모델 경로를 전달받아 **런타임 중 갱신 가능한 구조**로 전환하는 방향을 구상하였습니다.

* **EC2 환경의 성능 제약**
  KoBERT와 YOLOv8 모델을 프리티어 환경에서 서빙했을 때 응답 속도가 5분 이상 소요되는 등의 문제 발생
  → 시연 목적에서는 `ngrok`으로 로컬 서버를 외부에 노출해 해결하였고, 실제 운영을 고려한 인프라 확장의 필요성을 느꼈습니다.

* **모델 버전 관리 실패 경험**
  MLflow를 도입하려 했으나 구조 및 AWS 환경 이해 부족으로 실패
  → S3에 `.pt` 파일과 `params.yaml`을 직접 저장하고, `latest_model.yaml`을 통해 버전 관리를 수동으로 수행
  이 과정에서 **모델 용량 관리 미흡으로 과금 이슈**가 발생했으며, AWS에 비용 면제를 요청하여 해결했습니다.

* **데이터 전처리 및 분류 체계 설계**
  AI-Hub의 다양한 어노테이션 형식(JSON, BOX, POLYGON 등)을 분류하고 학습에 적합한 형태로 가공
  → BOX 타입만 선별하여 약 4만여 장의 이미지 중 9개 클래스로 통합 정제, 모델 성능 향상에 기여했습니다.

* **Airflow, Spark, Docker 등 초기 도입 기술들의 설계 부족**
  다양한 기술을 처음부터 도입하면서 **디렉터리 구조, 설정 파일 관리, DAG 구성의 일관성 부족**을 경험
  → `.env`, `.yaml`, DAG, Spark Job 파일의 관리 방식에 대해 설계적 고민의 중요성을 체감하였습니다.


## 6. 추후 개선 계획

* **모델 재로딩 API 및 SQS 연동 자동화**

  * `/reload_model` API를 구현하여 서버 재시작 없이 모델을 갱신할 수 있도록 개선 예정

* **모델 버전 및 실험 관리 체계 수립**

  * S3 경로를 기준으로 모델 버전을 폴더 단위로 관리하고, rollback 기능 도입 예정
  * 추후에는 MLflow 으로 **버전 및 실험 결과 통합 관리** 계획

* **TensorBoard 및 로그 시각화 개선**

  * 모델 이름 구분이 명확하지 않아 로그 해석이 어려웠던 초기 구조 개선
  * 모델별 로그 디렉터리 및 명확한 naming rule 적용 예정

* **데이터 파이프라인 구조 재정비**

  * 현재 산발적으로 흩어져 있는 설정 파일, 데이터 스크립트, DAG 파일 등을 역할에 따라 디렉터리 재구성
  * 관리 및 협업 효율성을 높이기 위한 구조 표준화 진행 예정

* **전이 학습 및 증분 학습 기능 강화**

  * 기존 모델을 기반으로 신규 데이터를 반영하는 **fine-tuning 흐름** 설계
  * 하이퍼파라미터 변경뿐 아니라 데이터 변화에도 유연하게 대응하는 구조 구현 목표
