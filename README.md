# 📘 MIDAS 모델 학습 및 데이터 파이프라인 구축

## 1. 프로젝트 개요

- **프로젝트명**: MIDAS 모델 학습 및 데이터 파이프라인 구축
- **기간**: 2024.03 ~ 2024.05
- **설명**: 분리배출 안내 서비스 `분리위키`의 AI 기능을 위해,
  - **혐오 표현 분류 (KoBERT)**
  - **이미지 기반 재활용 품목 분류 (YOLOv8)**
  
  모델 학습 자동화 및 배포 파이프라인을 구축한 프로젝트입니다.  
  주기적인 학습/저장/적재를 통해 **지속적으로 갱신 가능한 AI 모델 관리 시스템**을 구축하였습니다.

---

## 2. 기술 스택

| 구분 | 기술 |
|------|------|
| Language | Python 3.10 |
| Orchestration | Apache Airflow |
| Distributed Processing | Apache Spark |
| 모델 학습 | HuggingFace Transformers (KoBERT), YOLOv8 |
| DB | AWS RDS (MySQL), PostgreSQL |
| Storage | AWS S3 |
| Deployment | Docker, Docker Compose |
| 기타 | Pandas, SQLAlchemy, TensorBoard |

---

## 3. 주요 흐름

- Airflow DAG 스케줄링 기반 **정기적인 데이터 수집 → 전처리 → 학습 → S3 저장**
- Spark로 Google Sheet 및 RDS 데이터 병합/정제 처리
- 학습된 모델은 `.pt` 형태로 S3에 버전 관리하여 저장
- TensorBoard로 성능 및 PR Curve 모니터링
- 추후 FastAPI 서버 및 SQS 연동 기반 자동 모델 로딩 계획 중

---

## 4. 시스템 아키텍처

> DAG 기반 전체 구조
> - Hate Speech 수집 및 분류: `midas_hate_speech_dag`, `midas_training_dag`
> - 이미지 기반 분류 학습: `midas_yolo_dag`
> - Google Sheet 기반 정제: `midas_dag`

