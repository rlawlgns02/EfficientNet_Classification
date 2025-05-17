import os
from roboflow import Roboflow

# Roboflow API 초기화 (API 키 설정)
rf = Roboflow(api_key="2iQVik1UmC6foSnxOrYY")

# phone_classification 프로젝트 접근
project = rf.workspace("project-pfubb").project("phone_classification")

# 최신 버전의 데이터셋 다운로드
dataset = project.version(2).download("folder")

print(f"데이터셋이 {dataset.location} 위치에 다운로드되었습니다.")
