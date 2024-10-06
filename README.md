# AI 의료 진단 챗봇 (AI-Medical-Assistant)

## 개요
> **아이펠 mini Aifflethon 프로젝트** <br/> **프로젝트 기간: 2024.08.02 ~ 2024.08.08** <br/>
> **발표자료 보러가기** :
> - [발표자료](./미니 아이펠톤.pdf)<br>

## Setup
```bash
# Docker 이미지 빌드
docker build -t streamlit-app .

# Docker 컨테이너 실행
docker run -p 8501:8501 streamlit-app
```

## 사용 기술
- streamlit
- fastapi
- huggingface
- ollama

## Datasets
![Screenshot 2024-10-06 at 9 56 29 PM](https://github.com/user-attachments/assets/934bcf8c-6a0c-4ee1-bc68-0ae661e8d735)

## 사용 모델
(LLM) heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF
(Embedding) jhgan/ko-sroberta-multitask


## Process
![Screenshot 2024-10-06 at 9 58 42 PM](https://github.com/user-attachments/assets/a4069bec-777b-4f68-9b33-abc1790fa30f)

## 시연영상
- 링크 : https://youtu.be/PKL_7jF4HpA?si=D1xLo2BRaZx47mph

https://github.com/user-attachments/assets/d5058c8f-529a-4b95-84fb-3e507c4b9e20


### 질병예측페이지
![질병예측페이지](https://github.com/user-attachments/assets/3e51ba4c-0402-4f6c-b546-20513a36a841)

### 병원지도
![병원지도](https://github.com/user-attachments/assets/b24ab9b6-1ba8-478c-8162-83f84e8618e2)

