import pandas as pd
import multiprocessing as mp
import json
import torch
from transformers import AutoTokenizer, AutoModel
import os
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

# 데이터 경로
data_paths_file = './data/data_paths.csv'

# 데이터 경로 파일을 로드
df_paths = pd.read_csv(data_paths_file)

# 질문과 답변 데이터 경로 나누기
question_paths = df_paths[df_paths['File Path'].str.contains('1.질문')]['File Path'].tolist()
answer_paths = df_paths[df_paths['File Path'].str.contains('2.답변')]['File Path'].tolist()

# 질문 데이터 처리 함수
def process_question_files(file_paths):
    data_list = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                question_data = {
                    'fileName': data.get('fileName', ''),
                    'participantID': data['participantsInfo'].get('participantID', ''),
                    'gender': data['participantsInfo'].get('gender', ''),
                    'age': data['participantsInfo'].get('age', ''),
                    'history': data['participantsInfo'].get('history', ''),
                    'question': data.get('question', ''),
                    'rPlace': data['participantsInfo'].get('rPlace', ''),
                    'intention': data.get('intention', ''),
                    'disease_category': data.get('disease_category', ''),
                    'disease_name': data['disease_name'].get('kor', ''),
                    'disease_name_eng': data['disease_name'].get('eng', ''),
                    'entities_text': ', '.join([entity.get('text', '') for entity in data.get('entities', [])]),
                    'entities_entity': ', '.join([entity.get('entity', '') for entity in data.get('entities', [])]),
                    'entities_position': ', '.join([str(entity.get('position', '')) for entity in data.get('entities', [])])
                }
                data_list.append(question_data)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return data_list

# 답변 데이터 처리 함수
def process_answer_files(file_paths):
    data_list = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                answer_data = {
                    'fileName': data.get('fileName', ''),
                    'disease_category': data.get('disease_category', ''),
                    'disease_name': data['disease_name'].get('kor', ''),
                    'disease_name_eng': data['disease_name'].get('eng', ''),
                    'department': ', '.join(data.get('department', [])),
                    'intention': data.get('intention', ''),
                    'answer_intro': data['answer'].get('intro', ''),
                    'answer_body': data['answer'].get('body', ''),
                    'answer_conclusion': data['answer'].get('conclusion', '')
                }
                data_list.append(answer_data)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return data_list

# 데이터 청크로 나누기
def chunkify(lst, n):
    avg_chunk_size = len(lst) // n
    remainder = len(lst) % n
    start = 0
    for i in range(n):
        end = start + avg_chunk_size + (1 if i < remainder else 0)
        yield lst[start:end]
        start = end

num_chunks = 8
question_chunks = list(chunkify(question_paths, num_chunks))
answer_chunks = list(chunkify(answer_paths, num_chunks))

# 멀티프로세싱으로 질문 데이터 처리
with mp.Pool(processes=num_chunks) as pool:
    question_results = pool.map(process_question_files, question_chunks)

# 멀티프로세싱으로 답변 데이터 처리
with mp.Pool(processes=num_chunks) as pool:
    answer_results = pool.map(process_answer_files, answer_chunks)

# 결과를 하나의 리스트로 합침
questions = [item for sublist in question_results for item in sublist]
answers = [item for sublist in answer_results for item in sublist]

# 질문 데이터프레임 생성
question_df = pd.DataFrame(questions)
answer_df = pd.DataFrame(answers)

print('question data len:', len(question_df))
print('answer data len:', len(answer_df))

# ko-sroberta-multitask 모델과 토크나이저 불러오기
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# GPU가 사용 가능한지 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 텍스트 임베딩 함수 (배치 처리)
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 입력을 GPU로 이동
    with torch.no_grad():  # 그래디언트 비활성화
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # 결과를 CPU로 이동

# 질문과 답변 데이터 임베딩 계산 (배치 처리 적용)
batch_size = 32

question_embeddings = []
for i in tqdm(range(0, len(question_df), batch_size)):
    batch_texts = question_df['question'].iloc[i:i + batch_size].tolist()
    embeddings = get_embeddings(batch_texts)
    question_embeddings.extend(embeddings)

answer_embeddings = []
for i in tqdm(range(0, len(answer_df), batch_size)):
    batch_texts = answer_df['answer_body'].iloc[i:i + batch_size].tolist()
    embeddings = get_embeddings(batch_texts)
    answer_embeddings.extend(embeddings)

question_df['question_embedding'] = question_embeddings
answer_df['answer_body_embedding'] = answer_embeddings

print('Embedding complete')

# 사용하지 않는 tokenizer, model 삭제
del tokenizer, model
torch.cuda.empty_cache()

# 환경 변수 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_pairdata_name(sample, df2, df1_embedding_col, df2_embedding_col):
    temp_df2 = df2[
        # (df2['disease_category'] == sample['disease_category']) &
        (df2['disease_name'] == sample['disease_name']) &
        (df2['intention'] == sample['intention'])
    ]

    if temp_df2.empty:
        return None

    sample_embedding = sample[df1_embedding_col].reshape(1, -1)
    embeddings = np.stack(temp_df2[df2_embedding_col].values)
    distances = distance.cdist(sample_embedding, embeddings, 'cosine')
    best_match_idx = distances.argmin()
    best_match = temp_df2.iloc[best_match_idx]['fileName']
    
    return best_match

# 유사도 계산을 위한 카테고리별 멀티프로세싱 함수
def calculate_similarity_for_category(category):
    df_questions_cat = question_df[question_df['disease_category'] == category]
    df_answers_cat = answer_df[answer_df['disease_category'] == category]

    df_questions_cat['answer_fileName'] = df_questions_cat.apply(
        lambda row: get_pairdata_name(
            row, df_answers_cat, 'question_embedding', 'answer_body_embedding'
            ), axis=1
        )
    df_answers_cat['question_fileName'] = df_answers_cat.apply(
        lambda row: get_pairdata_name(
            row, df_questions_cat, 'answer_body_embedding', 'question_embedding'
            ), axis=1
        )

    return df_questions_cat, df_answers_cat

# 카테고리별로 멀티프로세싱 실행
categories = question_df['disease_category'].unique()
with mp.Pool(processes=len(categories)) as pool:
    results = list(
        tqdm(
            pool.imap(calculate_similarity_for_category, categories),
            total=len(categories)
        )
    )

# 결과 통합
question_results = [res[0] for res in results]
answer_results = [res[1] for res in results]

question_df = pd.concat(question_results, ignore_index=True)
answer_df = pd.concat(answer_results, ignore_index=True)

# 임베딩 컬럼 삭제
question_df = question_df.drop(columns=['question_embedding'], errors='ignore')
answer_df = answer_df.drop(columns=['answer_body_embedding'], errors='ignore')

# 결과 저장
question_df.to_csv('./data/question_with_answers.csv', index=False)
answer_df.to_csv('./data/answer_with_questions.csv', index=False)