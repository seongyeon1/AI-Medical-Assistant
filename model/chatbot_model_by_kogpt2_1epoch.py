import pandas as pd
import numpy as np
import random
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling

df = pd.read_csv('./data/text_only.csv', index_col = 0)
texts = "[CLS] " + df['question'] + " [SEP] " + df['answer_body'] + " [EOS]"
# texts = texts.tolist()

random.seed(526)

# 데이터 총 길이
num_texts = len(texts)

# val, test 데이터 개수
test_size = int(num_texts * 0.1)
val_size = int(num_texts * 0.1)

# 전체 인덱스 생성
indices = list(range(num_texts))
random.shuffle(indices)

# 테스트 인덱스와 학습 인덱스 분리
test_indices = indices[: test_size]
val_indices = indices[test_size : test_size + val_size]
train_indices = indices[test_size + val_size :]

# `texts`는 pandas.Series 객체이므로, 인덱스를 정수형으로 변경
texts = texts.reset_index(drop=True)

test_texts = texts.iloc[test_indices].tolist()
val_texts = texts.iloc[val_indices].tolist()
train_texts = texts.iloc[train_indices].tolist()

# 모델을 GPU로 이동 (가능할 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 토크나이저와 모델 로드
model_name = 'skt/kogpt2-base-v2'
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)

# model 레이어 freezing
for name, param in model.named_parameters():
    # 임베딩 레이어 freezing
    if 'transformer.wte' in name or 'transformer.wpe' in name:
        param.requires_grad = False
    # 블록 freezing 설정
    elif 'transformer.h.' in name:
        block_num = int(name.split('.')[2])
        if block_num < 8:  # 8 블록까지 freezing
            param.requires_grad = False
    # 최종 출력 레이어
    elif 'transformer.ln_f' in name:
        param.requires_grad = True

# 특수 토큰 추가
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'bos_token': '[CLS]', 'eos_token': '[EOS]', 'sep_token': '[SEP]'})

# 추가한 토큰 resize
model.resize_token_embeddings(len(tokenizer))

# 토큰화 및 인코딩
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# PyTorch Dataset 생성
class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = QADataset(train_encodings)
val_dataset = QADataset(val_encodings)
test_dataset = QADataset(test_encodings)

# 데이터 콜레이터 생성
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 모델 학습
trainer.train()

# 모델을 GPU로 이동 (CUDA가 사용 가능한 경우)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 예측 실행
def predict(texts):
    # 토큰화 및 인코딩
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    # 모든 입력 텐서를 동일한 디바이스로 이동
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    # 예측
    with torch.no_grad():
        outputs = model.generate(**encodings, max_length=256, num_beams=5, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    
    # 텍스트 디코딩
    decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return decoded_outputs

# 예측할 질문 리스트
test_texts = [
    "폐렴의 증상 중 면역력이 떨어지면 어떤 합병증이 발생하게 되는지 알려주세요."
]

# 예측 결과
predictions = predict(test_texts)
for i, text in enumerate(test_texts):
    print(f"질문: {text}")
    print(f"답변: {predictions[i]}")