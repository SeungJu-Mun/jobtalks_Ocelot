import pandas as pd
import hashlib

def simhash(text):
    # 전처리: 소문자 변환, 구두점 제거, 공백 제거
    text = re.sub(r'[\W_]+', '', str(text).lower())

    # 해시 함수 선택 (sha-1 사용)
    hash_func = hashlib.sha1

    # 문자열을 해시 값으로 변환
    hashed_text = [hash_func(token.encode('utf-8')).hexdigest() for token in [text[i:i+4] for i in range(0, len(text), 4)]]

    # 이진 해시 값 생성
    hash_value = ''.join([bin(int(h, 16))[2:].zfill(32) for h in hashed_text])

    # SIMHASH 값 계산
    simhash_value = ''.join(['1' if hash_value.count('1', i, i+32) > 16 else '0' for i in range(0, len(hash_value), 32)])

    return int(simhash_value, 2)

# 자기소개서 데이터셋 불러오기
df = pd.read_csv('예시-자기소개서-데이터셋.csv')

# SIMHASH 계산 및 중복 제거
df['simhash'] = df['Answer'].apply(simhash)
deduped_df = df.drop_duplicates(subset=['simhash'], keep='first')

deduped_df = deduped_df[['질문','답변']]
deduped_df.to_csv('저장할-데이터셋-이름.csv',index=False)
