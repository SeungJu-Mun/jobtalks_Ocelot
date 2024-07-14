import pandas as pd
import hashlib

# 중복 제거
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

# 특수문자 제거 예시
def remove_character(text):
    # \r\n, \r, \n제거
    text.str.replace(r'\r\n|\r|\n','')

    # 한글 및 소제목[]을 제외한 나머지 문자 제거
    text.str.replace(r'[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 \[\]]','')

    # 좋은점, 아쉬운점, 글자수 byte 제거하기
    text.str.replace(r'(아쉬운점|좋은점|글자수)\s*\d*[자Byte,]*', '', regex=True)

# 특수문자 제거 및 중복 데이터 제거
df['특수문자를 제거 할 컬럼'] = df['특수문자를 제거 할 컬럼'].apply(remove_character)
df['simhash'] = df['중복을 제거 할 컬럼'].apply(simhash)
deduped_df = df.drop_duplicates(subset=['simhash'], keep='first')


deduped_df = deduped_df[['질문','답변']]
deduped_df.to_csv('저장할-데이터셋-이름.csv',index=False)
