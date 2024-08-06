import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_disease_info(disease_name):
    url = f"https://www.amc.seoul.kr/asan/healthinfo/disease/diseaseList.do?searchKeyword={disease_name}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 원하는 정보를 추출합니다. 아래 예시는 '증상' 부분을 추출하는 예시입니다.
    #symptom_info = soup.find_all('div', {'class': 'contBox'})
    contBoxes = soup.find_all('div', class_='contBox')

    if contBoxes:

        disease_data = []

        for contBox in contBoxes:
            disease_info = {}
            
            # 질병명
            disease_info['질병명'] = contBox.find('strong', class_='contTitle').get_text(strip=True)
            
            # 증상
            symptoms = [a.get_text(strip=True) for a in contBox.find_all('dd')[0].find_all('a')]
            disease_info['증상'] = ', '.join(symptoms)
            
            # 관련질환
            related_diseases = [a.get_text(strip=True) for a in contBox.find_all('dd')[1].find_all('a')]
            disease_info['관련질환'] = ', '.join(related_diseases)
            
            # 진료과
            medical_departments = [a.get_text(strip=True) for a in contBox.find_all('dd')[2].find_all('a')]
            disease_info['진료과'] = ', '.join(medical_departments)
            
            # 동의어
            try:
                synonyms = contBox.find_all('dd')[3].get_text(strip=True)
                disease_info['동의어'] = synonyms
            except:
                disease_info['동의어'] = ''

            disease_data.append(disease_info)

        # 데이터프레임으로 변환
        df = pd.DataFrame(disease_data)

        # 데이터프레임 출력
        #print(df)
        return df
    else:
        print('결과를 찾을 수 없습니다. 다시 검색합니다.')
        return pd.DataFrame(columns=['질병명', '증상', '관련질환', '진료과', '동의어'])
    


def get_disease_dataset(disease_name):
    df = get_disease_info(disease_name)
    if df.shape[0] == 0:
        if (' and ' in disease_name) or (' 및 ' in disease_name):
            disease_names = re.split(' and | 및 ', disease_name)
            print(disease_names)
            for disease_name in disease_names:
                try:
                    tmp = get_disease_info(disease_name.strip())
                    df = pd.concat([df, tmp])
                except:
                    print(disease_name)
                    continue
            return df
        else:
            print('결과를 찾을 수 없습니다')
            return df
    else:
        return df