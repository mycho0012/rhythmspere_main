import pandas as pd
from datetime import datetime
import numpy as np
import plotly.graph_objects as go


# 천간과 지지 정의 (한글과 한자 분리)
SKY = [("갑", "甲"), ("을", "乙"), ("병", "丙"), ("정", "丁"), ("무", "戊"),
       ("기", "己"), ("경", "庚"), ("신", "辛"), ("임", "壬"), ("계", "癸")]
GROUND = [("자", "子"), ("축", "丑"), ("인", "寅"), ("묘", "卯"), ("진", "辰"), ("사", "巳"),
          ("오", "午"), ("미", "未"), ("신", "申"), ("유", "酉"), ("술", "戌"), ("해", "亥")]

# 시주 조견표 (한글만)
TIME_PILLAR_TABLE = {
    "갑기": ["갑자", "을축", "병인", "정묘", "무진", "기사", "경오", "신미", "임신", "계유", "갑술", "을해"],
    "을경": ["병자", "정축", "무인", "기묘", "경진", "신사", "임오", "계미", "갑신", "을유", "병술", "정해"],
    "병신": ["무자", "기축", "경인", "신묘", "임진", "계사", "갑오", "을미", "병신", "정유", "무술", "기해"],
    "정임": ["경자", "신축", "임인", "계묘", "갑진", "을사", "병오", "정미", "무신", "기유", "경술", "신해"],
    "무계": ["임자", "계축", "갑인", "을묘", "병진", "정사", "무오", "기미", "경신", "신유", "임술", "계해"]
}

OPPOSITE_GROUND = {
    "인": "신", "신": "인",
    "묘": "유", "유": "묘",
    "진": "술", "술": "진",
    "사": "해", "해": "사",
    "오": "자", "자": "오",
    "미": "축", "축": "미"
}
# 24절기와 월지의 대응 관계
SOLAR_TERMS = {
    "입춘": "인", "우수": "인", "경칩": "묘", "춘분": "묘",
    "청명": "진", "곡우": "진", "입하": "사", "소만": "사",
    "망종": "오", "하지": "오", "소서": "미", "대서": "미",
    "입추": "신", "처서": "신", "백로": "유", "추분": "유",
    "한로": "술", "상강": "술", "입동": "해", "소설": "해",
    "대설": "자", "동지": "자", "소한": "축", "대한": "축"
}


def load_saju_data(file_path):
    return pd.read_csv(file_path, encoding='utf-8')

def find_saju(df, birth_date):
    saju = df[df['그레고리력'] == birth_date]
    if len(saju) == 0:
        return None
    return saju.iloc[0]['음력간지']

def parse_saju(saju_str):
    parts = saju_str.split()
    year = parts[0].rstrip('년')
    month = parts[1].rstrip('월') if len(parts) > 2 and '월' in parts[1] else ""
    day = parts[-1].rstrip('일')
    return year, month, day

def find_ground(hour, minute):
    time_obj = datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()
    time_ranges = [
        (datetime.strptime("23:30", "%H:%M").time(), datetime.strptime("01:30", "%H:%M").time()),
        (datetime.strptime("01:30", "%H:%M").time(), datetime.strptime("03:30", "%H:%M").time()),
        (datetime.strptime("03:30", "%H:%M").time(), datetime.strptime("05:30", "%H:%M").time()),
        (datetime.strptime("05:30", "%H:%M").time(), datetime.strptime("07:30", "%H:%M").time()),
        (datetime.strptime("07:30", "%H:%M").time(), datetime.strptime("09:30", "%H:%M").time()),
        (datetime.strptime("09:30", "%H:%M").time(), datetime.strptime("11:30", "%H:%M").time()),
        (datetime.strptime("11:30", "%H:%M").time(), datetime.strptime("13:30", "%H:%M").time()),
        (datetime.strptime("13:30", "%H:%M").time(), datetime.strptime("15:30", "%H:%M").time()),
        (datetime.strptime("15:30", "%H:%M").time(), datetime.strptime("17:30", "%H:%M").time()),
        (datetime.strptime("17:30", "%H:%M").time(), datetime.strptime("19:30", "%H:%M").time()),
        (datetime.strptime("19:30", "%H:%M").time(), datetime.strptime("21:30", "%H:%M").time()),
        (datetime.strptime("21:30", "%H:%M").time(), datetime.strptime("23:30", "%H:%M").time())
    ]
    for i, (start, end) in enumerate(time_ranges):
        if start <= time_obj < end or (i == 0 and (time_obj >= start or time_obj < end)):
            return GROUND[i]
    return GROUND[0]  # Default to 자(子) if not found

def calculate_time_pillar(day_sky, hour, minute):
    ground_time = find_ground(hour, minute)
    day_sky_group = "갑기" if day_sky in ["갑", "기"] else \
                    "을경" if day_sky in ["을", "경"] else \
                    "병신" if day_sky in ["병", "신"] else \
                    "정임" if day_sky in ["정", "임"] else "무계"
    time_index = GROUND.index(ground_time)
    return TIME_PILLAR_TABLE[day_sky_group][time_index]

def get_hanja(korean_char):
    for sky in SKY:
        if sky[0] == korean_char:
            return sky[1]
    for ground in GROUND:
        if ground[0] == korean_char:
            return ground[1]
    return korean_char  # 일치하는 한자가 없으면 원래 문자 반환

def get_saju(input_str,df):
    
    birth_date = datetime.strptime(input_str, "%Y%m%d%H%M")
    birth_date_str = birth_date.strftime("%Y-%m-%d")

    saju_str = find_saju(df, birth_date_str)

    # 윤달 처리
    if saju_str is None:
        prev_date = birth_date - pd.Timedelta(days=1)
        prev_date_str = prev_date.strftime("%Y-%m-%d")
        saju_str = find_saju(df, prev_date_str)
        if saju_str is None:
            return "입력된 날짜에 대한 사주 정보를 찾을 수 없습니다."

    year_pillar, month_pillar, day_pillar = parse_saju(saju_str)

    # 일간(日干) 추출
    day_sky = day_pillar[0]

    time_pillar = calculate_time_pillar(day_sky, birth_date.hour, birth_date.minute)
    time_pillar_with_hanja = f"{time_pillar[0]}{time_pillar[1]}({get_hanja(time_pillar[0])}{get_hanja(time_pillar[1])})"

    return f"{year_pillar}년 {month_pillar}월 {day_pillar}일,{time_pillar_with_hanja}시"


def adjust_month_ground(day_sky, month_ground):
    """
    일간과 월지의 홀짝 관계를 확인하고 필요시 조정합니다.
    
    :param day_sky: 일간 (천간)
    :param month_ground: 월지 (지지)
    :return: 조정된 월지
    """
    CELESTIAL_STEMS = ["갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"]
    TERRESTRIAL_BRANCHES = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]

    day_index = CELESTIAL_STEMS.index(day_sky)
    month_index = TERRESTRIAL_BRANCHES.index(month_ground)
    
    if day_index % 2 == 0 and month_index % 2 != 0:  # 일간 홀수, 월지 짝수
        return TERRESTRIAL_BRANCHES[(month_index + 1) % 12]
    elif day_index % 2 != 0 and month_index % 2 == 0:  # 일간 짝수, 월지 홀수
        return TERRESTRIAL_BRANCHES[(month_index + 1) % 12]
    else:
        return month_ground

def get_ipchun_same(saju_result):
    # 사주 결과에서 일간과 월지 추출
    parts = saju_result.split(',')
    day_pillar = parts[0].split()[-1]  # 일주
    month_pillar = parts[0].split()[1]  # 월주

    day_sky = day_pillar[0]  # 일간
    month_ground = month_pillar[1]  # 월지

    adjusted_month_ground = adjust_month_ground(day_sky, month_ground)
    return f"{day_sky}{adjusted_month_ground}"

def get_ipchun_opposite(saju_result):
    # 사주 결과에서 일간과 월지 추출
    parts = saju_result.split(',')
    day_pillar = parts[0].split()[-1]  # 일주
    month_pillar = parts[0].split()[1]  # 월주

    day_sky = day_pillar[0]  # 일간
    month_ground = month_pillar[1]  # 월지
    adjust_opposit_month_ground = adjust_month_ground(day_sky, month_ground)

    # 월지의 정반대 지지 찾기
    opposite_ground = OPPOSITE_GROUND[adjust_opposit_month_ground]

    return f"{day_sky}{opposite_ground}"

def get_solar_term(month_ground):
    for term, ground in SOLAR_TERMS.items():
        if ground == month_ground:
            return term
    return "해당 절기를 찾을 수 없습니다"

# (기존의 상수 정의와 함수들은 그대로 유지)

def load_saju_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    df['그레고리력'] = pd.to_datetime(df['그레고리력'])
    return df

def find_first_ipchun(birth_year, ipchun_var, df):
    """
    사용자의 탄생년도 이후부터 시작하여 일간월지와 일치하는 첫 번째 해(입춘점)을 찾아 반환합니다.
    
    :param birth_year: 사용자의 탄생년도
    :param ipchun_var: 일간월지 (예: "정자")
    :param df: 사주 데이터가 저장된 DataFrame
    :param direction: "순방향" 또는 "역방향"
    :return: 입춘점 정보를 (년도, 나이) 형식의 튜플로 반환
    """
    # 사용자의 탄생년도 이후 데이터만 필터링
    df_filtered = df[df['그레고리력'].dt.year >= birth_year]
    
    for _, row in df_filtered.iterrows():
        year = row['그레고리력'].year
        lunar_date = row['음력간지'].split()
        year_gan_ji = lunar_date[0]  # 년간지
        
        if year_gan_ji.startswith(ipchun_var):
            age = year - birth_year + 1  # 한국식 나이 계산
            return (year, age)
    
    # DB의 마지막 년도 확인
    last_year = df['그레고리력'].dt.year.max()
    if last_year - birth_year < 60:
        return None
    else:
        return None
    

def find_first_ipchun_revised(birth_year, ipchun_var, df):
    """
   1930년부터 시작하여 일간월지와 일치하는 첫 번째 해(입춘점)을 찾고,
    탄생년도를 고려하여 적절한 입춘점을 반환합니다.
    또한 입력된 간지(ipchun_var)와 일치하는 모든 그레고리력 연도를 출력합니다.
    
    :param birth_year: 사용자의 탄생년도
    :param ipchun_var: 일간월지 (예: "정자")
    :param df: 사주 데이터가 저장된 DataFrame
    :return: 입춘점 정보를 (년도, 나이) 형식의 튜플로 반환, 찾지 못한 경우 None
    """
   
    # 1930년부터 데이터 필터링
    df_filtered = df[df['그레고리력'].dt.year >= 1930]
    
    matching_years = []  # 일치하는 그레고리력 연도를 저장할 리스트
    
    for _, row in df_filtered.iterrows():
        year = row['그레고리력'].year
        lunar_date = row['음력간지'].split()
        year_gan_ji = lunar_date[0]  # 년간지
        
        if year_gan_ji.startswith(ipchun_var):
            matching_years.append(year)
        
        if year_gan_ji.startswith(ipchun_var):
            if year < birth_year:
                # 입춘점이 탄생년도 이전이면 60년을 더함
                adjusted_year = year + 60
                while adjusted_year < birth_year:
                    adjusted_year += 60
            else:
                adjusted_year = year
            
            age = adjusted_year - birth_year + 1  # 한국식 나이 계산
            
            # 일치하는 그레고리력 연도 출력
            print(f"\n입력된 간지 '{ipchun_var}'와 일치하는 그레고리력 연도:")
            for matching_year in matching_years:
                print(f"- {matching_year}년")
            
            return (adjusted_year, age)
    
    # 일치하는 그레고리력 연도 출력 (입춘점을 찾지 못한 경우에도)
    if matching_years:
        print(f"\n입력된 간지 '{ipchun_var}'와 일치하는 그레고리력 연도:")
        for matching_year in matching_years:
            print(f"- {matching_year}년")
    else:
        
        print(f"\n입력된 간지 '{ipchun_var}'와 일치하는 그레고리력 연도를 찾을 수 없습니다.")
    
    # 일치하는 입춘점을 찾지 못한 경우
    return None


def get_solar_term_date(df, year, term):
    """
    주어진 년도와 절기에 해당하는 날짜를 찾습니다.
    """
    df_year = df[df['그레고리력'].dt.year == year]
    for _, row in df_year.iterrows():
        if term in row['음력간지']:
            return row['그레고리력']
    return None


def find_lifecycle_start(day_sky, month_ground, birth_year):
    for year in range(birth_year, birth_year + 61):
        year_sky_index = (year - 4) % 10
        year_ground_index = (year - 4) % 12
        if SKY[year_sky_index][0] == day_sky and GROUND[year_ground_index][0] == month_ground:
            return year
    return None

def calculate_lifecycle(saju_str, birth_year):
    day_sky = saju_str[0]
    month_ground = saju_str[1]


    lifecycle_start = find_lifecycle_start(day_sky, month_ground, birth_year)
    if lifecycle_start is None:
        return []
    
   
    # 입춘점에서 탄생년도로 내려가며 가장 가까운 절기 찾기
    closest_term_year = lifecycle_start
    while closest_term_year > birth_year:
        closest_term_year -= 2.5
    closest_term_year += 2.5  # 마지막으로 빼기 전의 연도로 돌아감
   
    # 가장 가까운 절기의 인덱스 찾기
    closest_term_index = int(((closest_term_year - lifecycle_start) % 60) / 2.5) % 24
    
    lifecycle = []
    solar_terms_list = list(SOLAR_TERMS.keys())
    for i in range(24):
        term_index = (closest_term_index + i) % 24 
        year = closest_term_year + i * 2.5 
        age = int(year - birth_year + 1)
        lifecycle.append((solar_terms_list[term_index], year, max(1, age)))  # 나이가 1세 미만인 경우 1세로 표시
    
   
    return lifecycle

def create_enhanced_3d_lifecycle_chart(lifecycle_data, birth_year):
    """
    개선된 3D 생애 주기 차트를 생성합니다.
    
    :param lifecycle_data: 리스트 형태의 60주기 데이터. 각 항목은 (절기명, 년도, 나이) 튜플입니다.
    :param birth_year: 사용자의 출생 년도
    :return: plotly Figure 객체
    """
    """
    개선된 3D 생애 주기 차트를 생성합니다. 계절에 따라 색상이 다르게 표시됩니다.
    
    :param lifecycle_data: 리스트 형태의 60주기 데이터. 각 항목은 (절기명, 년도, 나이) 튜플입니다.
    :param birth_year: 사용자의 출생 년도
    :return: plotly Figure 객체
    """
    # 계절별 색상 정의
    season_colors = {
        '봄': 'green',
        '여름': 'red',
        '가을': 'gray',
        '겨울': 'black'
    }

    # 24절기를 계절에 올바르게 매핑
    season_mapping = {
        '입춘': '봄', '우수': '봄', '경칩': '봄', '춘분': '봄', '청명': '봄', '곡우': '봄',
        '입하': '여름', '소만': '여름', '망종': '여름', '하지': '여름', '소서': '여름', '대서': '여름',
        '입추': '가을', '처서': '가을', '백로': '가을', '추분': '가을', '한로': '가을', '상강': '가을',
        '입동': '겨울', '소설': '겨울', '대설': '겨울', '동지': '겨울', '소한': '겨울', '대한': '겨울'
    }

    # DNA 구조 형태의 나선 생성
    t = np.linspace(0, 10*np.pi, 1000)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (10*np.pi) * 60  # z축을 60년에 맞춰 정규화

    # 주 트레이스 (DNA 구조)
    trace1 = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color='lightblue', width=3),
        hoverinfo='none'
    )
    
    # 반대편 나선
    trace2 = go.Scatter3d(
        x=-x, y=-y, z=z,
        mode='lines',
        line=dict(color='lightblue', width=3),
        hoverinfo='none'
    )

    # 절기 포인트
    x_points, y_points, z_points, text, colors = [], [], [], [], []
    for term, year, age in lifecycle_data:
        t = (year - birth_year) / 60 * 10 * np.pi
        x_points.append(1.2 * np.cos(t))  # 반경을 1.2배로 늘림
        y_points.append(1.2 * np.sin(t))
        z_points.append(year - birth_year)  # z축을 출생년도부터의 경과 년수로 설정
        season = season_mapping.get(term, '알 수 없음')
        text.append(f"{term}<br>{year:.1f}년<br>{age}세<br>{season}")
        colors.append(season_colors[season])

    trace3 = go.Scatter3d(
        x=x_points, y=y_points, z=z_points,
        mode='markers+text',
        marker=dict(
            size=7,
            color=colors,
            opacity=0.8
        ),
        text=text,
        hoverinfo='text'
    )

    layout = go.Layout(
        title='60년 생애 주기 3D 차트 (계절별 색상)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='출생 이후 경과 년수',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=2),  # z축을 2배로 늘림
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5)  # 카메라 위치 조정
            ),
        ),
        showlegend=False,
        width=800,  # 차트 너비
        height=1000  # 차트 높이
    )

    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    return fig
