import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from lifecycle_functions_v1 import find_first_ipchun_revised, get_ipchun_opposite, get_ipchun_same, get_saju, get_solar_term
from lifecycle_functions_v1 import calculate_lifecycle, create_enhanced_3d_lifecycle_chart
from lifecycle_functions_v1 import load_saju_data
from modified_heikinashi_fibonacci_functions import MRHATradingSystem, preprocess_codes, check_buy_signal
from dotenv import load_dotenv
import os

import streamlit as st
from streamlit_option_menu import option_menu
import base64


# API KEY 정보 로드
load_dotenv()

# 전역 변수로 사주 데이터 로드 (성능 최적화)
SAJU_DATA = load_saju_data('saju_cal.csv')

def format_lifecycle(lifecycle_data):
    formatted = ""
    for i, (term, year, age) in enumerate(lifecycle_data):
        if i % 4 == 0 and i != 0:
            formatted += "\n"
        formatted += f"{term}({year:.1f}년, {age}세) | "
    return formatted.strip()

def parse_date(input_str):
    if len(input_str) == 12 and input_str.isdigit():
        return input_str
    return None

def create_chain(model_choice):
    prompt = load_prompt("saju_prompt_general.yaml", encoding="utf-8")
    
    if model_choice == "OpenAI GPT":
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=3000)
    else:  # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def run_fibonacci_analysis(file_path, market_type):
    end_date = datetime.now()
    codes = preprocess_codes(file_path, market_type)
    
    st.write(f"총 {market_type} 개수: {len(codes)}")
    
    buy_signal_codes = []
    progress_bar = st.progress(0)
    
    for i, code in enumerate(codes):
        if check_buy_signal(code, end_date):
            buy_signal_codes.append(code)
        progress_bar.progress((i + 1) / len(codes))
    
    st.write(f"\n현재 매수 신호가 있는 {market_type} 개수: {len(buy_signal_codes)}")
    
    if buy_signal_codes:
        for code in buy_signal_codes:
            st.write(f"매수 신호 {market_type}: {code}")
            
        for code in buy_signal_codes:
            st.write(f"\n{code}에 대한 상세 분석:")
            analyze_single_code(code, end_date)
    else:
        st.write(f"현재 매수 신호가 있는 {market_type}가 없습니다.")

def analyze_single_code(code, end_date):
    try:
        trading_system = MRHATradingSystem(code, end_date - timedelta(days=365), end_date)
        trading_system.run_analysis()
        
        results = trading_system.get_results()
        
        if "error" in results:
            st.error(results["error"])
            return
        st.write(f"총 수익률: {results['Total Return']:.2%}")
        st.write(f"연간 수익률: {results['Annualized Return']:.2%}")
        st.write(f"샤프 비율: {results['Sharpe Ratio']:.2f}")
        st.write(f"최대 낙폭: {results['Max Drawdown']:.2%}")
        st.write(f"총 거래 횟수: {results['Total Trades']}")
        
        fig = trading_system.plot_results()
        st.plotly_chart(fig)
    except ValueError as e:
        st.error(f"오류: {str(e)}")
    except Exception as e:
        st.error(f"예기치 못한 오류: 종목코드를 확인해서 다시 입력해 주세요: {str(e)}")



def saju_analysis():
    st.header("AI LIFE CYCLE 길잡이 💬")

    with st.sidebar:
        if st.button("대화 초기화"):
            reset_session()
            st.rerun()

        st.session_state.model_choice = st.selectbox(
            "AI 모델을 선택해 주세요", ("OpenAI GPT", "Google Gemini"), index=0
        )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False

    if not st.session_state.analyzed:
        gender = st.radio("성별을 선택하세요:", ("남성", "여성"))
        birth_input = st.text_input("생년월일시를 입력하세요 (YYYYMMDDHHMMM 형식):")

        if st.button("분석 시작"):
            if gender and birth_input:
                parsed_date = parse_date(birth_input)
                if parsed_date:
                    try:
                        birth_year = int(parsed_date[:4])
                        current_year = datetime.now().year

                        saju_result = get_saju(parsed_date, SAJU_DATA)
                        st.write(f"당신의 사주팔자: {saju_result}")

                        ipchun_same = get_ipchun_same(saju_result)
                        ipchun_opposite = get_ipchun_opposite(saju_result)
                        month_ground = saju_result.split(',')[0].split()[1][1]
                        solar_term = get_solar_term(month_ground)

                        analysis_result = f"""
                        입춘점 정보:
                        1. 일간월지를 그대로 사용한 입춘점: {ipchun_same}년
                           해당 절기: {solar_term}
                        2. 월지의 정반대 지지를 사용한 입춘점: {ipchun_opposite}년
                           해당 절기: {get_solar_term(ipchun_opposite[1])}
                        """

                        first_ipchun_same = find_first_ipchun_revised(birth_year, ipchun_same, SAJU_DATA)
                        first_ipchun_opposite = find_first_ipchun_revised(birth_year, ipchun_opposite, SAJU_DATA)

                        if first_ipchun_same:
                            analysis_result += f"\n당신의 생애 첫 순방향 입춘점: 입춘({first_ipchun_same[0]}, {first_ipchun_same[1]}세)"
                        if first_ipchun_opposite:
                            analysis_result += f"\n당신의 생애 첫 역방향 입춘점: 입춘({first_ipchun_opposite[0]}, {first_ipchun_opposite[1]}세)"

                        lifecycle_forward = calculate_lifecycle(ipchun_same, birth_year)
                        lifecycle_backward = calculate_lifecycle(ipchun_opposite, birth_year)

                        if lifecycle_forward:
                            analysis_result += "\n\n순방향 입춘점으로 계산한 60년 생애주기:\n"
                            analysis_result += format_lifecycle(lifecycle_forward)
                        if lifecycle_backward:
                            analysis_result += "\n\n역방향 입춘점으로 계산한 60년 생애주기:\n"
                            analysis_result += format_lifecycle(lifecycle_backward)

                        st.write(analysis_result)

                        st.subheader("생애주기 3D 차트")
                        
                        st.subheader("순방향 생애주기 3D 차트")
                        fig_forward = create_enhanced_3d_lifecycle_chart(lifecycle_forward, birth_year)
                        st.plotly_chart(fig_forward, use_container_width=True)

                        st.subheader("역방향 생애주기 3D 차트")
                        fig_backward = create_enhanced_3d_lifecycle_chart(lifecycle_backward, birth_year)
                        st.plotly_chart(fig_backward, use_container_width=True)

                        chain = create_chain(st.session_state.model_choice)
                        
                        lifecycle_str = format_lifecycle(lifecycle_backward)

                        initial_response = chain.invoke({
                            "saju": saju_result,
                            "lifecycle": lifecycle_str,
                            "birth_year": str(birth_year),
                            "current_year": str(current_year),
                            "gender": gender,
                            "question": "전체적인 사주 해석과 lifecycle 의 24절기를 제공한 인생의 24절기를 참고해서 60년 인생의 생애주기를 자세하게 분석해주세요."
                        })

                        st.subheader("AI의 사주 해석 및 조언")
                        st.write(initial_response)

                        st.session_state.analyzed = True
                        st.session_state.saju_result = saju_result
                        st.session_state.lifecycle_str = lifecycle_str
                        st.session_state.birth_year = birth_year
                        st.session_state.gender = gender
                        st.session_state.messages.append({"role": "assistant", "content": initial_response})

                    except Exception as e:
                        st.error(f"오류 발생: {str(e)}")
                else:
                    st.error("올바른 형식으로 생년월일시를 입력해주세요.")
            else:
                st.error("성별과 생년월일시를 모두 입력해주세요.")

    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.analyzed:
        if prompt := st.chat_input("질문을 입력하세요:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            full_prompt = f"""
            당신은 30년 경력의 사주팔자 통변 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요:

            사용자의 사주팔자: {st.session_state.saju_result}
            생년: {st.session_state.birth_year}
            성별: {st.session_state.gender}
            
            사용자 질문: {prompt}
            
            답변 시 다음 사항을 고려해주세요:
            1. 사주팔자의 음양오행 균형을 고려하여 해석해주세요.
            2. 사용자의 성별과 나이에 맞는 사주팔자를 분석해주세요.
            3. 현대적 맥락에서 실용적인 입장에서 사주팔자를 분석해 주세요.
            4. 답변은 친절하고 이해하기 쉬운 언어로 제공해주세요.
            5. 필요하다면 생애주기 정보를 참고하여 답변할 수 있습니다.

            사주와 운세는 절대적인 것이 아니라 참고사항임을 언급하고, 개인의 노력과 선택이 중요함을 강조해주세요.
            """
            
            if st.session_state.model_choice == "OpenAI GPT":
                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0,max_tokens=3000)
            else:  # Google Gemini
                llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = llm.predict(full_prompt)
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

def fibonacci_analysis():
    st.header("Modified Ying Yang Fibonacci Signal")
    
    analysis_type = st.radio("분석 유형을 선택하세요:", ("ETF/KOSPI 리스트", "사용자 지정 코드"))
    
    if analysis_type == "ETF/KOSPI 리스트":
        market_type = st.radio("분석할 시장을 선택하세요:", ("ETF", "KOSPI"))
        
        if market_type == "ETF":
            file_path = "korea_etfs.csv"
        else:
            file_path = "kospi200_equity.csv"
        
        if st.button(f"{market_type} 분석 시작"):
            run_fibonacci_analysis(file_path, market_type)
    
    else:  # 사용자 지정 코드
        user_code = st.text_input("분석할 종목 코드를 입력하세요 (예: 005930.KS):")
        if st.button("사용자 지정 코드 분석 시작"):
            if user_code:
                st.write(f"{user_code}에 대한 상세 분석:")
                user_code = user_code+".KS" if not user_code.endswith(".KS") else user_code
                analyze_single_code(user_code, datetime.now())
            else:
                st.error("종목 코드를 입력해주세요.")


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="RhythmSphere Cycle Analysis", layout="centered")

    # 배경 이미지 추가 (이미지 파일이 있다고 가정)
    add_bg_from_local('background.png')  

    # 커스텀 CSS
    st.markdown("""
    <style>
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
    }
    .subtitle {
        font-size: 25px;
        color: #4682B4;
        text-align: center;
        margin-bottom: 50px;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">RhythmSphere Cycle Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Life Cycle 분석과 Fibonacci Cycle 분석을 위한 통합 플랫폼</p>', unsafe_allow_html=True)

    # 사이드바에 메뉴 추가
    with st.sidebar:
        choice = option_menu("메인 메뉴", ["HOME", "Life Cycle 분석", "Fibonacci Cycle 분석"],
                             icons=['house', 'calendar', 'graph-up'],
                             menu_icon="cast", default_index=0)

    if choice == "HOME":
        st.write("## 환영합니다!")
        st.write("이 애플리케이션은 60년 생애주기 분석과 주식시장 Fibonacci Cycle 분석을 제공합니다.")
        st.write("왼쪽 사이드바에서 원하는 분석을 선택해주세요.")
    

        col1, col2 = st.columns(2)
        with col1:
            st.info("### 생애주기분석\n\n사주팔자를 바탕으로 개인의 60년 생애주기를 분석합니다.")
        with col2:
            st.info("### Fibonacci Cycle 분석\n\nFibonacci 수열을 활용한 주식 시장 분석을 제공합니다.")

        st.markdown('<div class="footer"> © 2024 RhythmSphere Inc. All rights reserved.</div>', unsafe_allow_html=True)

    elif choice == "Life Cycle 분석":
        saju_analysis()

    elif choice == "Fibonacci Cycle 분석":
        fibonacci_analysis()

if __name__ == "__main__":
    main()
