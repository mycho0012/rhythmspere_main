
import streamlit as st

def main():
    st.set_page_config(page_title="RhythmSphere Cycle Analysis", layout="centered")

    st.markdown("""
    <style>
    .big-font {
        font-size:40px !important;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
    }
    .subtitle {
        font-size:20px;
        color: #4682B4;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .description {
        font-size: 16px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">RhythmSphere Cycle Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">원하시는 분석을 선택해주세요</p>', unsafe_allow_html=True)
 
    life_cycle_url = "https://sajulifecycle-ulbv3fne2qpckmssu5kxmx.streamlit.app"
    fibonacci_cycle_url = "https://mhacycle-2ruhnwcjt7vyqk65sil6bs.streamlit.app"


    col1, col2 = st.columns(2)
# Brief descriptions
    st.markdown("---")
    with col1:
        st.button("## Life Cycle Analysis", key="life_cycle", on_click=lambda: st.markdown(f'<meta http-equiv="refresh" content="0;url={life_cycle_url}">', unsafe_allow_html=True))
        st.markdown('<p class="description">계절의 순환주기인 24절기를 적용하여 인생의 60년주기를 분석.</p>', unsafe_allow_html=True)

    with col2:
        st.button("## Fibonacci Cycle Analysis", key="fibonacci", on_click=lambda: st.markdown(f'<meta http-equiv="refresh" content="0;url={fibonacci_cycle_url}">', unsafe_allow_html=True))
        st.markdown('<p class="description">Fibonacci 수열을 활용한 주식 시장의 주기성을 분석.</p>', unsafe_allow_html=True)
 
    st.markdown('<div class="footer">© 2024 RhythmSphere Cycle Analysis. All rights reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()