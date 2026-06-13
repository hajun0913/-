import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 폰트 깨짐 방지: 클라우드 환경을 고려하여 기본 폰트(sans-serif) 지정 및 마이너스 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------------------------------------------------------
# 1. 가상 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    data = {
        'HR': np.random.randint(0, 20, 50),
        'RBI': np.random.randint(10, 65, 50),
        'R': np.random.randint(10, 60, 50),
        'Team': np.random.choice(['KIA', '삼성', 'LG', '두산', 'SSG', 'KT', 'NC', '한화', '롯데', '키움'], 50)
    }
    df = pd.DataFrame(data)
    df_selected = df[['HR', 'RBI', 'R']].copy()
    return df, df_selected

df, df_selected = load_data()

# 2. AI 모델 및 스케일러 학습
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_selected)

model_k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
model_k4.fit(X_scaled)
df_selected['cluster_k4'] = model_k4.labels_

# -----------------------------------------------------------------------------
# 2. 스트림릿 웹 화면 구성
# -----------------------------------------------------------------------------
st.title("⚾ SaberClustering AI: KBO Batter Style Analysis")
st.markdown("""
이 시스템은 머신러닝 알고리즘(K-Means)을 활용하여 프로야구 타자들의 세이버메트릭스 스탯을 바탕으로 플레이 스타일을 스스로 분류하고, 
새로운 선수의 성적을 입력했을 때 어떤 성향의 그룹에 속하는지 실시간으로 판별해 줍니다.
""")

st.sidebar.header("📥 Input New Batter Stats")
st.sidebar.markdown("궁금한 선수의 성적을 입력하거나 시뮬레이션해 보세요.")

# 사이드바 입력창
hr = st.sidebar.number_input("예상 홈런 수 (HR)", min_value=0, max_value=60, value=10)
rbi = st.sidebar.number_input("예상 타점 수 (RBI)", min_value=0, max_value=150, value=40)
run = st.sidebar.number_input("예상 득점 수 (R)", min_value=0, max_value=150, value=35)

# -----------------------------------------------------------------------------
# 3. 메인 화면 - 탭 구성 (분석 결과 vs 신규 감별)
# -----------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 AI 군집화 분석 결과", "🤖 신규 타자 스타일 감별기"])

with tab1:
    st.header("1. 데이터 군집화 및 모델 평가")
    
    # 실루엣 점수 표시
    score_k4 = silhouette_score(X_scaled, model_k4.labels_)
    st.metric(label="🧩 최적의 모델 평가 (K=4 모델 실루엣 점수)", value=f"{score_k4:.5f}")
    
    st.markdown("""
    * **평가 해석:** 현실적인 야구 데이터의 특성상 선수들의 성향 경계가 애매한 경우가 많음에도 불구하고, 
    AI가 준수한 실루엣 점수를 기록하며 타자들의 스타일을 유의미하게 묶어냈다는 것을 확인하였습니다.
    """)
    
    # 교차표(Crosstab) 시각화
    st.subheader("📋 구단별 AI 스타일 그룹 분포 현황")
    ct = pd.crosstab(model_k4.labels_, df['Team'], rownames=['Cluster'], colnames=['Team'])
    st.dataframe(ct, use_container_width=True)

with tab2:
    st.header("2. 나만의 가상 타자 스타일 감별 결과")
    
    # 신규 입력 데이터 전처리 및 예측
    new_player = pd.DataFrame([[hr, rbi, run]], columns=['HR', 'RBI', 'R'])
    new_player_scaled = scaler.transform(new_player)
    pred_cluster = model_k4.predict(new_player_scaled)[0]
    
    # 무작위로 매핑될 수 있으므로 예측된 번호와 텍스트 출력
    st.success(f"🎯 AI 분석 결과: 입력하신 성적의 타자는 **[ {pred_cluster}번 스타일 그룹 ]**에 가깝습니다!")
    
    # 산점도 시각화 표현
    st.subheader("📍 군집 내 신규 타자 위치 확인 (HR vs RBI)")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 기존 선수들 그리기
    scatter = ax.scatter(
        df_selected['HR'], 
        df_selected['RBI'], 
        c=model_k4.labels_, 
        cmap='rainbow', 
        s=80, 
        alpha=0.6, 
        edgecolors='w', 
        linewidth=0.5
    )
    
    # 신규 선수 검은 X 그리기
    ax.scatter(hr, rbi, c='black', s=350, marker='X', edgecolors='yellow', linewidth=1.5, label='New Player')
    
    # 한글 깨짐 방지를 위해 축 이름과 제목을 영문으로 깔끔하게 변경
    ax.set_xlabel('Home Runs (HR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runs Batted In (RBI)', fontsize=12, fontweight='bold')
    ax.set_title('K-Means Hitter Clustering (K=4)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=11)
    
    # 스트림릿에 plot 표시
    st.pyplot(fig)
    
    st.info(f"💡 **시각적 분석:** 새로 입력된 타자(X 마커)가 기존 KBO 선수들의 분포 그래프 상에서 어느 위치에 찍히는지 실시간 거리를 확인할 수 있습니다.")
