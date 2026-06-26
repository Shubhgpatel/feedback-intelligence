import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


# ── Import LangChain agent ────────────────────────────────
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from langchain_agent import ask, build_llm

load_dotenv()

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DB connection ─────────────────────────────────────────
DB_USER     = os.getenv("MYSQL_USER", "root")
DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
DB_HOST     = os.getenv("MYSQL_HOST", "127.0.0.1")
DB_NAME     = os.getenv("MYSQL_DB", "feedback_intelligence")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{MYSQL_PORT}/{DB_NAME}"

@st.cache_resource
def get_engine():
    return create_engine(DB_URL, echo=False)

@st.cache_resource
def get_llm():
    return build_llm()

engine = get_engine()

# ── Query helpers ─────────────────────────────────────────
@st.cache_data(ttl=300)
def query(sql):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 1rem; }

    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f, #1e40af);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2563eb33;
        box-shadow: 0 4px 15px rgba(37,99,235,0.2);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #60a5fa;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-card-red {
        background: linear-gradient(135deg, #450a0a, #991b1b);
        border: 1px solid #dc262633;
    }
    .kpi-value-red { color: #f87171; }

    .kpi-card-green {
        background: linear-gradient(135deg, #052e16, #166534);
        border: 1px solid #16a34a33;
    }
    .kpi-value-green { color: #4ade80; }

    .kpi-card-orange {
        background: linear-gradient(135deg, #431407, #9a3412);
        border: 1px solid #ea580c33;
    }
    .kpi-value-orange { color: #fb923c; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        padding: 8px 0;
        border-bottom: 2px solid #2563eb;
        margin-bottom: 16px;
    }

    .chat-user {
        background: #1e3a5f;
        border-radius: 12px 12px 4px 12px;
        padding: 10px 16px;
        margin: 8px 0;
        color: #e2e8f0;
        text-align: right;
    }
    .chat-agent {
        background: #1e293b;
        border-radius: 12px 12px 12px 4px;
        padding: 10px 16px;
        margin: 8px 0;
        color: #cbd5e1;
        border-left: 3px solid #2563eb;
    }
    .sql-box {
        background: #0f1923;
        border-radius: 8px;
        padding: 8px 12px;
        font-family: monospace;
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 6px;
    }

    div[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Feedback Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "💬 AI Analyst", "🔍 Review Explorer", "⚠️ Anomalies"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("🐍 Python 3.11")
    st.markdown("🗄️ MySQL")
    st.markdown("🤖 LangChain + Ollama")
    st.markdown("📊 Streamlit + Plotly")
    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# 📊 Customer Feedback Intelligence Dashboard")
    st.markdown("Real-time insights from 50,000 customer reviews")
    st.markdown("---")

    # ── KPI Row ───────────────────────────────────────────
    kpi_df = query("""
        SELECT
            COUNT(*)                                                        AS total,
            SUM(CASE WHEN sentiment_label='pos' THEN 1 ELSE 0 END)         AS pos,
            SUM(CASE WHEN sentiment_label='neg' THEN 1 ELSE 0 END)         AS neg,
            ROUND(AVG(star_rating),2)                                       AS avg_stars
        FROM reviews
    """)
    anomaly_count = query(
        "SELECT COUNT(*) AS cnt FROM daily_sentiment WHERE anomaly_flag=1"
    )["cnt"][0]

    total   = int(kpi_df["total"][0])
    pos_pct = round(kpi_df["pos"][0] * 100 / total, 1)
    neg_pct = round(kpi_df["neg"][0] * 100 / total, 1)
    avg_stars = float(kpi_df["avg_stars"][0])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value">{total:,}</p>
            <p class="kpi-label">Total Reviews</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card kpi-card-green">
            <p class="kpi-value kpi-value-green">{pos_pct}%</p>
            <p class="kpi-label">Positive</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card kpi-card-red">
            <p class="kpi-value kpi-value-red">{neg_pct}%</p>
            <p class="kpi-label">Negative</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <p class="kpi-value">⭐ {avg_stars}</p>
            <p class="kpi-label">Avg Star Rating</p>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="kpi-card kpi-card-orange">
            <p class="kpi-value kpi-value-orange">{anomaly_count}</p>
            <p class="kpi-label">Anomaly Days</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Sentiment Trend + Sentiment Pie ────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<p class="section-header">📈 Sentiment Trend Over Time</p>',
                    unsafe_allow_html=True)
        trend_df = query("""
            SELECT
                DATE_FORMAT(review_date, '%Y-%m') AS month,
                SUM(pos_count)                    AS positive,
                SUM(neg_count)                    AS negative,
                SUM(neu_count)                    AS neutral
            FROM daily_sentiment
            GROUP BY month
            ORDER BY month
        """)
        if not trend_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df["month"], y=trend_df["positive"],
                name="Positive", line=dict(color="#4ade80", width=2),
                fill="tozeroy", fillcolor="rgba(74,222,128,0.1)"
            ))
            fig.add_trace(go.Scatter(
                x=trend_df["month"], y=trend_df["negative"],
                name="Negative", line=dict(color="#f87171", width=2),
                fill="tozeroy", fillcolor="rgba(248,113,113,0.1)"
            ))
            fig.add_trace(go.Scatter(
                x=trend_df["month"], y=trend_df["neutral"],
                name="Neutral", line=dict(color="#94a3b8", width=1.5),
            ))
            fig.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#e2e8f0", height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">🥧 Sentiment Split</p>',
                    unsafe_allow_html=True)
        pie_df = query("""
            SELECT sentiment_label, COUNT(*) as cnt
            FROM reviews GROUP BY sentiment_label
        """)
        if not pie_df.empty:
            fig2 = px.pie(
                pie_df, values="cnt", names="sentiment_label",
                color="sentiment_label",
                color_discrete_map={"pos":"#4ade80","neg":"#f87171","neu":"#94a3b8"},
                hole=0.5,
            )
            fig2.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#e2e8f0", height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Row 3: Top Topics + Star Distribution ─────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<p class="section-header">🏷️ Top Topics by Review Volume</p>',
                    unsafe_allow_html=True)
        topic_df = query("""
            SELECT t.topic_label, t.top_keywords, COUNT(r.review_id) AS review_count
            FROM reviews r JOIN topics t ON r.topic_id = t.topic_id
            GROUP BY t.topic_id, t.topic_label, t.top_keywords
            ORDER BY review_count DESC LIMIT 10
        """)
        if not topic_df.empty:
            fig3 = px.bar(
                topic_df, x="review_count", y="topic_label",
                orientation="h", color="review_count",
                color_continuous_scale="Blues",
                hover_data=["top_keywords"],
            )
            fig3.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#e2e8f0", height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                xaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<p class="section-header">⭐ Star Rating Distribution</p>',
                    unsafe_allow_html=True)
        star_df = query("""
            SELECT star_rating, COUNT(*) as cnt
            FROM reviews
            GROUP BY star_rating ORDER BY star_rating
        """)
        if not star_df.empty:
            colors = ["#f87171","#fb923c","#fbbf24","#a3e635","#4ade80"]
            fig4 = px.bar(
                star_df, x="star_rating", y="cnt",
                color="star_rating",
                color_discrete_sequence=colors,
                labels={"star_rating": "Stars", "cnt": "Reviews"},
            )
            fig4.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#e2e8f0", height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                xaxis=dict(gridcolor="#1e293b", tickmode="linear"),
                yaxis=dict(gridcolor="#1e293b"),
            )
            st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE 2 — AI ANALYST CHATBOT
# ════════════════════════════════════════════════════════
elif page == "💬 AI Analyst":
    st.markdown("# 💬 AI Analyst")
    st.markdown("Ask anything about your customer feedback data in plain English.")
    st.markdown("---")

    # Example questions
    st.markdown("**💡 Try asking:**")
    ex_cols = st.columns(3)
    examples = [
        "What percentage of reviews are positive?",
        "Which product has the most reviews?",
        "What are the top 5 topics?",
        "How many anomalies were detected?",
        "What is the average star rating?",
        "Show me 3 negative reviews about coffee",
    ]
    for i, ex in enumerate(examples):
        with ex_cols[i % 3]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state.example_query = ex

    st.markdown("---")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat input
    user_input = st.chat_input("Ask a question about your data ...")

    # Handle example button clicks
    if "example_query" in st.session_state:
        user_input = st.session_state.example_query
        del st.session_state.example_query

    if user_input:
        with st.spinner("🤖 Thinking ..."):
            try:
                llm = get_llm()
                sql, results, answer = ask(user_input, llm, verbose=False)
                st.session_state.chat_history.append({
                    "question": user_input,
                    "sql":      sql,
                    "results":  results[:500],
                    "answer":   answer,
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "question": user_input,
                    "sql":      "",
                    "results":  "",
                    "answer":   f"Error: {str(e)}",
                })

    # Display chat history
    for msg in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            st.write(msg["answer"])
            if msg["sql"]:
                with st.expander("🔍 View SQL query"):
                    st.code(msg["sql"], language="sql")
            if msg["results"]:
                with st.expander("📋 View raw results"):
                    st.text(msg["results"])

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


# ════════════════════════════════════════════════════════
# PAGE 3 — REVIEW EXPLORER
# ════════════════════════════════════════════════════════
elif page == "🔍 Review Explorer":
    st.markdown("# 🔍 Review Explorer")
    st.markdown("Browse, filter and search individual reviews.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        sentiment_filter = st.selectbox(
            "Sentiment", ["All", "pos", "neu", "neg"]
        )
    with col2:
        star_filter = st.selectbox(
            "Star Rating", ["All", "1", "2", "3", "4", "5"]
        )
    with col3:
        limit = st.selectbox("Show", [50, 100, 200, 500], index=0)

    search = st.text_input("🔎 Search in review text", placeholder="e.g. coffee, chocolate, dog...")

    # Build query
    conditions = []
    if sentiment_filter != "All":
        conditions.append(f"r.sentiment_label = '{sentiment_filter}'")
    if star_filter != "All":
        conditions.append(f"r.star_rating = {star_filter}")
    if search:
        conditions.append(f"r.review_text LIKE '%{search}%'")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    reviews_df = query(f"""
        SELECT
            r.review_id,
            r.product_id,
            r.star_rating,
            r.sentiment_label,
            ROUND(p.confidence, 3)  AS confidence,
            r.summary,
            LEFT(r.review_text, 200) AS review_preview,
            DATE(r.review_time)     AS review_date
        FROM reviews r
        LEFT JOIN predictions p ON r.review_id = p.review_id
        {where}
        ORDER BY r.review_time DESC
        LIMIT {limit}
    """)

    st.markdown(f"**Showing {len(reviews_df):,} reviews**")

    # Colour sentiment badges
    def badge(label):
        colors = {"pos": "#16a34a", "neg": "#dc2626", "neu": "#64748b"}
        c = colors.get(label, "#64748b")
        return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:999px;font-size:0.75rem">{label}</span>'

    for _, row in reviews_df.iterrows():
        with st.container():
            c1, c2, c3, c4 = st.columns([1, 1, 1, 7])
            with c1:
                st.markdown(f"⭐ {int(row['star_rating'])}")
            with c2:
                st.markdown(badge(row["sentiment_label"]), unsafe_allow_html=True)
            with c3:
                st.markdown(f"`{row['confidence']}`")
            with c4:
                st.markdown(f"**{row['summary']}**" if pd.notna(row['summary']) else "")
                st.caption(str(row["review_preview"]))
        st.divider()


# ════════════════════════════════════════════════════════
# PAGE 4 — ANOMALIES
# ════════════════════════════════════════════════════════
elif page == "⚠️ Anomalies":
    st.markdown("# ⚠️ Anomaly Detection")
    st.markdown("Days flagged as statistically unusual by IsolationForest.")
    st.markdown("---")

    anomaly_df = query("""
        SELECT review_date, total_reviews, pos_count, neg_count,
               neu_count, avg_star_rating, neg_ratio, anomaly_score
        FROM daily_sentiment
        ORDER BY review_date
    """)

    if anomaly_df.empty:
        st.warning("No daily sentiment data found. Run anomaly.py first.")
    else:
        # Timeline chart
        st.markdown('<p class="section-header">📅 Negative Ratio Timeline</p>',
                    unsafe_allow_html=True)

        normal_df  = anomaly_df[anomaly_df["anomaly_score"] > -0.5]
        flagged_df = anomaly_df[anomaly_df["anomaly_score"] <= -0.5]

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=normal_df["review_date"], y=normal_df["neg_ratio"],
            mode="markers", name="Normal",
            marker=dict(color="#4ade80", size=4, opacity=0.6),
        ))
        fig5.add_trace(go.Scatter(
            x=flagged_df["review_date"], y=flagged_df["neg_ratio"],
            mode="markers", name="Anomaly",
            marker=dict(color="#f87171", size=8, symbol="x"),
        ))
        fig5.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#e2e8f0", height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", title="Negative Ratio (%)"),
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Days", f"{len(anomaly_df):,}")
        with col2:
            flagged_count = len(flagged_df)
            st.metric("Anomalous Days", f"{flagged_count}", delta=f"{round(flagged_count*100/len(anomaly_df),1)}%")
        with col3:
            worst = anomaly_df.loc[anomaly_df["anomaly_score"].idxmin()]
            st.metric("Worst Day", str(worst["review_date"]))

        # Flagged days table
        st.markdown('<p class="section-header">🚨 Top Anomalous Days</p>',
                    unsafe_allow_html=True)
        top_anomalies = anomaly_df.nsmallest(20, "anomaly_score")[[
            "review_date", "total_reviews", "neg_ratio",
            "avg_star_rating", "anomaly_score"
        ]]
        st.dataframe(
            top_anomalies.style.background_gradient(
                subset=["neg_ratio"], cmap="RdYlGn_r"
            ),
            use_container_width=True,
            hide_index=True,
        )