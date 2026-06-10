# ============================================================
#  Call Center Sentiment & Emotion Analysis System
#  Built with Streamlit | TextBlob NLP
#  Bachelor Project — Full Working Model v2
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob

NRCLEX_AVAILABLE = False

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment & Emotion Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

[data-testid="metric-container"] {
    background: linear-gradient(135deg,#1e2a3a 0%,#243447 100%);
    border: 1px solid #2e4057;
    border-radius: 10px;
    padding: 14px 16px;
}
[data-testid="metric-container"] label {
    color:#8fa8c8 !important; font-size:0.78rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:#e8f0fe !important; font-size:1.55rem !important; font-weight:700 !important;
}

.sec-head {
    font-size:1.05rem; font-weight:700; color:#e8f0fe;
    border-left:4px solid #4e8cff; padding-left:10px;
    margin:1.2rem 0 0.7rem 0;
}

.hdiv { border-top:1px solid #2e4057; margin:1rem 0; }

.insight-card {
    background:#1a2535; border:1px solid #2e4057;
    border-radius:10px; padding:12px 16px; margin-bottom:0.6rem;
    font-size:0.85rem; color:#b0c8e8; line-height:1.5;
}
.insight-card b { color:#e8f0fe; }

[data-testid="stSidebar"] { background:#111c2b; }
[data-testid="stSidebar"] * { color:#c8daf0 !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  NLP FUNCTIONS
# ════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def get_sentiment(text: str):
    analysis   = TextBlob(str(text))
    polarity   = analysis.sentiment.polarity
    confidence = round(abs(polarity), 4)
    if polarity > 0.05:    return "Positive", confidence
    elif polarity < -0.05: return "Negative", confidence
    else:                  return "Neutral",  confidence

def get_emotion(text: str):
    text_l = str(text).lower()
    keywords = {
        "Anger":       ["angry","rude","unacceptable","furious","horrible","terrible","worst","hate"],
        "Happiness":   ["happy","excellent","great","amazing","wonderful","satisfied","good","love","thank","nice","awesome"],
        "Sadness":     ["sad","disappointed","sorry","unfortunate","unhappy","depressed","upset"],
        "Frustration": ["frustrated","annoying","annoyed","useless","waste","ridiculous","pathetic","fed up"],
        "Fear":        ["scared","worried","afraid","anxious","nervous","concerned","panic","stress"],
    }
    scores   = {e: sum(1 for kw in kws if kw in text_l) for e, kws in keywords.items()}
    dominant = max(scores, key=scores.get)
    return ("Neutral", scores) if scores[dominant] == 0 else (dominant, scores)

def safe_mean(df, col):
    return round(df[col].mean(), 1) if col in df.columns else 0


# ════════════════════════════════════════════════════════════
#  COLOUR MAPS & SHARED CHART LAYOUT
# ════════════════════════════════════════════════════════════
SENT_COLORS = {"Positive":"#2ecc71","Negative":"#e74c3c","Neutral":"#f39c12"}
EMO_COLORS  = {
    "Anger":"#e74c3c","Happiness":"#2ecc71","Sadness":"#3498db",
    "Frustration":"#e67e22","Fear":"#9b59b6","Neutral":"#95a5a6"
}
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font_color   ="#c8daf0",
    margin       =dict(t=36, b=10, l=10, r=10),
    legend       =dict(bgcolor="rgba(0,0,0,0)", font_color="#c8daf0"),
)


# ════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════
for key in ["df_raw", "df_analyzed", "text_col"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("", [
        "Upload & Analyse",
        "Dashboard",
        "Agent Performance",
        "Export Report"
    ], index=0)
    st.markdown("---")
    if st.session_state.df_analyzed is not None:
        n = len(st.session_state.df_analyzed)
        st.success(f"✅ {n:,} records analysed")
    else:
        st.info("No analysis run yet")


# ════════════════════════════════════════════════════════════
#  PAGE 1 — UPLOAD & ANALYSE
# ════════════════════════════════════════════════════════════
if page == "Upload & Analyse":
    st.title("Sentiment & Emotion Analysis System")
    st.caption("NLP-Based Analysis of Customer Service Call Transcripts")
    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Upload Dataset</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing call transcripts",
        type=["csv"],
        help="Required: one text column. Optional: agent, call_status, duration, waiting_time, time_bucket"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ **{len(df):,} rows** and **{len(df.columns)} columns** loaded successfully.")

            text_col = st.selectbox(
                "Select the column containing transcript text",
                df.columns,
                help="This column will be passed to the NLP pipeline"
            )

            nulls = df[text_col].isnull().sum()
            if nulls > 0:
                st.warning(f"⚠️ {nulls} empty rows found in '{text_col}' — these will be skipped.")

            df_clean = df.dropna(subset=[text_col]).reset_index(drop=True)
            st.session_state.df_raw   = df_clean
            st.session_state.text_col = text_col

            # Quick info metrics
            i1, i2, i3 = st.columns(3)
            i1.metric("Total Rows",  len(df_clean))
            i2.metric("Columns",     len(df_clean.columns))
            i3.metric("Text Column", text_col)

            st.markdown(
                "**Columns detected:** " +
                " · ".join([f"`{c}`" for c in df_clean.columns.tolist()])
            )

            st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-head">Run NLP Analysis</div>', unsafe_allow_html=True)

            if st.button("▶️ Run Full Analysis", type="primary", use_container_width=True):
                with st.spinner("Running NLP pipeline — please wait..."):
                    bar      = st.progress(0, text="Cleaning text...")
                    df_res   = df_clean.copy()
                    df_res["Cleaned_Text"] = df_res[text_col].apply(clean_text)

                    bar.progress(30, text="Analysing sentiment...")
                    sent = df_res[text_col].apply(get_sentiment)
                    df_res["Sentiment"]       = sent.apply(lambda x: x[0])
                    df_res["Sentiment_Score"] = sent.apply(lambda x: x[1])

                    bar.progress(65, text="Detecting emotions...")
                    emo = df_res["Cleaned_Text"].apply(get_emotion)
                    df_res["Emotion"] = emo.apply(lambda x: x[0])

                    bar.progress(100, text="Done!")
                    st.session_state.df_analyzed = df_res

                st.success("✅ Analysis complete! Navigate to **Dashboard** in the sidebar.")

                r  = st.session_state.df_analyzed
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total",      len(r))
                m2.metric("Positive",   (r["Sentiment"] == "Positive").sum())
                m3.metric("Negative",   (r["Sentiment"] == "Negative").sum())
                m4.metric("Neutral",    (r["Sentiment"] == "Neutral").sum())
                m5.metric("Top Emotion", r["Emotion"].mode()[0])

            elif st.session_state.df_analyzed is not None:
                st.info("✅ Analysis already done. Go to **Dashboard** to view results.")
                if st.button("🔄 Re-run Analysis", use_container_width=True):
                    st.session_state.df_analyzed = None
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Could not read file: {e}")

    else:
        st.markdown("""
    <div class="insight-card">
    <b>How to use this system:</b><br>
    1. Upload a CSV file with call transcript data<br>
    2. Select the column that contains the transcript text<br>
    3. Click <b>Run Full Analysis</b><br>
    4. Navigate to <b>Dashboard</b> to explore insights
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 2 — DASHBOARD (filters embedded)
# ════════════════════════════════════════════════════════════
elif page == "Dashboard":
    st.title("Analytics Dashboard")

    if st.session_state.df_analyzed is None:
        st.warning("⚠️ No results yet. Upload a dataset and run analysis first.")
        st.stop()

    df_all = st.session_state.df_analyzed
    tc     = st.session_state.text_col

    # ════════════════════════════════════════════
    # FILTER PANEL — embedded at top of dashboard
    # ════════════════════════════════════════════
    with st.expander("Filter Results", expanded=False):
        fa, fb, fc, fd = st.columns(4)

        with fa:
            sent_opts = df_all["Sentiment"].unique().tolist()
            sent_sel  = st.multiselect("Sentiment", sent_opts, default=sent_opts, key="f_sent")

        with fb:
            emo_opts = df_all["Emotion"].unique().tolist()
            emo_sel  = st.multiselect("Emotion", emo_opts, default=emo_opts, key="f_emo")

        with fc:
            if "agent" in df_all.columns:
                ag_opts = df_all["agent"].unique().tolist()
                ag_sel  = st.multiselect("Agent", ag_opts, default=ag_opts, key="f_ag")
            else:
                ag_sel = None

        with fd:
            sc_min, sc_max = st.slider(
                "Confidence Score", 0.0, 1.0, (0.0, 1.0), step=0.01, key="f_sc"
            )

    # ── Apply filters independently ──────────────────────
    df = df_all.copy()
    if sent_sel:
        df = df[df["Sentiment"].isin(sent_sel)]
    if emo_sel:
        df = df[df["Emotion"].isin(emo_sel)]
    if ag_sel is not None and "agent" in df.columns and ag_sel:
        df = df[df["agent"].isin(ag_sel)]
    df = df[(df["Sentiment_Score"] >= sc_min) & (df["Sentiment_Score"] <= sc_max)]

    total = len(df)
    if total == 0:
        st.error("No records match the current filters. Please adjust your selections.")
        st.stop()

    st.caption(f"Showing **{total:,}** of **{len(df_all):,}** records")

    # ════════════════════════════════════════════
    # ROW 1 — 8 KPI CARDS
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-head">Key Performance Indicators</div>', unsafe_allow_html=True)

    pos  = (df["Sentiment"] == "Positive").sum()
    neg  = (df["Sentiment"] == "Negative").sum()
    neu  = (df["Sentiment"] == "Neutral").sum()
    pos_pct = round(pos / total * 100, 1) if total else 0
    neg_pct = round(neg / total * 100, 1) if total else 0
    neu_pct = round(neu / total * 100, 1) if total else 0

    resolved   = (df["call_status"] == "resolved").sum()  if "call_status" in df.columns else 0
    abandoned  = (df["call_status"] == "abandoned").sum() if "call_status" in df.columns else 0
    complaints = (df["call_status"] == "complaint").sum() if "call_status" in df.columns else 0
    resolve_rt = round(resolved  / total * 100, 1) if total else 0
    abandon_rt = round(abandoned / total * 100, 1) if total else 0
    avg_dur    = safe_mean(df, "duration")
    avg_wait   = safe_mean(df, "waiting_time")
    top_emo    = df["Emotion"].mode()[0] if not df.empty else "N/A"
    top_emo_display = top_emo if len(str(top_emo)) <= 10 else str(top_emo)[:10] + ".."

    k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
    k1.metric("Total Records",  f"{total:,}")
    k2.metric("Positive %",     f"{pos_pct}%",  delta=f"{pos} calls")
    k3.metric("Negative %",     f"{neg_pct}%",  delta=f"{neg} calls", delta_color="inverse")
    k4.metric("Top Emotion",    top_emo_display)
    k5.metric("Resolution Rate",f"{resolve_rt}%")
    k6.metric("Abandoned Rate", f"{abandon_rt}%")
    k7.metric("Avg Duration",   f"{avg_dur}m")
    k8.metric("Avg Wait",       f"{avg_wait}m")

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # ROW 2 — 4 charts: sentiment donut | emotion donut | sentiment bar | emotion bar
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-head">Sentiment & Emotion Overview</div>', unsafe_allow_html=True)

    sent_counts = df["Sentiment"].value_counts().reset_index()
    sent_counts.columns = ["Sentiment", "Count"]
    emo_counts  = df["Emotion"].value_counts().reset_index()
    emo_counts.columns  = ["Emotion", "Count"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        fig = px.pie(sent_counts, values="Count", names="Sentiment",
                     color="Sentiment", color_discrete_map=SENT_COLORS,
                     title="Sentiment Split", hole=0.45)
        fig.update_traces(textinfo="percent+label", textposition="inside")
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.pie(emo_counts, values="Count", names="Emotion",
                     color="Emotion", color_discrete_map=EMO_COLORS,
                     title="Emotion Split", hole=0.45)
        fig.update_traces(textinfo="percent+label", textposition="inside")
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = px.bar(sent_counts, x="Sentiment", y="Count",
                     color="Sentiment", color_discrete_map=SENT_COLORS,
                     title="Sentiment Count", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(**CHART_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.bar(emo_counts, x="Emotion", y="Count",
                     color="Emotion", color_discrete_map=EMO_COLORS,
                     title="Emotion Count", text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(**CHART_LAYOUT, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # ROW 3 — Heatmap | Confidence histogram | 2 Gauges
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-head">Deep Analysis</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns([2, 2, 1])

    with d1:
        hm  = pd.crosstab(df["Sentiment"], df["Emotion"])
        fig = px.imshow(hm, color_continuous_scale="Blues", text_auto=True,
                        title="Sentiment × Emotion Heatmap",
                        labels=dict(x="Emotion", y="Sentiment", color="Count"))
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        fig = px.histogram(df, x="Sentiment_Score", color="Sentiment",
                           color_discrete_map=SENT_COLORS, nbins=25,
                           title="Confidence Score Distribution",
                           labels={"Sentiment_Score": "Score", "count": "Frequency"})
        fig.update_layout(**CHART_LAYOUT, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with d3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pos_pct,
            title={"text": "Satisfaction %", "font": {"color": "#c8daf0", "size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8fa8c8"},
                "bar":  {"color": "#2ecc71"},
                "steps": [
                    {"range": [0,  40], "color": "#2d1b1b"},
                    {"range": [40, 70], "color": "#2d2b1b"},
                    {"range": [70,100], "color": "#1b2d1b"}
                ],
            },
            number={"suffix": "%", "font": {"color": "#2ecc71", "size": 28}}
        ))
        fig.update_layout(**CHART_LAYOUT, height=210)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(go.Indicator(
            mode="gauge+number", value=neg_pct,
            title={"text": "Negative Risk %", "font": {"color": "#c8daf0", "size": 13}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8fa8c8"},
                "bar":  {"color": "#e74c3c"},
                "steps": [
                    {"range": [0,  30], "color": "#1b2d1b"},
                    {"range": [30, 60], "color": "#2d2b1b"},
                    {"range": [60,100], "color": "#2d1b1b"}
                ],
            },
            number={"suffix": "%", "font": {"color": "#e74c3c", "size": 28}}
        ))
        fig2.update_layout(**CHART_LAYOUT, height=210)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # ROW 4 — Call volume charts (optional columns)
    # ════════════════════════════════════════════
    has_time   = "time_bucket" in df.columns
    has_status = "call_status" in df.columns

    if has_time or has_status:
        st.markdown('<div class="sec-head">Call Volume Analysis</div>', unsafe_allow_html=True)
        cv1, cv2 = st.columns(2)

        if has_time and has_status:
            with cv1:
                tb = df["time_bucket"].value_counts().reset_index()
                tb.columns = ["Time Bucket", "Count"]
                fig = px.bar(
                    tb, x="Time Bucket", y="Count",
                    color="Count", color_continuous_scale="Blues",
                    title="Calls by Time Bucket", text="Count"
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with cv2:
                cs = df["call_status"].value_counts().reset_index()
                cs.columns = ["Status", "Count"]
                status_colors = {
                    "resolved": "#2ecc71", "complaint": "#e74c3c",
                    "abandoned": "#f39c12", "pending": "#3498db"
                }
                fig = px.pie(
                    cs, values="Count", names="Status",
                    color="Status", color_discrete_map=status_colors,
                    title="Call Status Distribution", hole=0.35
                )
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        elif has_status:
            cs = df["call_status"].value_counts().reset_index()
            cs.columns = ["Status", "Count"]
            status_colors = {
                "resolved": "#2ecc71", "complaint": "#e74c3c",
                "abandoned": "#f39c12", "pending": "#3498db"
            }

            with cv1:
                fig = px.pie(
                    cs, values="Count", names="Status",
                    color="Status", color_discrete_map=status_colors,
                    title="Call Status Distribution", hole=0.35
                )
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with cv2:
                fig = px.bar(
                    cs, x="Status", y="Count",
                    color="Status", color_discrete_map=status_colors,
                    title="Call Status Count", text="Count"
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(**CHART_LAYOUT, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        elif has_time:
            with cv1:
                tb = df["time_bucket"].value_counts().reset_index()
                tb.columns = ["Time Bucket", "Count"]
                fig = px.bar(
                    tb, x="Time Bucket", y="Count",
                    color="Count", color_continuous_scale="Blues",
                    title="Calls by Time Bucket", text="Count"
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with cv2:
                sent_tb = df.groupby(["time_bucket", "Sentiment"]).size().reset_index(name="Count")
                fig = px.bar(
                    sent_tb, x="time_bucket", y="Count", color="Sentiment",
                    color_discrete_map=SENT_COLORS, barmode="group",
                    title="Sentiment by Time Bucket", text="Count"
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # OPTIONAL DATE / WEEK / MONTH / YEAR ANALYSIS
    # Shows only if timestamp column exists
    # ════════════════════════════════════════════

    if "timestamp" in df.columns:
        st.markdown('<div class="sec-head">Date & Trend Analysis</div>', unsafe_allow_html=True)

        df_time = df.copy()
        df_time["timestamp"] = pd.to_datetime(df_time["timestamp"], errors="coerce")
        df_time = df_time.dropna(subset=["timestamp"])

        if not df_time.empty:
            df_time["Date"] = df_time["timestamp"].dt.date
            df_time["Weekday"] = df_time["timestamp"].dt.day_name()
            df_time["Month"] = df_time["timestamp"].dt.strftime("%B")
            df_time["Year"] = df_time["timestamp"].dt.year.astype(str)

            t1, t2 = st.columns(2)

            with t1:
                daily_calls = df_time.groupby("Date").size().reset_index(name="Count")
                fig = px.line(
                    daily_calls,
                    x="Date",
                    y="Count",
                    markers=True,
                    title="Daily Call Volume"
                )
                fig.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with t2:
                weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                weekday_calls = df_time["Weekday"].value_counts().reindex(weekday_order).fillna(0).reset_index()
                weekday_calls.columns = ["Weekday", "Count"]

                fig = px.bar(
                    weekday_calls,
                    x="Weekday",
                    y="Count",
                    title="Calls by Weekday",
                    text="Count"
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(**CHART_LAYOUT, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

            t3, t4 = st.columns(2)

            with t3:
                month_order = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"
                ]
                monthly_calls = df_time["Month"].value_counts().reindex(month_order).fillna(0).reset_index()
                monthly_calls.columns = ["Month", "Count"]

                fig = px.bar(
                    monthly_calls,
                    x="Month",
                    y="Count",
                    title="Calls by Month",
                    text="Count"
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(**CHART_LAYOUT, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

            with t4:
                year_calls = df_time.groupby("Year").size().reset_index(name="Count")

                fig = px.bar(
                    year_calls,
                    x="Year",
                    y="Count",
                    title="Calls by Year",
                    text="Count"
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # ROW 5 — Stacked bar + Treemap
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-head">Emotion Breakdown by Sentiment</div>', unsafe_allow_html=True)

    e1, e2 = st.columns(2)

    with e1:
        stack_df = df.groupby(["Sentiment", "Emotion"]).size().reset_index(name="Count")
        fig = px.bar(stack_df, x="Sentiment", y="Count", color="Emotion",
                     color_discrete_map=EMO_COLORS, barmode="stack",
                     title="Emotion Mix within Each Sentiment")
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with e2:
        fig = px.treemap(emo_counts, path=["Emotion"], values="Count",
                         color="Count", color_continuous_scale="Blues",
                         title="Emotion Treemap")
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # ROW 6 — Auto Insights
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-head">Auto Insights</div>', unsafe_allow_html=True)

    ins1, ins2 = st.columns(2)
    anger_count = (df["Emotion"] == "Anger").sum()
    anger_pct   = round(anger_count / total * 100, 1) if total else 0

    with ins1:
        risk_msg = (
            "🔴 High negative sentiment — consider urgent agent training review."
            if neg_pct > 35 else
            "🟡 Moderate negative sentiment — monitor closely."
            if neg_pct > 15 else
            "🟢 Low negative sentiment — customer satisfaction looks healthy."
        )
        st.markdown(f"""
        <div class="insight-card">📌 <b>Overall Sentiment:</b>
        {pos_pct}% Positive · {neg_pct}% Negative · {neu_pct}% Neutral</div>
        <div class="insight-card">⚠️ <b>Risk Assessment:</b> {risk_msg}</div>
        <div class="insight-card">🎭 <b>Dominant Emotion:</b> <b>{top_emo}</b> is the most
        frequent emotional state across all analysed transcripts.</div>
        """, unsafe_allow_html=True)

    with ins2:
        res_msg = (
            f"✅ {resolve_rt}% resolution rate — above target." if resolve_rt >= 70
            else f"⚠️ {resolve_rt}% resolution rate — below 70% target, review unresolved cases."
            if resolve_rt > 0 else "No call_status column detected."
        )
        st.markdown(f"""
        <div class="insight-card">😠 <b>Anger Detected:</b> {anger_count} calls ({anger_pct}%)
        show anger — these transcripts may need immediate follow-up.</div>
        <div class="insight-card">📈 <b>Resolution Rate:</b> {res_msg}</div>
        <div class="insight-card">⏱️ <b>Efficiency:</b>
        Average call duration <b>{avg_dur} min</b> · Average wait time <b>{avg_wait} min</b>.</div>
        """, unsafe_allow_html=True)
        
        if "agent" in df.columns:
            worst_agent = (
                df[df["Sentiment"]=="Negative"]
                .groupby("agent").size()
                .sort_values(ascending=False)
                .head(1)
            )
            if not worst_agent.empty:
                agent_name = worst_agent.index[0]
                st.markdown(f"""
                <div class="insight-card">
                👤 <b>Attention Needed:</b> <b>{agent_name}</b> has the highest number of negative calls.
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # DATA TABLE
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-head">Filtered Records</div>', unsafe_allow_html=True)
    display_cols = [tc, "Sentiment", "Sentiment_Score", "Emotion"]
    extra = [c for c in ["agent","call_status","duration","waiting_time"] if c in df.columns]
    st.dataframe(df[display_cols + extra].reset_index(drop=True), use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 3 — AGENT PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "Agent Performance":
    st.title("Agent Performance")

    if st.session_state.df_analyzed is None:
        st.warning("⚠️ Please upload and run analysis first.")
        st.stop()

    df = st.session_state.df_analyzed

    if "agent" not in df.columns:
        st.info("ℹ️ No 'agent' column found in your dataset. Add an 'agent' column to use this page.")
        st.stop()

    st.markdown('<div class="sec-head">Agent Summary Table</div>', unsafe_allow_html=True)

    agg = df.groupby("agent").agg(
        Total_Calls   =("agent",     "count"),
        Positive_Calls=("Sentiment", lambda x: (x == "Positive").sum()),
        Negative_Calls=("Sentiment", lambda x: (x == "Negative").sum()),
        Neutral_Calls =("Sentiment", lambda x: (x == "Neutral").sum()),
        Avg_Score     =("Sentiment_Score", "mean"),
    ).reset_index()

    agg["Positive_%"] = (agg["Positive_Calls"] / agg["Total_Calls"] * 100).round(1)
    agg["Negative_%"] = (agg["Negative_Calls"] / agg["Total_Calls"] * 100).round(1)
    agg["Avg_Score"]  = agg["Avg_Score"].round(3)

    if "duration" in df.columns:
        dur = df.groupby("agent")["duration"].mean().reset_index()
        dur.columns = ["agent","Avg_Duration"]
        agg = agg.merge(dur, on="agent")
        agg["Avg_Duration"] = agg["Avg_Duration"].round(2)

    st.dataframe(agg, use_container_width=True)
    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Performance Charts</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)

    with p1:
        top10 = agg.nlargest(10, "Total_Calls")
        fig   = px.bar(top10, x="agent", y="Total_Calls",
                       color="Positive_%", color_continuous_scale="RdYlGn",
                       title="Top 10 Agents — Call Volume (colour = Positive %)",
                       text="Total_Calls")
        fig.update_traces(textposition="outside")
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        worst10 = agg.nlargest(10, "Negative_%")
        fig     = px.bar(worst10, x="agent", y="Negative_%",
                         color="Negative_%", color_continuous_scale="Reds",
                         title="Agents — Highest Negative Rate (%)",
                         text="Negative_%")
        fig.update_traces(textposition="outside")
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    p3, p4 = st.columns(2)

    with p3:
        melted = (
            pd.crosstab(df["agent"], df["Emotion"])
            .reset_index()
            .melt(id_vars="agent", var_name="Emotion", value_name="Count")
        )
        fig = px.bar(melted, x="agent", y="Count", color="Emotion",
                     color_discrete_map=EMO_COLORS, barmode="stack",
                     title="Emotion Breakdown by Agent")
        fig.update_layout(**CHART_LAYOUT, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with p4:
        fig = px.scatter(agg, x="Total_Calls", y="Avg_Score",
                         size="Total_Calls", color="Positive_%",
                         color_continuous_scale="RdYlGn",
                         hover_name="agent",
                         title="Call Volume vs Avg Sentiment Score",
                         labels={"Total_Calls": "Total Calls",
                                 "Avg_Score":   "Avg Confidence Score"})
        fig.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 4 — EXPORT REPORT
# ════════════════════════════════════════════════════════════
elif page == "Export Report":
    st.title("Export Report")

    if st.session_state.df_analyzed is None:
        st.warning("⚠️ Please upload and run analysis first.")
        st.stop()

    df    = st.session_state.df_analyzed
    tc    = st.session_state.text_col
    total = len(df)
    pos   = (df["Sentiment"] == "Positive").sum()
    neg   = (df["Sentiment"] == "Negative").sum()
    neu   = (df["Sentiment"] == "Neutral").sum()
    top_emo     = df["Emotion"].mode()[0] if not df.empty else "N/A"
    resolve_rt  = (
        round((df["call_status"] == "resolved").sum() / total * 100, 1)
        if "call_status" in df.columns else "N/A"
    )

    st.markdown('<div class="sec-head">Summary Statistics</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Records", total)
    s2.metric("Positive",      f"{pos} ({round(pos/total*100,1)}%)")
    s3.metric("Negative",      f"{neg} ({round(neg/total*100,1)}%)")
    s4.metric("Top Emotion",   top_emo)
    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Sentiment Summary**")
        ss = df["Sentiment"].value_counts().reset_index()
        ss.columns = ["Sentiment","Count"]
        ss["%"] = (ss["Count"]/total*100).round(2).astype(str)+"%"
        st.table(ss)

    with t2:
        st.markdown("**Emotion Summary**")
        es = df["Emotion"].value_counts().reset_index()
        es.columns = ["Emotion","Count"]
        es["%"] = (es["Count"]/total*100).round(2).astype(str)+"%"
        st.table(es)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">⬇Download Options</div>', unsafe_allow_html=True)

    dl1, dl2 = st.columns(2)

    with dl1:
        st.download_button(
            "⬇️ Download Full Analysed Dataset (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="call_analysis_results.csv",
            mime="text/csv", use_container_width=True
        )

    with dl2:
        summary = pd.DataFrame({
            "Metric": ["Total Records","Positive","Negative","Neutral",
                       "Positive %","Negative %","Neutral %",
                       "Top Emotion","Avg Sentiment Score","Resolution Rate"],
            "Value":  [total, pos, neg, neu,
                       f"{round(pos/total*100,1)}%",
                       f"{round(neg/total*100,1)}%",
                       f"{round(neu/total*100,1)}%",
                       top_emo,
                       round(df["Sentiment_Score"].mean(), 4),
                       f"{resolve_rt}%" if isinstance(resolve_rt, float) else resolve_rt]
        })
        st.download_button(
            "⬇️ Download Summary Report (CSV)",
            data=summary.to_csv(index=False).encode("utf-8"),
            file_name="summary_report.csv",
            mime="text/csv", use_container_width=True
        )

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">Sample Records</div>', unsafe_allow_html=True)
    preview = [tc, "Sentiment", "Sentiment_Score", "Emotion"]
    extra   = [c for c in ["agent","call_status","duration"] if c in df.columns]
    st.dataframe(df[preview + extra].head(20), use_container_width=True)