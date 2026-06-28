import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AIED Lab DBR Dashboard Pro", layout="wide")
st.title("🔬 AIED Lab: DBR Climate Survey Dashboard")

import os
import glob
import json

# --- Local folder for saved CSVs and config ---
SAVE_DIR = "saved_data"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Overall trend charts: key event markers (defaults + local JSON)
EVENTS_FILE = os.path.join(SAVE_DIR, "trend_events.json")
DEFAULT_TREND_EVENTS = {"Week 5": "Spring Break", "Week 8": "WhatsApp Reminder"}


def load_trend_events():
    if os.path.isfile(EVENTS_FILE):
        try:
            with open(EVENTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_TREND_EVENTS.copy()


def save_trend_events(events_dict):
    try:
        with open(EVENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(events_dict, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def apply_trend_event_markers(fig, events, valid_weeks):
    """Draw per-week key events: faint dashed vertical lines and semi-transparent labels near the top of the score axis."""
    valid = set(valid_weeks)
    for week_key, label in (events or {}).items():
        if week_key not in valid or not str(label).strip():
            continue
        fig.add_vline(
            x=week_key,
            line_width=1,
            line_dash="dash",
            line_color="rgba(140, 140, 140, 0.75)",
            xref="x",
            layer="below",
        )
        fig.add_annotation(
            x=week_key,
            xref="x",
            y=3.92,
            yref="y",
            text=str(label),
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color="#333333"),
            align="center",
            bgcolor="rgba(255, 255, 255, 0.78)",
            bordercolor="rgba(190, 190, 190, 0.65)",
            borderwidth=1,
            borderpad=3,
        )

# Sidebar: upload new weekly CSVs
st.sidebar.header("📁 Data Center")
uploaded_files = st.sidebar.file_uploader("Upload new CSV file", type=["csv"], accept_multiple_files=True)

# Sniff encoding and re-save as UTF-8 when the user uploads files
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(SAVE_DIR, file.name)
        
        raw_bytes = file.getvalue()
        
        # Common encodings for Qualtrics-style exports
        encodings_to_try = ['utf-8', 'gbk', 'latin1', 'utf-16']
        saved_successfully = False
        
        for enc in encodings_to_try:
            try:
                decoded_text = raw_bytes.decode(enc)
                
                # utf-8-sig helps Excel open the file without mojibake
                with open(file_path, "w", encoding="utf-8-sig") as f:
                    f.write(decoded_text)
                saved_successfully = True
                break
                
            except UnicodeDecodeError:
                continue
        
        # Fallback: decode with replacement if nothing else worked
        if not saved_successfully:
            decoded_text = raw_bytes.decode('utf-8', errors='ignore')
            with open(file_path, "w", encoding="utf-8-sig") as f:
                f.write(decoded_text)

    st.sidebar.success("✅ The file has been automatically converted to UTF-8 encoding and permanently saved!")

likert_mapping = {
    "Strongly agree": 4, "Strongly Agree": 4, "Agree": 3, 
    "Disagree": 2, "Strongly disagree": 1, "Strongly Disagree": 1
}

# --- Auto-load every CSV under saved_data ---
saved_files = glob.glob(os.path.join(SAVE_DIR, "*.csv"))

# Render the dashboard whenever at least one saved file exists
if saved_files:
    all_data = []
    for file_path in saved_files:
        # Filename (e.g. "Week 1.csv") becomes the week label
        file_name = os.path.basename(file_path)
        week_label = file_name.replace('.csv', '')
        
        df = pd.read_csv(file_path, skiprows=[1])
        df.columns = df.columns.str.strip()
        df['Week'] = week_label
        all_data.append(df)
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.sort_values('Week')
    
    # --- Cleaning and derived columns ---
    needs_cols = [c for c in full_df.columns if 'Awareness_Psy_needs' in c]
    eng_cols = [c for c in full_df.columns if 'Engagement' in c and 'details' not in c]
    
    for col in needs_cols + eng_cols:
        full_df[col] = full_df[col].map(likert_mapping)

    # Core SDT / engagement dimensions
    full_df['Autonomy'] = full_df[[c for c in full_df.columns if 'needs_2' in c][0]]
    full_df['Competence'] = full_df[[c for c in full_df.columns if any(x in c for x in ['needs_1','needs_3','needs_4'])]].mean(axis=1)
    full_df['Relatedness'] = full_df[[c for c in full_df.columns if any(x in c for x in ['needs_5','needs_6'])]].mean(axis=1)
    
    full_df['Behavioral'] = full_df[[c for c in full_df.columns if 'Engagement_1' in c][0]]
    full_df['Cognitive'] = full_df[[c for c in full_df.columns if 'Engagement_2' in c][0]]
    full_df['Emotional'] = full_df[[c for c in full_df.columns if 'Engagement_3' in c][0]]
    full_df['Agentic'] = full_df[[c for c in full_df.columns if 'Engagement_4' in c][0]]
    full_df['Overall_Engagement'] = full_df[['Behavioral', 'Cognitive', 'Emotional', 'Agentic']].mean(axis=1)

    # --- Sidebar: view mode ---
    view_mode = st.sidebar.selectbox("Select view range", ["Overall trend", "Weekly trend analysis", "Individual ID tracking"])

    if view_mode == "Overall trend":
        st.header("📈 The evolution of psychological needs and engagement over the semester")
        st.caption("💡 **Coding Principle:** Strongly agree = 4, Agree = 3, Disagree = 2, Strongly disagree = 1")
        trend_df = full_df.groupby('Week')[['Behavioral', 'Cognitive', 'Emotional', 'Agentic', 'Autonomy', 'Competence', 'Relatedness']].mean().reset_index()
        
        trend_df['SDT_Avg'] = trend_df[['Autonomy', 'Competence', 'Relatedness']].mean(axis=1)
        trend_df['Engagement_Avg'] = trend_df[['Behavioral', 'Cognitive', 'Emotional', 'Agentic']].mean(axis=1)

        # --- Overall trend: charts ---
        st.header("📈 The evolution of psychological needs and engagement over the semester")
        st.caption("💡 **Coding Principle:** Strongly agree = 4, Agree = 3, Disagree = 2, Strongly disagree = 1")

        # Weekly stats: mean, min, max, and N for SDT + engagement items
        sdt_dims = ['Autonomy', 'Competence', 'Relatedness']
        eng_dims = ['Behavioral', 'Cognitive', 'Emotional', 'Agentic']
        
        show_error_bars = st.sidebar.toggle("Show Min/Max Range (Error Bars)", value=False)
        show_event_markers = st.sidebar.toggle("Show key event markers on trend charts", value=True)

        # Table used by line charts (means, range, sample size)
        summary_stats = full_df.groupby('Week').agg({
            **{col: ['mean', 'min', 'max'] for col in sdt_dims + eng_dims},
            'ID': 'count'
        }).reset_index()
        summary_stats.columns = ['Week'] + [f"{c[0]}_{c[1]}" for c in summary_stats.columns[1:-1]] + ['Response_Count']

        # Weekly composite means (for overall-average reference lines on trend charts)
        summary_stats['SDT_Avg'] = summary_stats[
            ['Autonomy_mean', 'Competence_mean', 'Relatedness_mean']
        ].mean(axis=1)
        summary_stats['Engagement_Avg'] = summary_stats[
            ['Behavioral_mean', 'Cognitive_mean', 'Emotional_mean', 'Agentic_mean']
        ].mean(axis=1)

        def create_advanced_trend_chart(
            df_stats, dimensions, title, overall_avg_col, y_label, events=None, show_event_markers=True
        ):
            fig = go.Figure()

            # Background bars: sample size on secondary y-axis
            fig.add_trace(go.Bar(
                x=df_stats['Week'],
                y=df_stats['Response_Count'],
                name="Sample Size (N)",
                marker_color='rgba(200, 200, 200, 0.2)',
                yaxis='y2'
            ))

            for dim in dimensions:
                fig.add_trace(go.Scatter(
                    x=df_stats['Week'],
                    y=df_stats[f"{dim}_mean"],
                    name=dim,
                    mode='lines+markers',
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=df_stats[f"{dim}_max"] - df_stats[f"{dim}_mean"],
                        arrayminus=df_stats[f"{dim}_mean"] - df_stats[f"{dim}_min"],
                        visible=show_error_bars,
                        thickness=1,
                        width=3
                    )
                ))

            # Overall average: last trace so it draws above dimension lines and stays visible over bars
            fig.add_trace(go.Scatter(
                x=df_stats['Week'],
                y=df_stats[overall_avg_col],
                name="🌟 Overall Average",
                mode='lines+markers',
                line=dict(color='black', width=4, dash='dash'),
                marker=dict(color='black', size=8),
                yaxis='y',
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Week",
                yaxis=dict(title=y_label, range=[1, 4]),
                yaxis2=dict(title="Sample Size (N)", overlaying='y', side='right', showgrid=False, range=[0, df_stats['Response_Count'].max() * 2]),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            if show_event_markers:
                apply_trend_event_markers(fig, events, df_stats["Week"].tolist())
            return fig

        if "trend_events" not in st.session_state:
            st.session_state.trend_events = load_trend_events()

        with st.expander("Key event annotations (edit here, saved locally)", expanded=False):
            st.caption(
                f"Markers appear on the SDT and Engagement charts below when enabled in the sidebar. "
                f"Storage file: `{EVENTS_FILE}`"
            )
            week_options = summary_stats["Week"].tolist()
            ev = st.session_state.trend_events
            for wk, lab in list(ev.items()):
                c_del, c_wk, c_lab = st.columns([1, 3, 6])
                with c_wk:
                    st.text(wk if wk in week_options else f"{wk} (not in current data — not shown on chart)")
                with c_lab:
                    st.text(lab)
                with c_del:
                    if st.button("Delete", key=f"trend_evt_del_{wk}"):
                        ev.pop(wk, None)
                        st.session_state.trend_events = ev
                        save_trend_events(ev)
                        st.rerun()
            with st.form("trend_evt_add", clear_on_submit=True):
                ac1, ac2, ac3 = st.columns([3, 5, 2])
                with ac1:
                    new_wk = st.selectbox("Week", options=week_options, key="trend_evt_week_pick")
                with ac2:
                    new_lab = st.text_input("Event label", placeholder="e.g., Midterm review")
                with ac3:
                    submitted = st.form_submit_button("Add or update")
                if submitted and new_lab.strip():
                    ev[new_wk] = new_lab.strip()
                    st.session_state.trend_events = ev
                    save_trend_events(ev)
                    st.rerun()
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Reset to built-in defaults", key="trend_evt_reset_default"):
                    st.session_state.trend_events = DEFAULT_TREND_EVENTS.copy()
                    save_trend_events(st.session_state.trend_events)
                    st.rerun()
            with b2:
                if st.button("Clear all events", key="trend_evt_clear_all"):
                    st.session_state.trend_events = {}
                    save_trend_events({})
                    st.rerun()

        events = st.session_state.trend_events

        st.plotly_chart(
            create_advanced_trend_chart(
                summary_stats,
                sdt_dims,
                "🧠 Psychological Needs (SDT) with Range & Sample Size",
                "SDT_Avg",
                "Score (1-4)",
                events=events,
                show_event_markers=show_event_markers,
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            create_advanced_trend_chart(
                summary_stats,
                eng_dims,
                "🔥 Engagement Trend with Range & Sample Size",
                "Engagement_Avg",
                "Score (1-4)",
                events=events,
                show_event_markers=show_event_markers,
            ),
            use_container_width=True,
        )

        st.divider()
        st.header("⚖️ The core design elements and environmental support summary")
        
        st.subheader("The effectiveness of design activities (Interest Rate)")
        info_col = [c for c in full_df.columns if 'External_events_info' in c and 'TEXT' not in c][0]
        follow_col = [c for c in full_df.columns if 'Info_Followup' in c and 'TEXT' not in c][0]
        
        def get_counts(col_name):
            return full_df[col_name].dropna().str.split(',').explode().str.strip().value_counts()
        
        part_counts = get_counts(info_col)
        interest_counts = get_counts(follow_col)
        
        design_stat = pd.DataFrame({'Participated': part_counts, 'Interested': interest_counts}).fillna(0)
        design_stat['Interest_Rate'] = (design_stat['Interested'] / design_stat['Participated']).fillna(0)
        design_stat = design_stat[design_stat.index != ""].sort_values('Interest_Rate', ascending=False)
        
        design_stat['Participated'] = design_stat['Participated'].astype(int)
        design_stat['Interested'] = design_stat['Interested'].astype(int)
        
        styled_df = design_stat.style.format({'Interest_Rate': '{:.0%}'}).background_gradient(cmap='Blues', subset=['Interest_Rate'])
        st.dataframe(styled_df, use_container_width=True)

        st.divider()

        st.subheader("The popularity of interpersonal support and interactive options (weekly distribution)")
        st.caption("The weekly frequency of positive interactions in all interpersonal multiple-choice questions.")
        
        rel_cols = [c for c in full_df.columns if any(x in c for x in ['Peer_Relational', 'Mentor_Relational', 'Culture_Relational']) and 'TEXT' not in c]
        all_options = []
        
        for col in rel_cols:
            exploded = full_df[['Week', col]].dropna().copy()
            exploded[col] = exploded[col].astype(str).str.split(',')
            exploded = exploded.explode(col)
            exploded[col] = exploded[col].str.strip()
            all_options.append(exploded.rename(columns={col: 'Option'}))
            
        if all_options:
            opt_df = pd.concat(all_options)
            opt_df = opt_df[(opt_df['Option'] != "") & (~opt_df['Option'].str.contains('None|did not', case=False, na=False))]
            opt_df = opt_df.reset_index(drop=True)
            
            pivot_df = pd.crosstab(opt_df['Option'], opt_df['Week'])
            pivot_df['Total'] = pivot_df.sum(axis=1)
            pivot_df = pivot_df.sort_values('Total', ascending=False)
            
            st.dataframe(pivot_df.style.background_gradient(cmap='Greens', subset=['Total']), use_container_width=True)
        else:
            st.write("No data found for interpersonal support options.")

    elif view_mode == "Individual ID tracking":
        target_id = st.sidebar.selectbox("Select Study ID", full_df['ID'].unique())
        user_df = full_df[full_df['ID'] == target_id].sort_values('Week')
        
        st.header(f"👤 Individual profile: {target_id}")
        
        col_id1, col_id2 = st.columns(2)
        with col_id1:
            st.subheader("The tracking of psychological needs and engagement scores")
            fig_user = go.Figure()

            # SDT sub-dimensions: lighter dotted traces
            sdt_cols = ['Autonomy', 'Competence', 'Relatedness']
            for col in sdt_cols:
                fig_user.add_trace(go.Scatter(x=user_df['Week'], y=user_df[col], name=f"Sub-dimension: {col}", mode='lines+markers', line=dict(width=1.5, dash='dot'), opacity=0.6))
            
            fig_user.add_trace(go.Scatter(x=user_df['Week'], y=user_df[sdt_cols].mean(axis=1), name="🌟 SDT overall mean", mode='lines+markers', line=dict(dash='dash', width=4, color='#FF6347')))

            eng_cols = ['Behavioral', 'Cognitive', 'Emotional', 'Agentic']
            for col in eng_cols:
                fig_user.add_trace(go.Scatter(x=user_df['Week'], y=user_df[col], name=f"Sub-dimension: {col}", mode='lines+markers', line=dict(width=1.5), opacity=0.6))
            
            fig_user.add_trace(go.Scatter(x=user_df['Week'], y=user_df['Overall_Engagement'], name="🌟 Engagement overall mean", mode='lines+markers', line=dict(width=4, color='#1E90FF')))

            fig_user.update_layout(
                yaxis_title="Score (1-4)", 
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_user, use_container_width=True)
        
        with col_id2:
            st.subheader("The radar chart of the current dimension")
            latest_week = user_df['Week'].iloc[-1]
            latest_data = user_df[user_df['Week'] == latest_week].iloc[0]
            
            categories = ['Autonomy', 'Competence', 'Relatedness', 'Behavioral', 'Cognitive', 'Emotional', 'Agentic']
            fig_radar = go.Figure(data=go.Scatterpolar(r=[latest_data[c] for c in categories] + [latest_data[categories[0]]], theta=categories + [categories[0]], fill='toself'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1, 4])))
            st.plotly_chart(fig_radar, use_container_width=True)

    elif view_mode == "Weekly trend analysis":
        selected_week = st.sidebar.selectbox("Select week", full_df['Week'].unique())
        week_df = full_df[full_df['Week'] == selected_week].copy()
        
        st.header(f"📅 {selected_week} Detailed depth analysis")
        
        # Weekly headline KPIs (includes response count)
        col_n, col1, col2, col3, col4 = st.columns(5)
        
        response_count = week_df['Overall_Engagement'].count() 
        
        col_n.metric("👤 The number of valid responses this week", int(response_count))
        col1.metric("This week Autonomy", round(week_df['Autonomy'].mean(), 2))
        col2.metric("This week Competence", round(week_df['Competence'].mean(), 2))
        col3.metric("This week Relatedness", round(week_df['Relatedness'].mean(), 2))
        col4.metric("Overall Engagement", round(week_df['Overall_Engagement'].mean(), 2))
        
        st.divider()

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.subheader("SDT vs Engagement radar chart")
            categories = ['Autonomy', 'Competence', 'Relatedness', 'Behavioral', 'Cognitive', 'Emotional', 'Agentic']
            values = [week_df[cat].mean() for cat in categories]
            values.append(values[0])
            radar_cats = categories + [categories[0]]
            
            fig_radar = go.Figure(data=go.Scatterpolar(r=values, theta=radar_cats, fill='toself'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1, 4])))
            st.plotly_chart(fig_radar, use_container_width=True)

        with row1_col2:
            st.subheader("Time investment and participation distribution")
            time_col_list = [c for c in week_df.columns if 'Time' in c]
            if time_col_list:
                fig_box = px.box(week_df, x=time_col_list[0], y="Overall_Engagement", points="all")
                st.plotly_chart(fig_box, use_container_width=True)

        st.divider()

        st.subheader("🚀 The effectiveness analysis of design elements this week (Design Elements Impact)")
        events_col_list = [col for col in week_df.columns if 'External_events_info' in col and 'TEXT' not in col]
        
        if events_col_list:
            events_col = events_col_list[0]
            week_df['Parsed_Events'] = week_df[events_col].dropna().astype(str).apply(lambda x: [item.strip() for item in x.split(',')])
            all_events = set([item for sublist in week_df['Parsed_Events'].dropna() for item in sublist])
            
            impact_data = []
            for event in all_events:
                if event == "" or event == "nan": continue
                participants = week_df[week_df['Parsed_Events'].apply(lambda x: event in x if isinstance(x, list) else False)]
                avg_eng = participants['Overall_Engagement'].mean()
                count = len(participants)
                impact_data.append({"Design Element": event, "Avg Engagement": avg_eng, "Participant Count": count})

            impact_df = pd.DataFrame(impact_data).sort_values(by="Avg Engagement", ascending=True)
            
            if not impact_df.empty:
                fig_impact = px.bar(
                    impact_df, 
                    x="Avg Engagement", 
                    y="Design Element", 
                    orientation='h',
                    color="Participant Count", 
                    title="The average Engagement score of participants in different activities this week",
                    color_continuous_scale="Blues"
                )
                fig_impact.update_layout(xaxis=dict(range=[1, 5]))
                st.plotly_chart(fig_impact, use_container_width=True)
            else:
                st.write("No enough data to evaluate the effectiveness of activities this week.")

        st.divider()

        st.subheader("🤝 The relationship between interpersonal support and engagement this week")
        rel_cols = [c for c in week_df.columns if any(x in c for x in ['Peer_Relational', 'Mentor_Relational', 'Culture_Relational']) and 'TEXT' not in c]
        
        for col in rel_cols:
            week_df[col+'_count'] = week_df[col].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == "" else len(str(x).split(',')))
        week_df['Relationships_Support'] = week_df[[c+'_count' for c in rel_cols]].sum(axis=1)
        
        fig_rel = px.scatter(
            week_df, 
            x='Relationships_Support', 
            y='Overall_Engagement', 
            trendline="ols", 
            hover_data=['ID'],
            labels={'Relationships_Support': 'The number of interpersonal support received this week', 'Overall_Engagement': 'Overall engagement this week'}
        )
        st.plotly_chart(fig_rel, use_container_width=True)

        st.divider()

        st.subheader("💬 The qualitative feedback this week")
        text_cols = [col for col in week_df.columns if '_TEXT' in col]
        for col in text_cols:
            responses = [str(c) for c in week_df[col].dropna() if str(c).strip() != "" and str(c).lower() != "nan"]
            if responses:
                with st.expander(f"View detailed feedback for {col}"):
                    for comment in responses:
                        st.write(f"- {comment}")
else:
    st.info("👈 Please upload your weekly CSV file from the sidebar to generate the report.")