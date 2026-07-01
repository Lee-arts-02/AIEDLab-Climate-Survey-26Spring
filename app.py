import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(page_title="AIED Lab DBR Dashboard Pro", layout="wide")
st.title("🔬 AIED Lab: DBR Climate Survey Dashboard")

import os
import glob
import json
import math
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

# --- Local folder for saved CSVs and config ---
SAVE_DIR = "saved_data"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Overall trend charts: key event markers (defaults + local JSON)
EVENTS_FILE = os.path.join(SAVE_DIR, "trend_events.json")
DEFAULT_TREND_EVENTS = {"Week 5": "Spring Break", "Week 8": "WhatsApp Reminder"}

QUESTION_LABELS = {
    "Awareness_Psy_needs_1 + Awareness_Psy_needs_2": "Psychological needs paired composite: items 1 and 2",
    "Awareness_Psy_needs_3 + Awareness_Psy_needs_4": "Psychological needs paired composite: items 3 and 4",
    "Awareness_Psy_needs_5 + Awareness_Psy_needs_6": "Psychological needs paired composite: items 5 and 6",
    "External_events_info": "Which information-based activities did you participate in this week?",
    "Info_Followup": "Which information-based activities facilitated your engagement this week?",
    "External_events_inte": "Which interaction-based activities did you participate in this week?",
    "Inte_Followup": "Which interaction-based activities facilitated your engagement this week?",
    "Peer_Relational": "Which peer-related factors supported your engagement this week?",
    "Mentor_Relational": "Which mentor-related factors supported your engagement this week?",
    "Culture_Relational": "Which lab cultural factors supported your engagement this week?",
}

TRANSCRIPT_DIR = os.path.join("Clean_Data", "transcript")
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


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


def extract_week_number(week_label):
    match = re.search(r"(\d+)", str(week_label))
    return int(match.group(1)) if match else 999


def read_weekly_csv(file_path):
    for encoding in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(file_path, skiprows=[1], encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(file_path, skiprows=[1], encoding="utf-8", encoding_errors="ignore")


def split_multi_select(value):
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def count_multi_select(value):
    return len(split_multi_select(value))


def is_real_response_id(value):
    text = str(value).strip()
    return bool(text) and "ImportId" not in text and text.lower() != "nan"


def is_real_text(value):
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan" and "ImportId" not in text


def significance_stars(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def effect_size_label(r_value):
    if pd.isna(r_value):
        return ""
    magnitude = abs(r_value)
    if magnitude >= 0.5:
        return "large"
    if magnitude >= 0.3:
        return "moderate"
    if magnitude >= 0.1:
        return "small"
    return "negligible"


def normal_approx_p_from_r(r_value, n_value):
    if n_value < 4 or pd.isna(r_value) or abs(r_value) >= 1:
        return math.nan
    z_score = math.atanh(float(r_value)) * math.sqrt(n_value - 3)
    return math.erfc(abs(z_score) / math.sqrt(2))


def benjamini_hochberg(p_values):
    indexed = [(idx, p) for idx, p in enumerate(p_values) if not pd.isna(p)]
    adjusted = [math.nan] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    total = len(indexed)
    previous = 1.0
    for rank, (idx, p_value) in reversed(list(enumerate(indexed, start=1))):
        corrected = min(previous, p_value * total / rank)
        adjusted[idx] = min(corrected, 1.0)
        previous = corrected
    return adjusted


def build_association_table(df, predictors, outcome="Overall_Engagement", predictor_labels=None):
    predictor_labels = predictor_labels or {}
    rows = []
    for predictor in predictors:
        if predictor not in df.columns:
            continue
        pair_df = df[[predictor, outcome]].dropna()
        n_value = len(pair_df)
        r_value = pair_df[predictor].corr(pair_df[outcome]) if n_value >= 4 else math.nan
        p_value = normal_approx_p_from_r(r_value, n_value)
        rows.append({
            "Predictor": predictor_labels.get(predictor, predictor),
            "N": n_value,
            "Pearson r": r_value,
            "p (raw)": p_value,
            "Effect size": effect_size_label(r_value),
        })
    p_adjusted = benjamini_hochberg([row["p (raw)"] for row in rows])
    for row, adj in zip(rows, p_adjusted):
        row["FDR p"] = adj
        row["Sig."] = significance_stars(adj)
    return pd.DataFrame(rows).sort_values(["FDR p", "Predictor"], ascending=[True, True])


def make_option_engagement_table(df, option_col, label):
    if option_col not in df.columns:
        return pd.DataFrame()
    options = sorted({
        item
        for value in df[option_col].dropna()
        for item in split_multi_select(value)
        if item and "None" not in item
    })
    rows = []
    for option in options:
        selected = df[df[option_col].apply(lambda value: option in split_multi_select(value))]
        not_selected = df[~df[option_col].apply(lambda value: option in split_multi_select(value))]
        if len(selected) == 0:
            continue
        rows.append({
            "Structure Type": label,
            "Design Structure": option,
            "Selected N": int(len(selected)),
            "Selected Avg Engagement": selected["Overall_Engagement"].mean(),
            "Not Selected Avg Engagement": not_selected["Overall_Engagement"].mean() if len(not_selected) else math.nan,
            "Difference": selected["Overall_Engagement"].mean() - not_selected["Overall_Engagement"].mean() if len(not_selected) else math.nan,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Selected Avg Engagement", ascending=False)


def make_raw_option_table(df, option_col, question_label):
    if option_col not in df.columns:
        return pd.DataFrame()
    rows = []
    options = sorted({
        item
        for value in df[option_col].dropna()
        for item in split_multi_select(value)
        if item and "None" not in item
    })
    for option in options:
        selected_mask = df[option_col].apply(lambda value: option in split_multi_select(value))
        selected = df[selected_mask]
        not_selected = df[~selected_mask]
        rows.append({
            "Question / field": question_label,
            "Original selected option": option,
            "Selected N": int(len(selected)),
            "Selected %": len(selected) / len(df) if len(df) else math.nan,
            "Mean engagement when selected": selected["Overall_Engagement"].mean(),
            "Mean engagement when not selected": not_selected["Overall_Engagement"].mean() if len(not_selected) else math.nan,
            "Difference": selected["Overall_Engagement"].mean() - not_selected["Overall_Engagement"].mean() if len(not_selected) else math.nan,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Question / field", "Selected N"], ascending=[True, False])


def make_raw_likert_association_table(df, item_cols, outcome="Overall_Engagement"):
    rows = []
    for col in item_cols:
        if col not in df.columns:
            continue
        pair_df = df[[col, outcome]].dropna()
        n_value = len(pair_df)
        r_value = pair_df[col].corr(pair_df[outcome]) if n_value >= 4 else math.nan
        p_value = normal_approx_p_from_r(r_value, n_value)
        rows.append({
            "Original item": col,
            "N": n_value,
            "Pearson r": r_value,
            "p (raw)": p_value,
            "Effect size": effect_size_label(r_value),
        })
    p_adjusted = benjamini_hochberg([row["p (raw)"] for row in rows])
    for row, adj in zip(rows, p_adjusted):
        row["FDR p"] = adj
        row["Sig."] = significance_stars(adj)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["FDR p", "Original item"], ascending=[True, True])


def make_weekly_raw_option_frequency(df, option_col, question_label):
    if option_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        for option in split_multi_select(row.get(option_col)):
            if not option or "None" in option:
                continue
            rows.append({
                "Week": row.get("Week", ""),
                "Week_Number": row.get("Week_Number", 999),
                "Question / field": question_label,
                "Original selected option": option,
            })
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    return (
        raw.groupby(["Week_Number", "Week", "Question / field", "Original selected option"])
        .size()
        .reset_index(name="Selected count")
        .sort_values(["Week_Number", "Question / field", "Selected count"], ascending=[True, True, False])
        .drop(columns=["Week_Number"])
    )


def build_combined_engagement_predictor_table(df, psych_pairs, option_groups, outcome="Overall_Engagement"):
    rows = []

    for composite_name, item_cols in psych_pairs:
        available_cols = [col for col in item_cols if col in df.columns]
        if not available_cols:
            continue
        predictor = df[available_cols].mean(axis=1)
        pair_df = pd.DataFrame({"predictor": predictor, outcome: df[outcome]}).dropna()
        n_value = len(pair_df)
        r_value = pair_df["predictor"].corr(pair_df[outcome]) if n_value >= 4 else math.nan
        p_value = normal_approx_p_from_r(r_value, n_value)
        rows.append({
            "Predictor type": "Psychological needs",
            "Original field": " + ".join(available_cols),
            "Survey question": QUESTION_LABELS.get(composite_name, composite_name),
            "Original selected option": composite_name,
            "N": n_value,
            "Selected N": math.nan,
            "Selected %": math.nan,
            "Mean engagement when selected": math.nan,
            "Mean engagement when not selected": math.nan,
            "Difference": math.nan,
            "Pearson r": r_value,
            "p (raw)": p_value,
            "Effect size": effect_size_label(r_value),
        })

    for predictor_type, option_col in option_groups:
        if option_col not in df.columns:
            continue
        options = sorted({
            item
            for value in df[option_col].dropna()
            for item in split_multi_select(value)
            if item and "None" not in item
        })
        for option in options:
            selected = df[option_col].apply(lambda value: option in split_multi_select(value)).astype(int)
            pair_df = pd.DataFrame({"predictor": selected, outcome: df[outcome]}).dropna()
            if pair_df["predictor"].nunique() < 2:
                continue
            selected_df = df[selected.astype(bool)]
            not_selected_df = df[~selected.astype(bool)]
            n_value = len(pair_df)
            r_value = pair_df["predictor"].corr(pair_df[outcome]) if n_value >= 4 else math.nan
            p_value = normal_approx_p_from_r(r_value, n_value)
            rows.append({
                "Predictor type": predictor_type,
                "Original field": option_col,
                "Survey question": QUESTION_LABELS.get(option_col, option_col),
                "Original selected option": option,
                "N": n_value,
                "Selected N": int(selected.sum()),
                "Selected %": selected.mean(),
                "Mean engagement when selected": selected_df[outcome].mean(),
                "Mean engagement when not selected": not_selected_df[outcome].mean(),
                "Difference": selected_df[outcome].mean() - not_selected_df[outcome].mean(),
                "Pearson r": r_value,
                "p (raw)": p_value,
                "Effect size": effect_size_label(r_value),
            })

    if not rows:
        return pd.DataFrame()
    p_adjusted = benjamini_hochberg([row["p (raw)"] for row in rows])
    for row, adj in zip(rows, p_adjusted):
        row["FDR p"] = adj
        row["Sig."] = significance_stars(adj)
    return pd.DataFrame(rows).sort_values(["FDR p", "Predictor type", "Original field"], ascending=[True, True, True])


def fit_week_fe_cluster_model(df, selected, outcome="Weekly_Engagement_Mean"):
    model_df = pd.DataFrame({
        "outcome": df[outcome],
        "selected": selected.astype(float),
        "Week_Number": df["Week_Number"],
        "ID": df["ID"],
    }).dropna()
    if len(model_df) < 8 or model_df["selected"].nunique() < 2 or model_df["ID"].nunique() < 2:
        return math.nan, math.nan, math.nan, len(model_df), model_df["ID"].nunique()

    week_dummies = pd.get_dummies(model_df["Week_Number"].astype(str), prefix="Week", drop_first=True, dtype=float)
    x_df = pd.concat([model_df[["selected"]], week_dummies], axis=1)
    x_df = sm.add_constant(x_df, has_constant="add")
    try:
        fitted = sm.OLS(model_df["outcome"], x_df).fit(
            cov_type="cluster",
            cov_kwds={"groups": model_df["ID"]},
        )
    except Exception:
        fitted = sm.OLS(model_df["outcome"], x_df).fit()
    return (
        fitted.params.get("selected", math.nan),
        fitted.bse.get("selected", math.nan),
        fitted.pvalues.get("selected", math.nan),
        len(model_df),
        model_df["ID"].nunique(),
    )


def build_structure_regression_table(df, option_groups, outcome="Weekly_Engagement_Mean"):
    rows = []
    for predictor_type, option_col in option_groups:
        if option_col not in df.columns:
            continue
        options = sorted({
            item
            for value in df[option_col].dropna()
            for item in split_multi_select(value)
            if item and "None" not in item
        })
        for option in options:
            selected = df[option_col].apply(lambda value: option in split_multi_select(value)).astype(int)
            selected_df = df[selected.astype(bool)]
            not_selected_df = df[~selected.astype(bool)]
            coef, se, p_value, n_value, clusters = fit_week_fe_cluster_model(df, selected, outcome=outcome)
            rows.append({
                "Structure type": predictor_type,
                "Original field": option_col,
                "Survey question": QUESTION_LABELS.get(option_col, option_col),
                "Original selected option": option,
                "N": n_value,
                "Student clusters": clusters,
                "Selected N": int(selected.sum()),
                "Selected %": selected.mean(),
                "Mean engagement when selected": selected_df[outcome].mean(),
                "Mean engagement when not selected": not_selected_df[outcome].mean(),
                "Beta 1 (Structure coefficient)": coef,
                "Cluster SE": se,
                "p (raw)": p_value,
            })
    if not rows:
        return pd.DataFrame()
    p_adjusted = benjamini_hochberg([row["p (raw)"] for row in rows])
    for row, adj in zip(rows, p_adjusted):
        row["FDR p"] = adj
        row["Sig."] = significance_stars(adj)
    return pd.DataFrame(rows).sort_values(["FDR p", "Structure type", "Original field"], ascending=[True, True, True])


def build_facilitation_table(df, participation_followup_pairs, relation_cols):
    rows = []
    for structure_type, participation_col, facilitating_col in participation_followup_pairs:
        if participation_col not in df.columns or facilitating_col not in df.columns:
            continue
        options = sorted({
            item
            for value in df[participation_col].dropna()
            for item in split_multi_select(value)
            if item and "None" not in item
        })
        for option in options:
            participated = df[participation_col].apply(lambda value: option in split_multi_select(value))
            facilitated = df[facilitating_col].apply(lambda value: option in split_multi_select(value))
            participated_n = int(participated.sum())
            facilitated_n = int(facilitated.sum())
            rows.append({
                "Structure type": structure_type,
                "Participation field": participation_col,
                "Facilitation field": facilitating_col,
                "Original selected option": option,
                "Participated": participated_n,
                "Reported as facilitating": facilitated_n,
                "Facilitation rate": facilitated_n / participated_n if participated_n else math.nan,
            })

    for relation_col in relation_cols:
        if relation_col not in df.columns:
            continue
        options = sorted({
            item
            for value in df[relation_col].dropna()
            for item in split_multi_select(value)
            if item and "None" not in item
        })
        for option in options:
            support_n = int(df[relation_col].apply(lambda value: option in split_multi_select(value)).sum())
            rows.append({
                "Structure type": "Relations",
                "Participation field": "",
                "Facilitation field": relation_col,
                "Original selected option": option,
                "Participated": math.nan,
                "Reported as facilitating": support_n,
                "Facilitation rate": math.nan,
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Structure type", "Reported as facilitating"], ascending=[True, False])


def get_rq2_qualitative_findings_quotes():
    rows = [
        {"Engagement context": "High engaged moments", "Moment": "Showing work", "Quote": "I think when you have the opportunity to show your work, or share your thoughts with other colleagues, that moment for me is very engaging.", "Participant": "Shiyun"},
        {"Engagement context": "High engaged moments", "Moment": "Showing work", "Quote": "In the AI symposium... we were sharing our work to the audience. I think that moment is very engaging because you have more opportunities to show what you are doing and receive feedback from other colleagues.", "Participant": "Shiyun"},
        {"Engagement context": "High engaged moments", "Moment": "Showing work", "Quote": "What I appreciate most is the opportunity to present the work, because not every day we get the chance. It's also a way to showcase the result after some time working on our project.", "Participant": "Mai"},
        {"Engagement context": "High engaged moments", "Moment": "Showing work", "Quote": "The most recent one I can remember is the last lab-wide meeting... we did a presentation to the whole group... everyone was listening to our presentation very carefully.", "Participant": "Mengchen"},
        {"Engagement context": "High engaged moments", "Moment": "Contribution to something bigger than oneself", "Quote": "I think definitely the symposium... that was the first time when the entire lab worked together as a team.", "Participant": "Cina"},
        {"Engagement context": "High engaged moments", "Moment": "Contribution to something bigger than oneself", "Quote": "We were all hosting together, and we wanted people to feel welcomed and included... that was the moment I was like, we are a big team.", "Participant": "Cina"},
        {"Engagement context": "High engaged moments", "Moment": "Contribution to something bigger than oneself", "Quote": "The moment I was most engaged was when we were prepping for our symposium... I was doing the outreach, emailing people across Philadelphia, schools, colleges... it was so nice to feel like I'm doing good things to make our symposium bigger.", "Participant": "Lika"},
        {"Engagement context": "High engaged moments", "Moment": "Contribution to something bigger than oneself", "Quote": "We all had different tasks and we made this project as best as we could... it was the feeling of teamwork and the feeling of a successful support organized by us.", "Participant": "Peter"},
        {"Engagement context": "High engaged moments", "Moment": "Agentic contribution", "Quote": "What reflects most of my engagement in the lab is my attempt to extend the existing experiment into qualitative research, like using text analysis.", "Participant": "Ziqiao"},
        {"Engagement context": "High engaged moments", "Moment": "Agentic contribution", "Quote": "I made a code table and coded my interaction with the model. This code table was recognized by my lab mentors... I'm very proud of my engagement in this research process.", "Participant": "Ziqiao"},
        {"Engagement context": "High engaged moments", "Moment": "Agentic contribution", "Quote": "I really did feel engaged because it got me thinking. I had to really reflect about which marks to award, how we teach in general, just by looking at what the models are struggling to do.", "Participant": "Beryl"},
        {"Engagement context": "High engaged moments", "Moment": "Agentic contribution", "Quote": "I think engagement is not a problem as long as we are communicating... like, 'I am really interested in this, and I would love to spearhead writing a Substack article or doing something else.'", "Participant": "Cina"},
        {"Engagement context": "Low engaged moments", "Moment": "Unclear expectations and research ambiguity", "Quote": "At the very beginning of this semester... I wasn't very sure how this would look. I didn't have an expectation of it... it was a little confusing at the beginning, like what's expected of me.", "Participant": "Cina"},
        {"Engagement context": "Low engaged moments", "Moment": "Unclear expectations and research ambiguity", "Quote": "Sometimes the weekly task for me is kind of confusing. I don't know why I'm doing this. I don't know what they expect us to do, or what the next step is.", "Participant": "Shiyun"},
        {"Engagement context": "Low engaged moments", "Moment": "Unclear expectations and research ambiguity", "Quote": "There will be times when I don't feel like I know what I'm doing... we don't know much about it. That's why we're doing research and interaction with it.", "Participant": "Mai"},
        {"Engagement context": "Low engaged moments", "Moment": "Unclear expectations and research ambiguity", "Quote": "We are confused about what we are expected to do for the next step, how we can improve our work... nobody knows how we can assign tasks in a reasonable way.", "Participant": "Shiyun"},
        {"Engagement context": "Low engaged moments", "Moment": "Autonomy without structure can become confusion", "Quote": "Research can have a lot of freedom, and we are very creative with the way we approach research, but then it was a little confusing at the beginning.", "Participant": "Cina"},
        {"Engagement context": "Low engaged moments", "Moment": "Autonomy without structure can become confusion", "Quote": "I feel like they lack structural stuff, and that will make especially master's students more confused.", "Participant": "Shiyun"},
        {"Engagement context": "Low engaged moments", "Moment": "Autonomy without structure can become confusion", "Quote": "I feel like we need a structure, or research rigor, or somebody that gives us some scaffolds to do that.", "Participant": "Shiyun"},
        {"Engagement context": "Low engaged moments", "Moment": "Autonomy without structure can become confusion", "Quote": "A more structural process in managing our progress and checking whether we are using the correct material for research could make us even more engaged.", "Participant": "Ziqiao"},
        {"Engagement context": "Low engaged moments", "Moment": "Time conflict", "Quote": "Due to my time conflict, I was not able to attend every lab-wide meeting. So I think maybe we can have more flexible time for everyone.", "Participant": "Mengchen"},
        {"Engagement context": "Low engaged moments", "Moment": "Time conflict", "Quote": "Our last celebration also... I had a time conflict with the celebration. I was unable to attend that celebration, even though I really wanted to go.", "Participant": "Mengchen"},
        {"Engagement context": "Low engaged moments", "Moment": "Time conflict", "Quote": "Those workshops were held Wednesday noon, but that time conflicted with our group meeting, so we could not make it.", "Participant": "Shiyun"},
        {"Engagement context": "Low engaged moments", "Moment": "Time conflict", "Quote": "Time conflict is a big issue, I feel like.", "Participant": "Shiyun"},
        {"Engagement context": "Low engaged moments", "Moment": "Time conflict", "Quote": "I wish there was a clearer guide on the timing... it would have been lovely to have it in blocks.", "Participant": "Beryl"},
    ]
    return pd.DataFrame(rows)


def read_docx_paragraphs(file_path):
    try:
        with ZipFile(file_path) as docx_zip:
            xml_bytes = docx_zip.read("word/document.xml")
    except (OSError, KeyError):
        return []

    root = ET.fromstring(xml_bytes)
    paragraphs = []
    for paragraph in root.iter(WORD_NS + "p"):
        parts = [text_node.text for text_node in paragraph.iter(WORD_NS + "t") if text_node.text]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def load_interview_transcripts():
    rows = []
    transcript_files = sorted(glob.glob(os.path.join(TRANSCRIPT_DIR, "*.docx")))
    for file_path in transcript_files:
        participant = os.path.basename(file_path).replace("Transcript_", "").replace(".docx", "")
        for para_idx, paragraph in enumerate(read_docx_paragraphs(file_path), start=1):
            if len(paragraph) < 35:
                continue
            if re.match(r"^(Li|Sora)\s+\d{2}:\d{2}", paragraph):
                continue
            rows.append({
                "Participant": participant,
                "File": os.path.basename(file_path),
                "Paragraph": para_idx,
                "Text": paragraph,
            })
    return pd.DataFrame(rows)


IDENTITY_THEME_KEYWORDS = {
    "Researcher self-concept": ["researcher", "real researcher", "research process", "stereotype", "grow up"],
    "Agency / ownership": ["agency", "freedom", "opportunity", "show your work", "share your thoughts", "present", "showcase"],
    "Skill and methodological growth": ["skill", "criteria", "evaluate", "evaluation", "rigorous", "methods", "next step", "develop our research"],
    "Contribution and recognition": ["contribution", "recognized", "valued", "symposium", "presentation", "present my projects"],
    "Belonging and collaboration": ["belong", "part of", "team", "collaborative", "friendly", "support", "colleagues", "lab members"],
    "Future research engagement": ["interested", "continue", "next semester", "future", "project", "still interested"],
}


def code_interview_identity_themes(transcript_df):
    if transcript_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    coded_rows = []
    lower_text = transcript_df["Text"].str.lower()
    for theme, keywords in IDENTITY_THEME_KEYWORDS.items():
        pattern = "|".join(re.escape(keyword.lower()) for keyword in keywords)
        matched = transcript_df[lower_text.str.contains(pattern, na=False, regex=True)].copy()
        if matched.empty:
            continue
        coded_rows.append({
            "Theme": theme,
            "Matching excerpts": int(len(matched)),
            "Participants": int(matched["Participant"].nunique()),
            "Example excerpt": matched.iloc[0]["Text"],
        })
    theme_df = pd.DataFrame(coded_rows).sort_values(["Participants", "Matching excerpts"], ascending=[False, False]) if coded_rows else pd.DataFrame()

    excerpt_rows = []
    for _, row in transcript_df.iterrows():
        text_lower = row["Text"].lower()
        matched_themes = [
            theme
            for theme, keywords in IDENTITY_THEME_KEYWORDS.items()
            if any(keyword.lower() in text_lower for keyword in keywords)
        ]
        if not matched_themes:
            continue
        excerpt_rows.append({
            "Participant": row["Participant"],
            "Paragraph": row["Paragraph"],
            "Matched theme(s)": "; ".join(matched_themes),
            "Excerpt": row["Text"],
        })
    excerpt_df = pd.DataFrame(excerpt_rows)
    return theme_df, excerpt_df


def render_predictor_section(title, df):
    st.markdown(f"**{title}**")
    if df.empty:
        st.info("No rows available for this section.")
        return
    st.dataframe(
        df.style.format({
            "Selected %": "{:.1%}",
            "Mean engagement when selected": "{:.2f}",
            "Mean engagement when not selected": "{:.2f}",
            "Difference": "{:.2f}",
            "Pearson r": "{:.2f}",
            "Beta 1 (Structure coefficient)": "{:.3f}",
            "Cluster SE": "{:.3f}",
            "p (raw)": "{:.4f}",
            "FDR p": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


def collect_quotes(df, columns, keywords=None, limit=8):
    quotes = []
    keywords = [kw.lower() for kw in (keywords or [])]
    for _, row in df.iterrows():
        for col in columns:
            if col not in df.columns or not is_real_text(row.get(col)):
                continue
            text = str(row[col]).strip()
            if keywords and not any(keyword in text.lower() for keyword in keywords):
                continue
            quotes.append({
                "Week": row.get("Week", ""),
                "ID": row.get("ID", ""),
                "Source field": col,
                "Quote": text,
                "Engagement": row.get("Overall_Engagement", math.nan),
            })
    return pd.DataFrame(quotes).head(limit)


def render_quote_table(quotes_df):
    if quotes_df.empty:
        st.info("No matching open-ended/interview-style quotes were found in the current data scope.")
    else:
        st.dataframe(quotes_df, use_container_width=True, hide_index=True)


def collect_rq_qualitative_evidence(df, rq_key, columns, limit=12):
    keyword_map = {
        "RQ1": ["support", "belong", "valued", "mentor", "peer", "collaborat", "pressure", "comfortable", "engaged"],
        "RQ2": ["idea", "contribution", "question", "uncertain", "skill", "research", "meaningful", "confidence", "project", "interested"],
    }
    keywords = keyword_map.get(rq_key, [])
    rows = []
    for _, row in df.iterrows():
        for col in columns:
            if col not in df.columns or not is_real_text(row.get(col)):
                continue
            text = str(row[col]).strip()
            if keywords and not any(keyword in text.lower() for keyword in keywords):
                continue
            rows.append({
                "Week": row.get("Week", ""),
                "ID": row.get("ID", ""),
                "Original source field": col,
                "Evidence excerpt": text,
            })
    return pd.DataFrame(rows).head(limit)


def collect_interview_inventory(df, columns):
    rows = []
    for _, row in df.iterrows():
        is_exit_week = row.get("Week_Number") == 13
        for col in columns:
            if col not in df.columns or not is_real_text(row.get(col)):
                continue
            text = str(row[col]).strip()
            if not is_exit_week and "interview" not in text.lower():
                continue
            rows.append({
                "Week": row.get("Week", ""),
                "ID": row.get("ID", ""),
                "Original source field": col,
                "Content": text,
            })
    return pd.DataFrame(rows)

# Sidebar: upload new weekly CSVs
st.sidebar.header("📁 Data Center")
uploaded_files = st.sidebar.file_uploader("Upload weekly CSV files", type=["csv"], accept_multiple_files=True)
st.sidebar.caption("Use uploaded files on Streamlit Cloud; local Clean_Data is used when no files are uploaded.")

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

# --- Prefer uploaded files, then local Clean_Data, then saved_data fallback ---
DATA_DIR = "Clean_Data"
if uploaded_files:
    saved_files = [os.path.join(SAVE_DIR, file.name) for file in uploaded_files]
elif glob.glob(os.path.join(DATA_DIR, "*.csv")):
    saved_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
else:
    saved_files = glob.glob(os.path.join(SAVE_DIR, "*.csv"))

# Render the dashboard whenever at least one saved file exists
if saved_files:
    all_data = []
    for file_path in saved_files:
        # Filename (e.g. "Week 1.csv") becomes the week label
        file_name = os.path.basename(file_path)
        week_label = file_name.replace('.csv', '')
        
        df = read_weekly_csv(file_path)
        df.columns = df.columns.str.strip()
        df['Week'] = week_label
        df['Week_Number'] = extract_week_number(week_label)
        all_data.append(df)
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.sort_values(['Week_Number', 'Week'])
    full_df = full_df[full_df['ID'].apply(is_real_response_id)].copy()
    
    # --- Cleaning and derived columns ---
    needs_cols = [c for c in full_df.columns if 'Awareness_Psy_needs' in c]
    eng_cols = [c for c in full_df.columns if 'Engagement' in c and 'details' not in c]
    
    for col in needs_cols + eng_cols:
        full_df[col] = full_df[col].map(likert_mapping)

    # Core SDT / engagement dimensions
    full_df['Clear_Expectations'] = full_df[[c for c in full_df.columns if 'needs_1' in c][0]]
    full_df['Autonomy'] = full_df[[c for c in full_df.columns if 'needs_2' in c][0]]
    full_df['Skill_Growth'] = full_df[[c for c in full_df.columns if 'needs_3' in c][0]]
    full_df['Meaningful_Outcomes'] = full_df[[c for c in full_df.columns if 'needs_4' in c][0]]
    full_df['Belonging'] = full_df[[c for c in full_df.columns if 'needs_5' in c][0]]
    full_df['Contribution_Value'] = full_df[[c for c in full_df.columns if 'needs_6' in c][0]]
    full_df['Autonomy'] = full_df[['Clear_Expectations', 'Autonomy']].mean(axis=1)
    full_df['Competence'] = full_df[['Skill_Growth', 'Meaningful_Outcomes']].mean(axis=1)
    full_df['Relatedness'] = full_df[['Belonging', 'Contribution_Value']].mean(axis=1)
    full_df['Need_Satisfaction_Mean'] = full_df[['Autonomy', 'Competence', 'Relatedness']].mean(axis=1)
    
    full_df['Behavioral'] = full_df[[c for c in full_df.columns if 'Engagement_1' in c][0]]
    full_df['Cognitive'] = full_df[[c for c in full_df.columns if 'Engagement_2' in c][0]]
    full_df['Emotional'] = full_df[[c for c in full_df.columns if 'Engagement_3' in c][0]]
    full_df['Agentic'] = full_df[[c for c in full_df.columns if 'Engagement_4' in c][0]]
    full_df['Overall_Engagement'] = full_df[['Behavioral', 'Cognitive', 'Emotional', 'Agentic']].mean(axis=1)
    full_df['Weekly_Engagement_Mean'] = full_df['Overall_Engagement']

    info_col_list = [c for c in full_df.columns if 'External_events_info' in c and 'TEXT' not in c]
    follow_col_list = [c for c in full_df.columns if 'Info_Followup' in c and 'TEXT' not in c]
    interaction_col_list = [c for c in full_df.columns if 'External_events_inte' in c and 'TEXT' not in c]
    interaction_follow_col_list = [c for c in full_df.columns if 'Inte_Followup' in c and 'TEXT' not in c]
    rel_cols_all = [c for c in full_df.columns if any(x in c for x in ['Peer_Relational', 'Mentor_Relational', 'Culture_Relational']) and 'TEXT' not in c]

    full_df['Information_Structure_Count'] = full_df[info_col_list[0]].apply(count_multi_select) if info_col_list else 0
    full_df['Helpful_Information_Count'] = full_df[follow_col_list[0]].apply(count_multi_select) if follow_col_list else 0
    full_df['Interaction_Structure_Count'] = full_df[interaction_col_list[0]].apply(count_multi_select) if interaction_col_list else 0
    full_df['Helpful_Interaction_Count'] = full_df[interaction_follow_col_list[0]].apply(count_multi_select) if interaction_follow_col_list else 0
    full_df['Relationship_Support_Count'] = full_df[rel_cols_all].apply(
        lambda row: sum(count_multi_select(value) for value in row), axis=1
    ) if rel_cols_all else 0
    full_df['Identity_Signal'] = full_df[['Agentic', 'Contribution_Value', 'Meaningful_Outcomes', 'Belonging']].mean(axis=1)

    # --- Sidebar: view mode ---
    dashboard_mode = st.sidebar.radio("Dashboard version", ["Basic Dashboard", "RQ Dashboard"])
    view_mode = st.sidebar.selectbox("Select view range", ["Overall trend", "Weekly trend analysis", "Individual ID tracking"]) if dashboard_mode == "Basic Dashboard" else None

    if dashboard_mode == "RQ Dashboard":
        st.header("Research Question Dashboard")
        st.caption(
            "Quantitative analyses use Weeks 1-12. Qualitative evidence uses open-ended weekly responses and Week 13 exit-interview style entries when available."
        )

        rq_choice = st.sidebar.radio(
            "Select RQ",
            [
                "RQ1: Need satisfaction and engagement over time",
                "RQ2: Lab design structures and research engagement",
                "RQ3: Research engagement and researcher identity",
            ],
        )

        quant_df = full_df[full_df["Week_Number"].between(1, 12)].copy()
        qualitative_df = full_df.copy()
        quote_cols = ['Engagement_details', 'Ending'] + [c for c in full_df.columns if '_TEXT' in c]

        metric_cols = st.columns(4)
        metric_cols[0].metric("Quantitative responses", int(len(quant_df)))
        metric_cols[1].metric("Study IDs", int(quant_df['ID'].nunique()))
        metric_cols[2].metric("Weeks analyzed", "1-12")
        metric_cols[3].metric(
            "Open-ended entries",
            int(sum(qualitative_df[col].apply(is_real_text).sum() for col in quote_cols if col in qualitative_df.columns)),
        )

        if rq_choice.startswith("RQ1"):
            st.subheader("RQ1. How do students' psychological need satisfaction and engagement fluctuate across the semester?")
            st.caption(
                "Psychological need satisfaction is computed from SDT subscales: Autonomy = need items 1/2, Competence = items 3/4, Relatedness = items 5/6. "
                "Weekly engagement is the mean of Engagement_1 through Engagement_4."
            )

            trend_df = (
                quant_df.groupby(["Week_Number", "Week"])
                .agg(
                    Response_Count=("ID", "count"),
                    Need_Satisfaction_Mean=("Need_Satisfaction_Mean", "mean"),
                    Autonomy=("Autonomy", "mean"),
                    Competence=("Competence", "mean"),
                    Relatedness=("Relatedness", "mean"),
                    Weekly_Engagement_Mean=("Weekly_Engagement_Mean", "mean"),
                )
                .reset_index()
                .sort_values("Week_Number")
            )

            st.markdown("**Analysis 1. Descriptive trend**")
            count_fig = px.bar(
                trend_df,
                x="Week",
                y="Response_Count",
                title="Response count by week",
                labels={"Response_Count": "Survey response count"},
            )
            st.plotly_chart(count_fig, use_container_width=True)

            need_fig = px.line(
                trend_df,
                x="Week",
                y=["Need_Satisfaction_Mean", "Autonomy", "Competence", "Relatedness"],
                markers=True,
                title="Psychological need satisfaction across weeks",
            )
            need_fig.update_layout(yaxis=dict(range=[1, 4], title="Mean score"))
            st.plotly_chart(need_fig, use_container_width=True)

            engagement_fig = px.line(
                trend_df,
                x="Week",
                y="Weekly_Engagement_Mean",
                markers=True,
                title="Weekly engagement across weeks",
                labels={"Weekly_Engagement_Mean": "Weekly Engagement Mean"},
            )
            engagement_fig.update_layout(yaxis=dict(range=[1, 4], title="Mean score"))
            st.plotly_chart(engagement_fig, use_container_width=True)

            st.dataframe(
                trend_df[[
                    "Week",
                    "Response_Count",
                    "Need_Satisfaction_Mean",
                    "Autonomy",
                    "Competence",
                    "Relatedness",
                    "Weekly_Engagement_Mean",
                ]].style.format({
                    "Need_Satisfaction_Mean": "{:.2f}",
                    "Autonomy": "{:.2f}",
                    "Competence": "{:.2f}",
                    "Relatedness": "{:.2f}",
                    "Weekly_Engagement_Mean": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**Analysis 2. Need satisfaction-engagement association**")
            association_df = build_association_table(
                quant_df,
                ["Need_Satisfaction_Mean", "Autonomy", "Competence", "Relatedness"],
                outcome="Weekly_Engagement_Mean",
            )
            st.dataframe(
                association_df.style.format({
                    "Pearson r": "{:.2f}",
                    "p (raw)": "{:.4f}",
                    "FDR p": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            association_scatter = px.scatter(
                quant_df,
                x="Need_Satisfaction_Mean",
                y="Weekly_Engagement_Mean",
                color="Week",
                hover_data=["ID", "Week"],
                trendline="ols",
                title="Need Satisfaction Mean and Weekly Engagement Mean",
            )
            st.plotly_chart(association_scatter, use_container_width=True)

        elif rq_choice.startswith("RQ2"):
            st.subheader("RQ2. How do lab design structures shape students' research engagement?")
            st.caption(
                "Actual participation is modeled with OLS regressions using week fixed effects and cluster-robust standard errors by student ID. "
                "Perceived facilitation is summarized descriptively as facilitating count divided by participation count where both fields exist."
            )
            with st.expander("How to read the statistical columns", expanded=True):
                st.latex(r"Engagement_{it} = \beta_0 + \beta_1 Structure_{it} + \gamma_t + \epsilon_{it}")
                st.markdown(
                    """
                    **Model A: one structure at a time.** Each row fits a separate model because the current sample is small and many structures overlap.
                    - **Predictor coding**: each selected-choice option is converted to a binary predictor: 1 = selected, 0 = not selected.
                    - **Engagement_it**: student *i*'s Weekly Engagement Mean in week *t*.
                    - **Structure_it**: whether student *i* reported participating in that design structure in week *t*.
                    - **gamma_t**: week fixed effects.
                    - **Beta 1**: adjusted association between selecting the structure and Weekly Engagement Mean, controlling for week fixed effects.
                    - **Cluster SE**: standard error clustered by student ID.
                    - **p (raw)**: uncorrected p-value for the selected structure.
                    - **FDR p**: Benjamini-Hochberg corrected p-value across tested structures.
                    - **Significance stars**: `*` FDR p < .05, `**` FDR p < .01, `***` FDR p < .001. The dashboard treats FDR p < .05 as statistically significant.
                    - **Unit of analysis**: person-week response.
                    """
                )

            actual_participation_groups = (
                [("External events", col) for col in info_col_list + interaction_col_list]
                + [("Relations", col) for col in rel_cols_all]
            )
            regression_df = build_structure_regression_table(quant_df, actual_participation_groups)
            st.markdown("**1. Actual participation**")
            if regression_df.empty:
                st.info("No design-structure options were available for regression.")
            else:
                external_reg_df = regression_df[regression_df["Structure type"] == "External events"]
                relation_reg_df = regression_df[regression_df["Structure type"] == "Relations"]
                render_predictor_section("External events: OLS with week fixed effects", external_reg_df)
                render_predictor_section("Relations: OLS with week fixed effects", relation_reg_df)

            facilitation_pairs = []
            if info_col_list and follow_col_list:
                facilitation_pairs.append(("External events: information-based", info_col_list[0], follow_col_list[0]))
            if interaction_col_list and interaction_follow_col_list:
                facilitation_pairs.append(("External events: interaction-based", interaction_col_list[0], interaction_follow_col_list[0]))
            facilitation_df = build_facilitation_table(quant_df, facilitation_pairs, rel_cols_all)
            st.markdown("**2. Perceived facilitation**")
            if facilitation_df.empty:
                st.info("No facilitation data were available.")
            else:
                st.dataframe(
                    facilitation_df.style.format({
                        "Facilitation rate": "{:.1%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("**3. Qualitative findings: interview quotes only**")
            rq2_quotes_df = get_rq2_qualitative_findings_quotes()
            quote_context = st.selectbox(
                "Filter qualitative quotes",
                ["All"] + list(rq2_quotes_df["Engagement context"].drop_duplicates()),
            )
            display_quotes_df = rq2_quotes_df
            if quote_context != "All":
                display_quotes_df = rq2_quotes_df[rq2_quotes_df["Engagement context"] == quote_context]
            st.dataframe(display_quotes_df, use_container_width=True, hide_index=True)

        elif rq_choice.startswith("RQ3"):
            st.subheader("RQ3. How might research engagement shape students' emerging researcher identities?")
            st.caption(
                "This RQ uses the interview transcripts in Clean_Data/transcript. The table below is a keyword-assisted thematic index for review, not an automated final qualitative coding."
            )

            transcript_df = load_interview_transcripts()
            theme_df, excerpt_df = code_interview_identity_themes(transcript_df)
            if transcript_df.empty:
                st.info("No transcript text was found in Clean_Data/transcript.")
            else:
                corpus_cols = st.columns(3)
                corpus_cols[0].metric("Transcript files", int(transcript_df["File"].nunique()))
                corpus_cols[1].metric("Participants", int(transcript_df["Participant"].nunique()))
                corpus_cols[2].metric("Transcript paragraphs", int(len(transcript_df)))

                st.markdown("**Theme summary from interview transcripts**")
                if theme_df.empty:
                    st.info("No identity-related theme matches were found with the current keyword index.")
                else:
                    st.dataframe(theme_df, use_container_width=True, hide_index=True)

                st.markdown("**Identity-relevant transcript excerpts**")
                if excerpt_df.empty:
                    st.info("No identity-relevant excerpts were found with the current keyword index.")
                else:
                    theme_options = ["All"] + sorted({theme for themes in excerpt_df["Matched theme(s)"] for theme in themes.split("; ")})
                    selected_theme = st.selectbox("Filter by theme", theme_options)
                    display_excerpt_df = excerpt_df
                    if selected_theme != "All":
                        display_excerpt_df = excerpt_df[excerpt_df["Matched theme(s)"].str.contains(re.escape(selected_theme), regex=True)]
                    st.dataframe(display_excerpt_df.head(50), use_container_width=True, hide_index=True)

    elif view_mode == "Overall trend":
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
    st.info("Upload weekly CSV files from the sidebar, or add CSV files to the local Clean_Data folder.")
