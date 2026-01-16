import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1) Setup Constants & CSS
# -----------------------------
PRIORITIZE_PATTERNS = [
    r"\bmfa\b", r"\botp\b", r"\bverification\b", r"\bverify\b",
    r"\bsecurity code\b", r"\blogin code\b", r"\b2fa\b",
    r"\bpassword reset\b", r"\breset your password\b", r"\bone[- ]time\b",
    r"\bconfirm your email\b", r"\bverification code\b",
    r"\baccount verification\b", r"\bsign[- ]in\b", r"\blog[- ]in\b"
]

SLOW_PATTERNS = [
    r"\bunsubscribe\b", r"\bpromo\b", r"\bpromotion\b", r"\bdiscount\b",
    r"\bsale\b", r"\bdeal\b", r"\boffer\b", r"\bmarketing\b",
    r"\bnewsletter\b", r"\bblack friday\b", r"\bcyber monday\b",
    r"\bcoupon\b"
]

examples = [
    ["security@paypal.com", "Verification code", "123456 is your code. Expires in 10 min"],
    ["news@shop.com", "Weekend Sale!", "50% off everything. Unsubscribe here."],
    ["team@company.com", "Project Alpha Update", "Hi team, please find the meeting notes attached from this morning."]
]

CUSTOM_CSS = """
.gradio-container button[role="tab"]{
  color:#166534 !important;
  font-weight:700 !important;
  opacity:.65 !important;
  border-bottom: 3px solid transparent !important;
}
.gradio-container button[role="tab"][aria-selected="true"]{
  color:#16a34a !important;
  opacity:1 !important;
  border-bottom: 3px solid #22c55e !important;
}
#submit-btn{
  background:#22c55e !important;
  color:#fff !important;
  font-weight:800 !important;
  border:none !important;
}
#submit-btn:hover{ background:#16a34a !important; }
h2, h3 { color:#166534 !important; }
"""

# -----------------------------
# 2) Helper Functions
# -----------------------------
FROM_RE = re.compile(r"(?im)^from\s*:\s*(.*)$")
SUBJECT_RE = re.compile(r"(?im)^subject\s*:\s*(.*)$")
HEADER_RE = re.compile(
    r"(?im)^(from|to|subject|cc|bcc|date|reply-to|sent|received|message-id|mime-version|content-type)\s*:\s.*$"
)

def parse_email_fields(raw_email: str):
    if not isinstance(raw_email, str):
        return "", "", ""
    from_match = FROM_RE.search(raw_email)
    subj_match = SUBJECT_RE.search(raw_email)
    from_val = from_match.group(1).strip() if from_match else ""
    subj_val = subj_match.group(1).strip() if subj_match else ""
    body = re.sub(HEADER_RE, " ", raw_email)
    body = re.sub(r"\s+", " ", body).strip()
    return from_val, subj_val, body

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = re.sub(HEADER_RE, " ", s)
    t = re.sub(r"(https?://\S+|www\.\S+)", " url ", t)
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " email ", t)
    t = re.sub(r"\b\d+(\.\d+)?\b", " num ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def rule_label(text_lower: str):
    """
    Returns:
      ("Prioritize"/"Slow", explanation_string) if rule triggers
      (None, None) otherwise
    """
    if any(re.search(p, text_lower) for p in PRIORITIZE_PATTERNS):
        return "Prioritize", "Rule-signal: security/auth (MFA/OTP/verification/login) detected."
    if any(re.search(p, text_lower) for p in SLOW_PATTERNS):
        return "Slow", "Rule-signal: promotional/marketing content detected."
    return None, None

# -----------------------------
# 3) Data & Training (ONLY sample dataset)
# -----------------------------
df = pd.read_csv("email_classification_dataset.csv").dropna(subset=["email"]).reset_index(drop=True)

df[["from", "subject", "body"]] = df["email"].apply(lambda x: pd.Series(parse_email_fields(x)))

# Silver labels (from rules) for training + evaluation on the provided dataset
df["priority_label"] = df.apply(lambda r: (
    "Prioritize" if any(re.search(p, f"{r['from']} {r['subject']} {r['body']}".lower()) for p in PRIORITIZE_PATTERNS)
    else "Slow" if any(re.search(p, f"{r['from']} {r['subject']} {r['body']}".lower()) for p in SLOW_PATTERNS)
    else "Default"
), axis=1)

df["model_text"] = (df["from"].fillna("") + " " + df["subject"].fillna("") + " " + df["body"].fillna("")).apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["model_text"],
    df["priority_label"],
    test_size=0.2,
    stratify=df["priority_label"],
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="multinomial"))
])

model.fit(X_train, y_train)

# Honest metrics: ML only (no rule forcing during evaluation)
y_pred = model.predict(X_test)
ACC = accuracy_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred, average="macro")
REPORT = classification_report(y_test, y_pred)
CM = confusion_matrix(y_test, y_pred, labels=model.classes_)

def cm_plot():
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(CM, display_labels=model.classes_).plot(ax=ax, cmap="Greens", colorbar=False)
    plt.tight_layout()
    return fig

# -----------------------------
# 4) Prediction Logic (RULES CAN SET LABEL, CONFIDENCE ALWAYS REAL)
# -----------------------------
def predict_email_event(sender, subject, body):
    combined = f"{sender} {subject} {body}"
    cleaned = clean_text(combined)

    # ML prediction + probabilities
    probs = model.predict_proba([cleaned])[0]
    classes = model.classes_
    ml_pred = model.predict([cleaned])[0]
    ml_conf = float(probs[int(np.where(classes == ml_pred)[0][0])])

    # Rule signal (if any)
    text_lower = combined.lower()
    r_label, r_reason = rule_label(text_lower)

    # Final label:
    # If rule triggers, we follow requirement semantics and use that label.
    # If not, use ML prediction.
    final_label = r_label if r_label is not None else ml_pred

    # Confidence must correspond to the final label (NOT forced to 100%)
    final_idx = int(np.where(classes == final_label)[0][0])
    final_conf = float(probs[final_idx])

    # Reasoning (transparent)
    if r_label is not None and r_label != ml_pred:
        reasoning = (
            f"{r_reason} Overrode ML prediction '{ml_pred}' "
            f"(ML confidence: {ml_conf:.2%})."
        )
    elif r_label is not None and r_label == ml_pred:
        reasoning = f"{r_reason} ML agrees with this label."
    else:
        reasoning = f"ML-only: classified as '{ml_pred}' based on learned statistical patterns."

    return final_label, f"{final_conf:.2%}", reasoning

def predict_ui(sender, subject, body):
    return predict_email_event(sender, subject, body)

# -----------------------------
# 5) UI
# -----------------------------
with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("## Email Prioritization Dashboard")

    with gr.Tabs():
        with gr.Tab("Test Email"):
            with gr.Row():
                with gr.Column(scale=1):
                    s_in = gr.Textbox(label="Sender", placeholder="sender@example.com")
                    sb_in = gr.Textbox(label="Subject", placeholder="Email subject")
                    b_in = gr.Textbox(label="Body", lines=5, placeholder="Email body")
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit", elem_id="submit-btn")
                with gr.Column(scale=1):
                    res_l = gr.Textbox(label="Priority Label")
                    res_c = gr.Textbox(label="Confidence")
                    res_r = gr.Textbox(label="Reasoning")

            gr.Examples(examples=examples, inputs=[s_in, sb_in, b_in])
            submit_btn.click(predict_ui, [s_in, sb_in, b_in], [res_l, res_c, res_r])
            clear_btn.click(lambda: ("","","","","",""), outputs=[s_in, sb_in, b_in, res_l, res_c, res_r])

        with gr.Tab("Metrics"):
            gr.Markdown("### Model Performance (ML on held-out test set)")
            with gr.Row():
                gr.Number(label="Accuracy", value=float(round(ACC, 4)))
                gr.Number(label="Macro-F1", value=float(round(F1, 4)))
            gr.Plot(value=cm_plot)
            gr.Textbox(label="Classification Report", value=REPORT, lines=12)

demo.launch()
