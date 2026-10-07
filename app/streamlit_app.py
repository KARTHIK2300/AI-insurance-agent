"""CoverWise AI - Streamlit dashboard.  Run:  streamlit run app/streamlit_app.py"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from coverwise import premium_model  # noqa: E402
from coverwise import review as rv  # noqa: E402
from coverwise.config import DISCLAIMER, LABELS, MODELS_DIR, REPORTS_DIR, SAMPLES_DIR, feature_enabled  # noqa: E402
from coverwise.extractor import FIELD_LABELS  # noqa: E402
from coverwise.llm import get_llm  # noqa: E402
from coverwise.pipeline import analyze, analyze_pair, load_policy, refresh_pair, sample_paths  # noqa: E402
from coverwise.rag import PolicyIndex, answer_question  # noqa: E402
from coverwise.report import generate_report  # noqa: E402
from coverwise.risk import advisor_questions  # noqa: E402
from coverwise.scoring import ScoringModel, model_status  # noqa: E402
from coverwise.summary import polish_summary  # noqa: E402

st.set_page_config(page_title="CoverWise AI", page_icon="🛡️", layout="wide")
SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "⚪"}
VERDICT_BG = {"Improved": "#d7f5dd", "Worse": "#fbd9d9", "Unclear": "#fff0c2", "Same": "#f1f3f6", "Info": "#f1f3f6"}


@st.cache_resource
def get_model():
    return ScoringModel.load()


def clause_table(analysis: dict) -> pd.DataFrame:
    rows = []
    for k, f in analysis["extraction"]["fields"].items():
        if k not in FIELD_LABELS:
            continue
        rows.append(
            {
                "Clause": FIELD_LABELS[k],
                "Value": "Not found" if f["value"] is None else str(f["value"]),
                "Page": f["page"],
                "Section": f["section"],
                "Source": f["source"],
                "Confidence": round(f["confidence"], 2),
                "Evidence": f["snippet"],
            }
        )
    return pd.DataFrame(rows)


def show_policy(analysis: dict, title: str):
    sc = analysis["score"]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{title}: Coverage Clarity Score", f"{sc['score']:.0f} / 100")
    c2.metric("Risk level", sc["label"] + (" - needs advisor review" if sc["label"] == "High Risk" else ""))
    c3.metric("Model confidence", f"{sc['confidence']:.0%}" if sc.get("confidence") else "rule-based")
    st.caption(
        f"Method: {sc['method']}. Rule-based cross-check: {sc['rule_score']:.0f} ({sc['rule_label']}). "
        f"Citation coverage of key clauses: {analysis['meta']['citation_coverage_score']:.0f}%."
    )
    st.markdown("**Plain-English summary**")
    st.write("\n".join(f"- {ln}" for ln in analysis["summary_lines"]))
    left, right = st.columns(2)
    with left:
        st.markdown("**Why this score (points vs an average policy)**")
        if sc["reasons"]:
            st.bar_chart(pd.DataFrame(sc["reasons"]).set_index("factor")["points"], horizontal=True)
    with right:
        st.markdown("**Risk flags and caution points**")
        for f in analysis["flags"]:
            pg = f" (p.{f['page']})" if f["page"] else ""
            st.markdown(f"{SEV_ICON[f['severity']]} **{f['title']}**{pg} - {f['detail']}")
        if not analysis["flags"]:
            st.write("No flags raised.")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛡️ CoverWise AI")
    st.caption("Coverage Insight and Risk Intelligence System")
    st.warning(DISCLAIMER)
    use_samples = st.checkbox("Use bundled fictional sample policies", value=False, key="use_samples")
    st.subheader("Your profile")
    age = st.number_input("Age of eldest insured", 18, 90, 35, key="age")
    n_children = st.number_input("Dependent children", 0, 6, 1, key="n_children")
    planning_child = st.checkbox("Planning a child / maternity cover matters", key="planning_child")
    has_ped = st.checkbox("A family member has a pre-existing condition", key="has_ped")
    concern = st.text_input("Optional concern (e.g. co-payment, waiting period, switching)", key="concern")
    with st.expander("Enter premium / sum insured if not detected"):
        cur_prem = st.number_input("Current annual premium (Rs.)", 0, 10_000_000, 0, key="cur_prem")
        cur_si = st.number_input("Current sum insured (Rs.)", 0, 100_000_000, 0, key="cur_si")
        new_prem = st.number_input("Proposed annual premium (Rs.)", 0, 10_000_000, 0, key="new_prem")
        new_si = st.number_input("Proposed sum insured (Rs.)", 0, 100_000_000, 0, key="new_si")
    with st.expander("Optional modules (switch on when needed)"):
        opt_review = st.checkbox("Human review mode (correct values, save labelled data)", value=feature_enabled("human_review"), key="opt_review")
        opt_premium = st.checkbox("Premium benchmark (practice model)", value=feature_enabled("premium_benchmark"), key="opt_premium")
        st.caption(
            "Also available from the command line: real-policy evaluation "
            "(scripts/evaluate_real_policies.py) and deployment (Docker, EC2, Hugging Face)."
        )
    with st.expander("LLM (optional)"):
        provider = st.selectbox("Provider", ["None (offline rules)", "Gemini", "Groq"], key="provider")
        api_key = st.text_input("API key (kept in memory only; or set env var)", type="password", key="api_key")
        st.caption(
            "Without a key the app runs fully offline using the rule-based extractor. Personal identifiers are masked before text is sent to an LLM."
        )

st.header("Policy decoder, comparison and risk explanation")
c1, c2 = st.columns(2)
cur_file = c1.file_uploader("Current policy PDF", type="pdf", key="cur_file", disabled=use_samples)
new_file = c2.file_uploader("Proposed policy PDF / brochure (optional)", type="pdf", key="new_file", disabled=use_samples)
st.caption("Use anonymised or sample documents only for public demos - never upload real personal policy documents to a public deployment.")

if st.button("Analyze policy", type="primary", key="analyze_btn"):
    profile = {"age": age, "n_children": n_children, "planning_child": planning_child, "has_pre_existing": has_ped, "concern": concern}
    overrides = {
        "current_premium": cur_prem or None,
        "current_si": cur_si or None,
        "proposed_premium": new_prem or None,
        "proposed_si": new_si or None,
    }
    llm = get_llm(None if provider.startswith("None") else provider.lower(), api_key or None) if not provider.startswith("None") else None
    try:
        with st.spinner("Reading, cleaning and analysing the policy..."):
            if use_samples:
                cp, np_ = sample_paths(SAMPLES_DIR)
                cur_doc, new_doc = load_policy("Current policy (sample)", cp), load_policy("Proposed policy (sample)", np_)
            elif cur_file is not None:
                cur_doc = load_policy(cur_file.name, cur_file.getvalue())
                new_doc = load_policy(new_file.name, new_file.getvalue()) if new_file is not None else None
            else:
                st.error("Upload a current policy PDF or tick the sample-policies option.")
                st.stop()
            model = get_model()
            if new_doc is not None:
                result = analyze_pair(cur_doc, new_doc, llm, model, profile, overrides)
            else:
                cur = analyze(cur_doc, llm, model, profile, {"sum_insured": overrides["current_si"], "annual_premium": overrides["current_premium"]})
                result = {"current": cur, "questions": advisor_questions(cur["flags"], None, profile)}
            result["llm_used"] = llm is not None
            st.session_state.update(result=result, profile=profile, indexes={}, chat=[])
    except Exception as exc:
        st.error(f"Could not process the document: {exc}")
        st.stop()

result = st.session_state.get("result")
if not result:
    st.info("Upload a policy (or tick the sample option) and press **Analyze policy**.")
    st.stop()

pages = [d for d in result["current"]["doc"].pages]
if any(p["method"] == "ocr" for p in pages):
    st.info("Some pages were scanned images and were read with OCR - extraction may be less accurate.")
tab_names = ["Score & summary", "Extracted clauses", "Compare policies", "Ask the policy", "Advisor questions & report"]
if opt_review:
    tab_names.append("Human review")
tab_names.append("Models & evaluation")
tab = dict(zip(tab_names, st.tabs(tab_names)))

with tab["Score & summary"]:
    show_policy(result["current"], "Current policy")
    if "proposed" in result:
        st.divider()
        show_policy(result["proposed"], "Proposed policy")

with tab["Extracted clauses"]:
    for key, label in (("current", "Current policy"), ("proposed", "Proposed policy")):
        if key in result:
            st.subheader(label)
            ex = result[key]["extraction"]
            st.dataframe(clause_table(result[key]), width="stretch", hide_index=True)
            for msg in ex["validation"] + ex["llm_conflicts"]:
                st.warning(msg)
            if ex["unclear_fields"]:
                st.error("Unclear / not found - verify with the insurer: " + ", ".join(ex["unclear_fields"]))
            if ex["llm_note"]:
                st.caption(ex["llm_note"])

with tab["Compare policies"]:
    if "comparison" not in result:
        st.info("Upload a proposed policy to see the comparison.")
    else:
        cmp_ = result["comparison"]
        st.success(cmp_["premium"]["message"])
        df = pd.DataFrame(cmp_["rows"])[["label", "current", "proposed", "verdict"]].rename(columns=str.title)
        styler = df.style
        styler = (styler.map if hasattr(styler, "map") else styler.applymap)(
            lambda v: f"background-color:{VERDICT_BG.get(v, '')}", subset=["Verdict"]
        )
        st.dataframe(styler, width="stretch", hide_index=True)
        a, b, c = st.columns(3)
        a.markdown("**Appears stronger**\n\n" + ("\n".join(f"- {x}" for x in cmp_["improved"]) or "none"))
        b.markdown("**Needs caution**\n\n" + ("\n".join(f"- {x}" for x in cmp_["worse"]) or "none"))
        c.markdown("**Unclear - verify**\n\n" + ("\n".join(f"- {x}" for x in cmp_["unclear"]) or "none"))
        st.markdown("**Switching risks**")
        st.write("\n".join(f"- {x}" for x in cmp_["switching_risks"]))
        if cmp_["user_cautions"]:
            st.markdown("**Cautions specific to your profile**")
            st.write("\n".join(f"- {x}" for x in cmp_["user_cautions"]))

with tab["Ask the policy"]:
    options = [k for k in ("current", "proposed") if k in result]
    which = st.radio("Ask about", options, horizontal=True, key="qa_which")
    a = result[which]
    idx = st.session_state["indexes"].setdefault(which, PolicyIndex(a["doc"]))
    st.caption("Answers come only from the uploaded document and show the page/section. Unsupported questions are declined.")
    examples = [
        "Is co-payment applicable in this policy?",
        "What is the waiting period for pre-existing diseases?",
        "Is room rent capped?",
        "Does the policy cover newborn baby or dependent child?",
        "What are the major exclusions?",
        "What should I verify before switching?",
        "Should I buy this policy?",
    ]
    picked = st.selectbox("Example questions", [""] + examples, key="qa_example")
    q = st.text_input("Your question", value=picked, key="qa_text")
    if q:
        llm = (
            get_llm(
                None if st.session_state.get("provider", "None").startswith("None") else st.session_state["provider"].lower(),
                st.session_state.get("api_key") or None,
            )
            if not st.session_state.get("provider", "None").startswith("None")
            else None
        )
        res = answer_question(q, idx, a["extraction"], a["flags"], result.get("questions"), llm)
        (st.warning if res["refused"] else st.success)("Declined / not found" if res["refused"] else "Answered from the document")
        st.write(res["answer"])
        if res["citations"]:
            with st.expander("Sources"):
                for c in res["citations"]:
                    st.markdown(f"**p.{c['page']} - {c['section']}**: {c['text']}")

with tab["Advisor questions & report"]:
    st.subheader("Questions to ask your advisor / insurer")
    for i, qn in enumerate(result["questions"], 1):
        st.markdown(f"{i}. {qn}")
    summary_text = None
    if result.get("llm_used"):
        llm = get_llm(st.session_state["provider"].lower(), st.session_state.get("api_key") or None)
        summary_text = polish_summary(result["current"]["summary_lines"], llm)
    if st.button("Generate PDF report", key="report_btn"):
        out = Path(tempfile.gettempdir()) / "coverwise_report.pdf"
        generate_report(out, result, st.session_state["profile"], summary_text)
        st.session_state["report_bytes"] = out.read_bytes()
    if st.session_state.get("report_bytes"):
        st.download_button("Download PDF report", st.session_state["report_bytes"], "coverwise_report.pdf", "application/pdf", key="dl_btn")

if opt_review:
    with tab["Human review"]:
        st.subheader("Human review of extracted clauses")
        st.caption(
            "Confirm or correct values; your corrections re-score the policy and can be saved as labelled data "
            "(only a document hash and numeric features are stored - never document text)."
        )
        which_r = st.radio("Policy to review", [k for k in ("current", "proposed") if k in result], horizontal=True, key="rv_which")
        a_r = result[which_r]
        original = st.session_state.setdefault("orig_extractions", {}).setdefault(which_r, a_r["extraction"])
        rows = [
            {
                "field": k,
                "label": FIELD_LABELS[k],
                "extracted": "Not found" if original["fields"][k]["value"] is None else str(original["fields"][k]["value"]),
                "page": original["fields"][k]["page"],
                "corrected value (blank = keep)": "",
            }
            for k in rv.REVIEW_FIELDS
        ]
        edited = st.data_editor(
            pd.DataFrame(rows), hide_index=True, width="stretch", key=f"rv_editor_{which_r}", disabled=["field", "label", "extracted", "page"]
        )
        c1, c2 = st.columns(2)
        reviewer = c1.text_input("Reviewer name / id", key="rv_reviewer")
        final_label = c2.selectbox("Final risk label (your judgement)", LABELS, index=LABELS.index(a_r["score"]["label"]), key="rv_label")
        if st.button("Apply corrections and re-score", key="rv_apply"):
            fixes, bad = {}, []
            for r in edited.itertuples(index=False):
                txt = str(r[-1]).strip()
                if txt and txt.lower() != "nan":
                    ok, val = rv.parse_correction(r.field, txt)
                    (fixes.__setitem__(r.field, val) if ok else bad.append(r.field))
            if bad:
                st.error("Could not read values for: " + ", ".join(bad) + " (use numbers, yes/no, or 'not stated').")
            else:
                reviewed = rv.apply_review(original, fixes)
                result[which_r] = analyze(a_r["doc"], model=get_model(), profile=st.session_state["profile"], extraction=reviewed)
                refresh_pair(result, st.session_state["profile"])
                st.session_state["indexes"].pop(which_r, None)
                st.success(
                    f"Re-scored with {len(fixes)} corrected field(s): "
                    f"{result[which_r]['score']['score']:.0f}/100 - {result[which_r]['score']['label']}"
                )
                st.rerun()
        if original is not a_r["extraction"]:
            st.info(
                "Reviewed fields: " + ", ".join(FIELD_LABELS[k] for k in a_r["extraction"]["reviewed_fields"])
                if a_r["extraction"]["reviewed_fields"]
                else "No fields changed yet."
            )
        if st.button("Save review to labelled set", key="rv_save"):
            try:
                rv.log_review(a_r, original, reviewer, final_label)
                n_rows = len(rv.export_labelled())
                st.success(f"Saved. The labelled set now has {n_rows} reviewed policy(ies) (data/labelled/real_labelled.csv).")
            except ValueError as exc:
                st.error(str(exc))
        summ = rv.review_summary()
        if summ["n_reviews"]:
            st.markdown(
                f"**Review statistics:** {summ['n_reviews']} reviews, label agreement with model {summ['label_agreement_with_model']:.0%}, "
                f"overall field accuracy {summ['overall_field_accuracy']:.0%}"
            )
            if summ["field_correction_rate"]:
                st.bar_chart(pd.Series(summ["field_correction_rate"], name="correction rate"))
            if rv.EXPORT_PATH.exists():
                st.download_button("Download labelled CSV", rv.EXPORT_PATH.read_bytes(), "real_labelled.csv", "text/csv", key="rv_dl")
            st.caption(
                "Retrain on real reviewed data with: python scripts/train_model.py --include-labelled "
                "(needs 20+ reviews), then run the quality gates."
            )

with tab["Models & evaluation"]:
    ms = model_status()
    if ms["available"]:
        (st.success if ms["verified"] else st.error)(
            f"Model {ms['version']} - integrity: {ms['message']} - sha256 {ms['sha256']}... - CI quality gates passed: {ms['gates_passed']}"
        )
    else:
        st.warning(ms["message"])
    gate_file = REPORTS_DIR / "quality_gate_result.json"
    if gate_file.exists():
        with st.expander("CI quality-gate results"):
            st.dataframe(pd.DataFrame(json.loads(gate_file.read_text())["checks"]), hide_index=True, width="stretch")
    mpath, epath = MODELS_DIR / "metrics.json", REPORTS_DIR / "evaluation_results.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        st.subheader("Risk scoring models (5-fold out-of-fold)")
        st.dataframe(pd.DataFrame(m["classifiers"]).T.round(3), width="stretch")
        fig = REPORTS_DIR / "figures" / "confusion_matrix.png"
        if fig.exists():
            st.image(str(fig), width=380)
        st.caption("risk_flags_count is derived from the other features; the production model excludes it to avoid leakage.")
    if epath.exists():
        st.subheader("Extraction and Q&A evaluation on the sample policies")
        st.json(json.loads(epath.read_text()), expanded=False)
    pm = premium_model.load() if opt_premium else None
    if pm:
        with st.expander("Optional: indicative premium benchmark (practice model, not used for scoring)"):
            st.caption("Trained on the synthetic Acko/Kaggle-style premium data, which has no policy clauses. Treat as a rough benchmark only.")
            a1, a2, a3 = st.columns(3)
            p_age = a1.number_input("Age", 18, 90, 35, key="pm_age")
            p_smoke = a2.selectbox("Smoker", ["No", "Yes"], key="pm_smoke")
            p_type = a3.selectbox("Plan tier", ["Basic", "Comprehensive", "Premium"], key="pm_type")
            st.metric(
                "Indicative premium",
                f"Rs. {premium_model.predict_premium(pm, {'Age': p_age, 'Smoking Status': p_smoke, 'Policy Type': p_type}):,.0f}",
            )

st.divider()
st.caption(DISCLAIMER)
