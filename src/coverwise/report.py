"""PDF report generation (reportlab). Uses 'Rs.' instead of the rupee glyph (not in built-in fonts)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import DISCLAIMER
from .extractor import FIELD_LABELS
from .risk import advisor_questions

NAVY, GREY = colors.HexColor("#14213d"), colors.HexColor("#f1f3f6")
VERDICT_COLOR = {"Improved": "#1a7f37", "Worse": "#c62828", "Unclear": "#b26a00", "Same": "#57606a", "Info": "#57606a"}
SEV_COLOR = {"high": "#c62828", "medium": "#b26a00", "low": "#57606a"}
SHOWN_FIELDS = [
    "insurer_name",
    "policy_type",
    "sum_insured",
    "annual_premium",
    "policy_term",
    "entry_age",
    "pre_hosp_days",
    "post_hosp_days",
    "initial_waiting_months",
    "ped_waiting_months",
    "specific_waiting_months",
    "copayment_percentage",
    "room_rent_limit",
    "icu_limit",
    "disease_sub_limit_count",
    "exclusion_count",
    "restoration_benefit",
    "no_claim_bonus",
    "child_coverage",
    "newborn_coverage",
    "maternity_coverage",
    "day_care",
    "ambulance",
    "cashless",
    "reimbursement",
    "grace_period_days",
]


def _p(text, style):
    return Paragraph(escape(str(text)), style)


def _fmt(f: dict) -> str:
    v = f["value"]
    if v is None:
        return "Not found"
    if isinstance(v, bool):
        return "Yes" if v else ("No" if f["confidence"] >= 0.5 else "Not mentioned")
    if isinstance(v, int) and v >= 1000:
        return f"{v:,}"
    return str(v) + (f" - {f['detail']}" if f.get("detail") and "Conditional" in f["detail"] else "")


def generate_report(path: str | Path, result: dict, profile: dict | None = None, summary_text: str | None = None) -> Path:
    """result: output of pipeline.analyze_pair (has 'comparison') or {'current': analysis}."""
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=18, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=NAVY, fontSize=12.5, spaceBefore=12, spaceAfter=5)
    B = ParagraphStyle("B", parent=ss["Normal"], fontSize=9.2, leading=12.5)
    S = ParagraphStyle("S", parent=B, fontSize=8.2, leading=10.5)
    story = [
        Paragraph("CoverWise AI - Policy Understanding Report", H1),
        _p(f"Generated {date.today().isoformat()} | Decision-support only", S),
        Spacer(1, 6),
    ]
    box = Table([[_p("IMPORTANT: " + DISCLAIMER, B)]], colWidths=[174 * mm])
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#c62828")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5f5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story += [box]

    if profile:
        bits = [f"{k.replace('_', ' ')}: {v}" for k, v in profile.items() if v not in (None, "", [], False)]
        if bits:
            story += [Paragraph("Your profile (as entered)", H2), _p("; ".join(bits), B)]

    sections = [("current", "Current policy")] + ([("proposed", "Proposed policy")] if "proposed" in result else [])
    for key, title in sections:
        a = result[key]
        F, sc = a["extraction"]["fields"], a["score"]
        story.append(Paragraph(f"{title}: {escape(str(F['policy_name']['value'] or a['extraction']['policy_name']))}", H2))
        lines = summary_text.split("\n") if summary_text and key == "current" else a["summary_lines"]
        for ln in lines:
            story.append(_p("- " + ln.lstrip("- "), B))
        rev = a["extraction"].get("reviewed_fields") or []
        if rev:
            story.append(_p("Human-reviewed: " + ", ".join(FIELD_LABELS[k] for k in rev) + " (values corrected by a reviewer).", S))
        conf = f" | model confidence {sc['confidence']:.0%}" if sc.get("confidence") else ""
        label = sc["label"] + (" / Needs advisor review" if sc["label"] == "High Risk" else "")
        story += [
            Spacer(1, 4),
            _p(
                f"Coverage Clarity Score: {sc['score']:.0f}/100 | Risk level: {label}{conf}",
                ParagraphStyle("sc", parent=B, fontSize=10.5, textColor=NAVY, fontName="Helvetica-Bold"),
            ),
            _p(
                f"Method: {sc['method']}. Rule-based cross-check: {sc['rule_score']:.0f} ({sc['rule_label']}). Citation coverage of key clauses: "
                f"{a['meta']['citation_coverage_score']:.0f}%.",
                S,
            ),
        ]
        rs = [[_p("Top reasons behind the score (vs an average policy)", S), _p("Points", S)]] + [
            [_p(f"{r['factor']} (value: {r['value']})", S), _p(f"{r['points']:+.1f}", S)] for r in sc["reasons"][:6]
        ]
        t = Table(rs, colWidths=[140 * mm, 24 * mm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), GREY), ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey)]))
        story += [Spacer(1, 4), t]

        rows = [[_p("Clause", S), _p("Value", S), _p("Page", S)]]
        for k in SHOWN_FIELDS:
            rows.append([_p(FIELD_LABELS[k], S), _p(_fmt(F[k]), S), _p(F[k]["page"] or "-", S)])
        t = Table(rows, colWidths=[62 * mm, 92 * mm, 20 * mm], repeatRows=1)
        t.setStyle(
            TableStyle(
                [("BACKGROUND", (0, 0), (-1, 0), GREY), ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
            )
        )
        story += [Paragraph(f"{title}: extracted coverage and restrictions", H2), t]

        story.append(Paragraph(f"{title}: risk flags and caution points", H2))
        for f in a["flags"] or []:
            col = SEV_COLOR[f["severity"]]
            pg = f" (p.{f['page']})" if f["page"] else ""
            story.append(
                Paragraph(f'<font color="{col}"><b>[{f["severity"].upper()}]</b></font> <b>{escape(f["title"])}</b>{pg} - {escape(f["detail"])}', B)
            )
        if not a["flags"]:
            story.append(_p("No flags raised by the rules.", B))
        if a["extraction"]["unclear_fields"] or a["extraction"]["validation"]:
            story.append(Paragraph("Unclear or inconsistent items to verify", H2))
            for u in a["extraction"]["unclear_fields"] + a["extraction"]["validation"] + a["extraction"]["llm_conflicts"]:
                story.append(_p("- " + u, B))

    if "comparison" in result:
        cmp_ = result["comparison"]
        story.append(Paragraph("Current vs proposed policy", H2))
        rows = [[_p("Item", S), _p("Current", S), _p("Proposed", S), _p("Verdict", S)]]
        for r in cmp_["rows"]:
            rows.append(
                [
                    _p(r["label"], S),
                    _p(r["current"], S),
                    _p(r["proposed"], S),
                    Paragraph(f'<font color="{VERDICT_COLOR[r["verdict"]]}"><b>{r["verdict"]}</b></font>', S),
                ]
            )
        t = Table(rows, colWidths=[64 * mm, 40 * mm, 40 * mm, 22 * mm], repeatRows=1)
        t.setStyle(
            TableStyle(
                [("BACKGROUND", (0, 0), (-1, 0), GREY), ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
            )
        )
        story += [t, Spacer(1, 6), Paragraph("Premium vs coverage", H2), _p(cmp_["premium"]["message"], B)]
        story.append(Paragraph("The proposed policy appears stronger in these areas", H2))
        story.append(_p(", ".join(cmp_["improved"]) or "none identified", B))
        story.append(Paragraph("The proposed policy needs caution in these areas", H2))
        story.append(_p(", ".join(cmp_["worse"]) or "none identified", B))
        story.append(Paragraph("Unclear - must be verified", H2))
        story.append(_p(", ".join(cmp_["unclear"]) or "none", B))
        story.append(Paragraph("Switching risks", H2))
        for s in cmp_["switching_risks"]:
            story.append(_p("- " + s, B))
        if cmp_["user_cautions"]:
            story.append(Paragraph("User-specific caution points", H2))
            for s in cmp_["user_cautions"]:
                story.append(_p("- " + s, B))
        questions = result.get("questions") or []
    else:
        questions = advisor_questions(result["current"]["flags"], None, profile)

    story.append(KeepTogether([Paragraph("Questions to ask your advisor / insurer", H2)] + [_p(f"{i}. {q}", B) for i, q in enumerate(questions, 1)]))
    story += [
        Paragraph("Limitations and responsible-AI note", H2),
        _p(
            "Values were extracted automatically from the uploaded documents and may be incomplete or misread, especially for scanned files. "
            "Scores come from a model trained on a synthetic dataset and describe how clear and restrictive the wording appears; they do not "
            "predict whether a claim will be paid. This report does not recommend buying, switching, renewing or cancelling any policy. "
            "A human review of the original policy wording by the insurer or a licensed advisor is required before any decision.",
            B,
        ),
    ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(
            18 * mm, 10 * mm, "CoverWise AI - policy understanding and decision support only. Not insurance, legal, financial or medical advice."
        )
        canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm, title="CoverWise AI report"
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return path
