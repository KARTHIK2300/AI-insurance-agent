"""Generates two FICTIONAL, anonymised sample policy PDFs for development and testing.

No real insurer, customer or product is represented. Values are deliberately varied so the
extractor, comparison and risk logic can be exercised (including conditional and missing clauses).
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

POLICY_A = {
    "file": "sample_current_policy_surecare_plus.pdf",
    "header": "Sample Shield Health Insurance Co. Ltd. | SureCare Plus (Sample) | FICTIONAL TEST DOCUMENT",
    "title": "SureCare Plus (Sample) - Policy Wording",
    "sections": [
        (
            "SECTION 1: POLICY SCHEDULE",
            [
                "Insurer: Sample Shield Health Insurance Co. Ltd.",
                "Policy Name: SureCare Plus (Sample)",
                "Policy Type: Family Floater Health Insurance Policy",
                "Policy Number: SSH-2024-000123",
                "Policyholder Name: Test User",
                "Contact: 9876543210, test.user@example.com",
                "Sum Insured: Rs. 5,00,000 per policy year on a floater basis.",
                "Annual Premium: Rs. 14,200 (inclusive of taxes).",
                "Policy Term: 1 year, renewable for life.",
                "Entry Age: 91 days to 65 years for adults. Dependent children are covered from 91 days up to 25 years.",
                "Members covered: Self, spouse and up to 2 dependent children.",
            ],
        ),
        (
            "SECTION 2: WHAT IS COVERED",
            [
                "1. In-patient hospitalisation: The Company will pay admissible medical expenses for hospitalisation of more than 24 hours up to "
                "the Sum Insured.",
                "2. Pre-hospitalisation: medical expenses incurred up to 30 days before admission are covered.",
                "3. Post-hospitalisation: medical expenses incurred up to 60 days after discharge are covered.",
                "4. Day care procedures: 140 listed day care procedures are covered when treatment takes less than 24 hours.",
                "5. Ambulance: road ambulance charges are covered up to Rs. 2,000 per hospitalisation.",
                "6. Domiciliary hospitalisation: treatment at home is covered when the patient cannot be moved to a hospital.",
                "7. AYUSH treatment: Ayurveda, Yoga, Unani, Siddha and Homeopathy inpatient treatment is covered in a recognised AYUSH hospital.",
                "8. No Claim Bonus: a cumulative bonus of 5% of the base Sum Insured is added for each claim-free year, up to a maximum of 50%.",
                "9. Dependent children can be added at inception or at renewal. Newborn babies are covered only from the next renewal after birth.",
                "This policy does not offer a restoration or recharge benefit.",
            ],
        ),
        (
            "SECTION 3: WAITING PERIODS",
            [
                "Initial waiting period: 30 days from the first policy start date, except for accidental injuries.",
                "Pre-existing diseases are covered only after a waiting period of 36 months of continuous coverage.",
                "Specific diseases waiting period: 24 months for cataract, hernia, hysterectomy, joint replacement and kidney stones.",
                "Maternity expenses are not covered under this policy.",
            ],
        ),
        (
            "SECTION 4: CO-PAYMENT AND LIMITS",
            [
                "Co-payment: 10% of every admissible claim is payable by the insured person. This applies to all claims and all insured persons.",
                "Room rent is limited to 1% of the Sum Insured per day for a normal room. ICU charges are limited to 2% of the Sum Insured per day.",
                "If the room rent exceeds the limit, all other expenses are reduced in the same proportion (proportionate deduction).",
                "Sub-limit for cataract surgery: Rs. 25,000 per eye.",
                "Sub-limit for hernia and hysterectomy: Rs. 50,000 per event.",
                "Sub-limit for joint replacement: Rs. 1,00,000 per event.",
                "Non-payable items such as consumables are not payable as per the standard list.",
            ],
        ),
        (
            "SECTION 5: EXCLUSIONS",
            [
                "1. Cosmetic or plastic surgery unless required after an accident.",
                "2. Dental treatment unless caused by an accident.",
                "3. Treatment for obesity or weight control.",
                "4. War, nuclear risks and terrorism.",
                "5. Self-inflicted injury, suicide attempt and substance abuse.",
                "6. Infertility, assisted reproduction and sterilisation.",
                "7. Hearing aids, spectacles and contact lenses.",
                "8. Experimental or unproven treatments.",
                "9. Adventure sports injuries.",
                "10. Congenital external diseases.",
                "11. Sexually transmitted diseases other than HIV.",
                "12. Treatment outside India.",
                "13. Vitamins and tonics unless part of inpatient treatment.",
                "14. Any expense that is not reasonable and customary as determined by the Company.",
            ],
        ),
        (
            "SECTION 6: CLAIMS PROCEDURE",
            [
                "Cashless: inform the Company at least 48 hours before a planned admission and within 24 hours of an emergency admission.",
                "Reimbursement: submit the claim form, discharge summary, original bills and investigation reports within 15 days of discharge.",
                "Claims may be reduced where charges are not as per the Company's underwriting guidelines.",
            ],
        ),
        (
            "SECTION 7: RENEWAL, PORTABILITY AND GRACE PERIOD",
            [
                "The policy is renewable for life. A grace period of 30 days is allowed for renewal payment.",
                "Portability to or from another insurer is available as per IRDAI rules. Premium rates may be revised from time to time.",
            ],
        ),
    ],
}

POLICY_B = {
    "file": "sample_proposed_policy_horizon_max.pdf",
    "header": "Northstar Sample General Insurance Ltd. | Horizon Max (Sample) | FICTIONAL TEST DOCUMENT",
    "title": "Horizon Max (Sample) - Product Brochure and Terms",
    "sections": [
        (
            "SECTION 1: POLICY SCHEDULE",
            [
                "Insurer: Northstar Sample General Insurance Ltd.",
                "Policy Name: Horizon Max (Sample)",
                "Policy Type: Family Floater Health Insurance Policy",
                "Sum Insured: Rs. 10,00,000 per policy year on a floater basis.",
                "Annual Premium: Rs. 19,800 (inclusive of taxes).",
                "Policy Term: 1 year, renewable for life.",
                "Entry Age: 18 years to 70 years for adults. Dependent children are covered from day 1 up to 23 years.",
                "Members covered: Self, spouse, up to 3 dependent children and parents.",
            ],
        ),
        (
            "SECTION 2: WHAT IS COVERED",
            [
                "1. In-patient hospitalisation: admissible expenses for hospitalisation of more than 24 hours are covered up to the Sum Insured.",
                "2. Pre-hospitalisation: expenses incurred up to 60 days before admission are covered.",
                "3. Post-hospitalisation: expenses incurred up to 90 days after discharge are covered.",
                "4. Day care procedures: all day care procedures are covered.",
                "5. Ambulance: road ambulance charges are covered up to Rs. 5,000 per hospitalisation.",
                "6. AYUSH treatment is covered up to 25% of the Sum Insured.",
                "7. Restore Benefit: 100% of the base Sum Insured is restored once per policy year for an unrelated illness.",
                "8. No Claim Bonus: 10% of the base Sum Insured is added for each claim-free year, up to 100%.",
                "9. Maternity: maternity expenses are covered up to Rs. 50,000 per delivery after a waiting period of 36 months.",
                "10. Newborn baby is covered from day 1 up to 90 days within the maternity limit.",
                "11. OPD consultation benefit up to Rs. 5,000 per year is available as an optional add-on.",
            ],
        ),
        (
            "SECTION 3: WAITING PERIODS",
            [
                "Initial waiting period: 30 days from the first policy start date, except for accidental injuries.",
                "Pre-existing diseases are covered after a waiting period of 48 months of continuous coverage. Waiting period may be reduced at "
                "the sole discretion of the Company for portability cases.",
                "Specific diseases waiting period: 36 months for cataract, hernia, hysterectomy and joint replacement.",
            ],
        ),
        (
            "SECTION 4: CO-PAYMENT AND LIMITS",
            [
                "Co-payment: no co-payment applies for insured persons aged up to 60 years. A co-payment of 20% applies for insured persons aged "
                "61 years and above.",
                "Room rent: any room category up to a single private room is covered. There is no capping on room rent for this Sum Insured.",
                "Sub-limit for cataract surgery: Rs. 40,000 per eye.",
                "Sub-limit for joint replacement: Rs. 1,50,000 per event.",
                "Non-payable items are not payable as per the IRDAI list.",
            ],
        ),
        (
            "SECTION 5: EXCLUSIONS",
            [
                "1. Cosmetic or plastic surgery unless required after an accident.",
                "2. Dental treatment unless caused by an accident.",
                "3. Treatment for obesity or weight control.",
                "4. War, nuclear risks and terrorism.",
                "5. Self-inflicted injury and substance abuse.",
                "6. Infertility and assisted reproduction.",
                "7. Hearing aids, spectacles and contact lenses.",
                "8. Experimental treatments.",
                "9. Adventure sports injuries.",
                "10. Treatment outside India.",
                "11. Expenses on treatments that are subject to underwriting loading.",
                "12. Any other condition as may be decided by the Company from time to time.",
            ],
        ),
        (
            "SECTION 6: CLAIMS PROCEDURE",
            [
                "Cashless: network hospital admission with pre-authorisation. Intimate the Company within 24 hours of emergency admission.",
                "Reimbursement: submit the claim form, discharge summary and original bills within 30 days of discharge.",
            ],
        ),
        (
            "SECTION 7: RENEWAL, PORTABILITY AND GRACE PERIOD",
            [
                "The policy is renewable for life. A grace period of 30 days is allowed for renewal payment.",
                "Portability is available as per IRDAI rules.",
            ],
        ),
    ],
}


def _build(spec: dict, out_path: Path) -> Path:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=4)
    head = ParagraphStyle("h", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(20 * mm, 285 * mm, spec["header"])
        canvas.drawString(95 * mm, 10 * mm, f"Page {doc.page} of 4")
        canvas.restoreState()

    story = [Paragraph(spec["title"], styles["Title"]), Spacer(1, 6)]
    for idx, (heading, lines) in enumerate(spec["sections"]):
        if idx in (1, 3, 5):  # force 4 pages so headers/footers repeat and citations span pages
            story.append(PageBreak())
        story.append(Paragraph(heading, head))
        for ln in lines:
            story.append(Paragraph(ln.replace("&", "&amp;"), body))
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=25 * mm, bottomMargin=20 * mm)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return out_path


def generate_samples(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [_build(POLICY_A, out_dir / POLICY_A["file"]), _build(POLICY_B, out_dir / POLICY_B["file"])]


if __name__ == "__main__":  # pragma: no cover
    from .config import SAMPLES_DIR

    print(generate_samples(SAMPLES_DIR))
