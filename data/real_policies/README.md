# Real policy PDFs (optional module, git-ignored)

Download policy wordings / brochures / CIS documents yourself (HDFC ERGO, Star Health, Care Health, Niva Bupa, IRDAI listings -
see the main README for links) and respect each site's terms of use. Put the PDFs here, then run:

    python scripts/evaluate_real_policies.py --pdf-dir data/real_policies [--truth truth.csv] [--llm gemini]

`truth.csv` is optional: columns `file,field,value` (for example `star_wording.pdf,copayment_percentage,20`).
Never commit documents that contain personal data; this folder's PDFs are ignored by git.
