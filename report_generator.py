# report_generator.py

import pdfkit
import datetime
import logging
import os
import platform
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger("report_gen")
logger.setLevel(logging.INFO)

def make_json_serializable(obj):
    """
    Recursively convert custom objects to JSON-safe data types.
    (Pandas, NumPy, or dataclasses, etc.)
    """
    import numpy as np
    import pandas as pd
    from dataclasses import is_dataclass, asdict

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    # Dataclass -> asdict
    if is_dataclass(obj):
        return make_json_serializable(asdict(obj))
    # Fallback: string
    return str(obj)


def generate_investor_report(
    doc: dict,
    system_dyn: Optional[List[float]] = None,
    sir_data: Optional[Any] = None,
    hpc_data: Optional[List[dict]] = None,
    patterns_matched: Optional[List[dict]] = None,
    cash_flow: Optional[List[float]] = None,
    unit_economics: Optional[dict] = None,
    include_recommendations: bool = True,
    include_financials: bool = True,
    include_market: bool = True,
    include_team: bool = True,
    include_tech: bool = True,
    include_appendix: bool = False
) -> bytes:
    """
    Generates a multi-page PDF with all your sections:
    - Executive Summary
    - Patterns & Observations
    - Scenario Modeling
    - Financial Summary
    - Market & Sector
    - Team & Intangible
    - Technical Overview
    - Extended Analysis
    - (Optional) JSON Appendix

    If you find it produces 200+ pages from the JSON,
    you can truncate the doc's JSON by limiting docjson length.
    """

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Fallback if name not in doc
    name = doc.get("name", "StartupX")
    sector = doc.get("sector", "N/A").capitalize()
    stage = doc.get("stage", "N/A")
    final_score = float(doc.get("flashdna_score", doc.get("success_prob", 50.0)))
    sp = float(doc.get("success_prob", 50.0))
    intangible = float(doc.get("intangible", 50.0))
    team = float(doc.get("team_score", 0.0))
    moat = float(doc.get("moat_score", 0.0))

    # Turn matched patterns into bullet list
    pat_html = "<p>No patterns matched</p>"
    if patterns_matched:
        pat_html = "<ul>"
        for pm in patterns_matched:
            nm = pm.get("name", "Pattern")
            ds = pm.get("description", "")
            pat_html += f"<li><strong>{nm}</strong>: {ds}</li>"
        pat_html += "</ul>"

    # HPC scenario examples
    scenario_html = "<p>No scenario data available</p>"
    if hpc_data:
        # Sort by final_users descending just as an example
        sorted_hpc = sorted(hpc_data, key=lambda x: x.get("final_users", 0), reverse=True)
        scenario_html = "<ol>"
        for sc in sorted_hpc[:5]:
            c = sc.get("churn", 0)
            r = sc.get("referral", 0)
            f = sc.get("final_users", 0)
            scenario_html += f"<li>Churn={c:.2%}, Referral={r:.2%}, FinalUsers={f:,.0f}</li>"
        scenario_html += "</ol>"

    sys_html = ""
    if system_dyn and len(system_dyn) > 0:
        sys_html = f"<p>Scenario engine final user count => {system_dyn[-1]:,.0f}</p>"

    sir_html = ""
    if sir_data and len(sir_data) == 3:
        # Typically sir_data is (S,I,R)
        I = sir_data[1] if sir_data[1] else []
        if I:
            peak = max(I)
            sir_html = f"<p>SIR Model peak active => {peak:,.0f}</p>"

    # Cash flow & runway
    cf_html = ""
    if cash_flow:
        runw = doc.get("runway_months", 9999)
        if runw >= 9999:
            runw_txt = "∞"
        else:
            runw_txt = str(runw)
        ending_cash = doc.get("ending_cash", 0)
        cf_html = (
            f"<p>Runway => <strong>{runw_txt}</strong> months<br/>"
            f"Ending Cash => ${ending_cash:,.2f}</p>"
        )

    # Unit economics
    ue_html = ""
    if unit_economics:
        ltv = unit_economics.get("ltv", 0.0)
        cac = unit_economics.get("cac", 0.0)
        ratio = unit_economics.get("ltv_cac_ratio", 0.0)
        gm = unit_economics.get("gross_margin", 0.0) * 100
        ue_html = f"""
        <ul>
          <li>LTV => ${ltv:,.2f}</li>
          <li>CAC => ${cac:,.2f}</li>
          <li>LTV:CAC => {ratio:.2f}x</li>
          <li>Gross Margin => {gm:.1f}%</li>
        </ul>
        """

    # rating_text
    if final_score < 40:
        rating_text = "Below Average => big improvements needed"
    elif final_score < 60:
        rating_text = "Average => moderate improvements recommended"
    elif final_score < 80:
        rating_text = "Above Average => strong fundamentals"
    else:
        rating_text = "Exceptional => intangible & advanced scenario synergy"

    disclaimers = """
    <p><strong>Disclaimer:</strong><br/>
    This analysis is for illustrative purposes only.
    No warranty is given or implied. Forward-looking statements carry risk.
    </p>
    """

    extended_text = """
    <h3>Extended Analysis</h3>
    <p>
    <strong>1)</strong> We combine advanced scenario modeling (churn/referral-based user growth)
    with an AI intangible rating (clarity/passion).  
    This merges numeric projections with investor perception.
    </p>
    <p>
    <strong>2)</strong> The scenario engine can vary monthly marketing or churn to find your final user expansions.
    The intangible rating shows how effectively the pitch resonates.
    Each aspect should reinforce the other.
    </p>
    <p>
    <strong>3)</strong> As your startup matures, run intangible AI with updated pitch text if scenario data changes
    (like improved burn or new marketing channels).  
    If intangible rating is moderate, but scenario is highly favorable, highlight numeric traction more strongly.
    If intangible rating is excellent but numeric scenario is weak, refine your top metrics or churn plan.
    </p>
    """

    rec_html = ""
    if include_recommendations:
        rec_html = """
        <h2>Recommendations</h2>
        <ul>
          <li>Review scenario monthly => update churn/referral assumptions with real data.</li>
          <li>Continue refining intangible AI rating for clarity improvements.</li>
          <li>Balance strong narrative with actual KPI traction => investor confidence stems from both.</li>
        </ul>
        """

    # Build out big HTML in multi-sections
    html_sections = []

    # 1) Title
    html_sections.append(f"""
    <h1>Investor Report for {name}</h1>
    <p><em>Generated: {now}</em></p>
    <p>Sector: {sector}, Stage: {stage}</p>
    <p>Score: {final_score:.1f}</p>
    <hr/>
    """)

    # 2) Disclaimers at top (or bottom, your call)
    html_sections.append(f"""{disclaimers}""")

    # 3) Executive Summary
    html_sections.append(f"""
    <h2>Executive Summary</h2>
    <p><strong>Overall Score</strong>: {final_score:.1f}/100 => {rating_text}</p>
    <ul>
        <li>Success Probability => {sp:.1f}%</li>
        <li>Intangible => {intangible:.1f}</li>
        <li>Team Depth => {team:.1f}</li>
        <li>Moat => {moat:.1f}</li>
    </ul>
    """)

    # 4) Patterns
    html_sections.append(f"""
    <h2>Patterns & Observations</h2>
    {pat_html}
    """)

    # 5) Scenario Modeling
    html_sections.append(f"""
    <h2>Scenario Modeling & System Dynamics</h2>
    {sys_html}
    {sir_html}
    <h3>Scenario Analysis</h3>
    {scenario_html}
    """)

    # 6) Financial Summary
    if include_financials:
        html_sections.append(f"""
        <h2>Financial Summary</h2>
        {cf_html}
        {ue_html}
        <p>Showing financial alignment with intangible momentum. Scenario expansions help plan capital usage & marketing ROI.</p>
        """)

    # 7) Market & Sector
    if include_market:
        html_sections.append(f"""
        <h2>Market & Sector</h2>
        <p>Sector => {sector}, Stage => {stage}.</p>
        <p>Market forces can accelerate or constrain scenario expansions, so watch competitor moves & macro shifts.</p>
        """)

    # 8) Team & Intangible
    if include_team:
        html_sections.append(f"""
        <h2>Team & Intangible Analysis</h2>
        <p>Team Depth => {team:.1f}/100, Moat => {moat:.1f}/100, Intangible => {intangible:.1f}/100.</p>
        <p>A rating that reflects how effectively your pitch resonates, combining clarity & passion signals with scenario logic.</p>
        """)

    # 9) Tech
    if include_tech:
        html_sections.append(f"""
        <h2>Technical Overview</h2>
        <p>Scalable backend is crucial if the scenario modeling projects surging user inflows.
        Align intangible momentum with robust technical readiness to sustain investor confidence.</p>
        """)

    # 10) Extended
    html_sections.append(f"""
    {extended_text}
    """)

    # 11) Optional recs
    if rec_html:
        html_sections.append(rec_html)

    # 12) Optional JSON Appendix
    appendix_html = ""
    if include_appendix:
        safe_doc = make_json_serializable(doc)
        docjson = json.dumps(safe_doc, indent=2)
        # (Optional) Truncate docjson to avoid huge PDF
        if len(docjson) > 20000:
            docjson = docjson[:20000] + "\n... [TRUNCATED to avoid giant PDF] ..."

        appendix_html = f"""
        <div style="page-break-before: always;"></div>
        <h2>Appendix: Full JSON</h2>
        <pre>{docjson}</pre>
        """
        html_sections.append(appendix_html)

    # Combine them
    final_html_body = "\n".join(html_sections)

    # Minimal CSS
    css = """
    <style>
    body {
      font-family: "Arial", sans-serif;
      margin: 15px;
      color: #111;
      line-height: 1.4;
      font-size: 14px;
    }
    h1 { color: #0B3D91; }
    h2 { color: #B22222; margin-top: 1.2rem; }
    h3 { color: #006400; margin-top: 1rem; }
    ul, p { margin-bottom: 0.8rem; }
    hr {
      margin: 1rem 0;
      border: none;
      border-top: 2px solid #aaa;
    }
    pre {
      background: #f3f3f3;
      padding: 10px;
      white-space: pre-wrap;
      border: 1px dashed #999;
    }
    </style>
    """

    html = f"""
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Investor Report</title>
      {css}
    </head>
    <body>
      {final_html_body}
    </body>
    </html>
    """

    # PDFKit config
    options = {
        "page-size": "A4",
        "margin-top": "10mm",
        "margin-right": "10mm",
        "margin-bottom": "10mm",
        "margin-left": "10mm",
        "encoding": "UTF-8"
    }

    system_name = platform.system().lower()
    default_path = "/usr/bin/wkhtmltopdf"
    if system_name == "windows":
        default_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    elif system_name == "darwin":
        default_path = "/usr/local/bin/wkhtmltopdf"

    custom_path = os.environ.get("PDFKIT_PATH", default_path)
    logger.info(f"Using wkhtmltopdf => {custom_path}")

    try:
        config = pdfkit.configuration(wkhtmltopdf=custom_path)
    except OSError as ex:
        logger.error(f"Cannot find wkhtmltopdf => {ex}")
        raise RuntimeError(f"wkhtmltopdf not found => {custom_path}")

    try:
        pdf_data = pdfkit.from_string(html, False, options=options, configuration=config)
        return pdf_data
    except Exception as e:
        logger.error(f"Error generating PDF => {str(e)}")
        error_html = f"<html><body><h1>PDF Error</h1><p>{e}</p></body></html>"
        return pdfkit.from_string(error_html, False, configuration=config)