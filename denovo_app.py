"""
BioDrugDiscover — De Novo Molecule Generation UI
Uses fine-tuned MolT5 (LoRA) model to generate novel molecules from protein pocket descriptions.
Mirrors the style/layout of the main BioDrugDiscover app (app2.py).

Run:
    conda activate ipykernel
    pip install streamlit torch transformers peft rdkit-pypi
    streamlit run denovo_app.py
"""

import streamlit as st
import pandas as pd
import json, os, re
from pathlib import Path
from io import StringIO

# ─── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="BioDrugDiscover — De Novo Gen",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background:#FAFAF9; border-right:1px solid #E5E5E2; }
div[data-testid="metric-container"] {
    background:#F5F5F3; border:0.5px solid #D3D1C7; border-radius:10px; padding:.8rem 1rem;
}
.stButton > button {
    background:#1D9E75 !important; color:white !important; border:none !important;
    border-radius:8px !important; font-weight:500 !important; font-size:14px !important;
}
.stButton > button:hover { background:#157A5A !important; }
.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid #E5E5E2; }
.stTabs [data-baseweb="tab"] { font-size:13px; font-weight:500; color:#888; padding:6px 14px; }
.stTabs [aria-selected="true"] { background:#E1F5EE !important; color:#1D9E75 !important; }
.badge-pass  { display:inline-block; padding:3px 12px; border-radius:20px; background:#E1F5EE;
               color:#1D9E75; border:1px solid #9FE1CB; font-size:12px; font-weight:500; }
.badge-fail  { display:inline-block; padding:3px 12px; border-radius:20px; background:#FCEBEB;
               color:#A32D2D; border:1px solid #F09595; font-size:12px; font-weight:500; }
.badge-gen   { display:inline-block; padding:3px 12px; border-radius:20px; background:#EEEDFE;
               color:#534AB7; border:1px solid #AFA9EC; font-size:12px; font-weight:500; }
.badge-warn  { display:inline-block; padding:3px 12px; border-radius:20px; background:#FEF9E1;
               color:#9E7A1D; border:1px solid #E1CB9F; font-size:12px; font-weight:500; }
.step-ok  { background:#E1F5EE; border:0.5px solid #9FE1CB; border-radius:7px; padding:5px 12px;
            font-size:12px; color:#1D9E75; font-family:'IBM Plex Mono',monospace; margin:3px 0; }
.step-run { background:#EEEDFE; border:0.5px solid #AFA9EC; border-radius:7px; padding:5px 12px;
            font-size:12px; color:#534AB7; font-family:'IBM Plex Mono',monospace; margin:3px 0; }
.step-err { background:#FCEBEB; border:0.5px solid #F09595; border-radius:7px; padding:5px 12px;
            font-size:12px; color:#A32D2D; font-family:'IBM Plex Mono',monospace; margin:3px 0; }
.sec-hdr  { font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.07em;
            color:#888780; margin:1rem 0 .4rem; }
.mol-card { background:#F9FAFB; border:1px solid #E5E5E2; border-radius:12px; padding:1rem;
            margin-bottom:0.75rem; }
.prop-row { display:flex; gap:8px; flex-wrap:wrap; margin-top:0.5rem; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading MolT5 de novo model…")
def load_model(model_path: str):
    """
    Load fine-tuned MolT5 + LoRA weights from disk.
    Returns (tokenizer, model, device) or raises with an error message.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel, PeftConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config    = PeftConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    base      = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model     = PeftModel.from_pretrained(base, model_path).to(device)
    model.eval()
    return tokenizer, model, device


# ══════════════════════════════════════════════════════════════
# MOLECULE UTILITIES
# ══════════════════════════════════════════════════════════════

def clean_smiles(smiles: str) -> str:
    """
    Exact clean_smiles() from the notebook:
    strip whitespace, fix unclosed parentheses (common model hallucination).
    """
    if not smiles:
        return ""
    smiles = smiles.strip().replace(" ", "")
    # Fix parentheses imbalance (common LLM error)
    open_count  = smiles.count("(")
    close_count = smiles.count(")")
    if open_count > close_count:
        smiles += ")" * (open_count - close_count)
    return smiles


def compute_properties(smiles: str) -> dict:
    """Compute QED, Lipinski, PAINS, MW, LogP, HBD, HBA, TPSA, RotBonds."""
    result = {
        "valid": False,
        "smiles": smiles,
        "qed": None,
        "mw": None,
        "logp": None,
        "hbd": None,
        "hba": None,
        "tpsa": None,
        "rot_bonds": None,
        "lipinski": None,
        "pains": None,
        "pains_alerts": [],
        "lipinski_violations": [],
    }
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED, rdMolDescriptors
        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result

        result["valid"] = True
        result["qed"]       = round(QED.qed(mol), 4)
        result["mw"]        = round(Descriptors.MolWt(mol), 2)
        result["logp"]      = round(Descriptors.MolLogP(mol), 3)
        result["hbd"]       = rdMolDescriptors.CalcNumHBD(mol)
        result["hba"]       = rdMolDescriptors.CalcNumHBA(mol)
        result["tpsa"]      = round(rdMolDescriptors.CalcTPSA(mol), 2)
        result["rot_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # Lipinski Rule-of-5
        viols = []
        if result["mw"]   >= 500: viols.append(f"MW={result['mw']} ≥ 500")
        if result["logp"]  >= 5:  viols.append(f"LogP={result['logp']} ≥ 5")
        if result["hbd"]   > 5:   viols.append(f"HBD={result['hbd']} > 5")
        if result["hba"]   > 10:  viols.append(f"HBA={result['hba']} > 10")
        result["lipinski"] = len(viols) == 0
        result["lipinski_violations"] = viols

        # PAINS filter
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        entry   = catalog.GetFirstMatch(mol)
        result["pains"]        = entry is None          # True = PAINS-free
        result["pains_alerts"] = [] if entry is None else [entry.GetDescription()]

    except ImportError:
        result["valid"] = None   # RDKit not available
    except Exception:
        pass

    return result


def generate_molecules(pocket_input: str, tokenizer, model, device,
                        num_candidates: int = 5,
                        temperature: float = 1.1,
                        top_p: float = 0.95,
                        top_k: int = 50) -> list[str]:
    """
    Exact generate_ligand_from_pocket() logic from the notebook.
    Generates `num_candidates` sequences, then returns ALL that are:
      - non-empty
      - unique within this call
      - valid per RDKit
    Falls back to the first raw candidate if none pass.
    """
    import torch
    from rdkit import Chem

    inputs = tokenizer(
        pocket_input,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            do_sample=True,        # must be True for diverse sampling
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=1.2,
            num_return_sequences=num_candidates,
        )

    # Decode all candidates
    raw_candidates = [
        clean_smiles(tokenizer.decode(o, skip_special_tokens=True))
        for o in outputs
    ]

    # Filter: valid SMILES + unique (mirrors notebook's validity+uniqueness check)
    seen      = set()
    valid_out = []
    for smi in raw_candidates:
        if smi and smi not in seen:
            seen.add(smi)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid_out.append(smi)

    # Fallback: if ALL candidates fail RDKit, return the raw first one
    # so the UI can still show something and report "invalid"
    if not valid_out:
        return [raw_candidates[0]] if raw_candidates else []

    return valid_out


# ══════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ══════════════════════════════════════════════════════════════

def badge(label: str, kind: str = "gen") -> str:
    cls = {"pass": "badge-pass", "fail": "badge-fail",
           "warn": "badge-warn", "gen": "badge-gen"}.get(kind, "badge-gen")
    return f'<span class="{cls}">{label}</span>'


def render_molecule_image(smiles: str, size: tuple = (300, 200)) -> bytes | None:
    """
    Render a 2-D molecule depiction as PNG bytes (for st.image).
    Uses MolDraw2DCairo when available (best quality), falls back to
    MolToImage (PIL), then returns None if RDKit is absent.
    Mirrors the Draw.MolsToGridImage / MolDraw2D visualisation from the notebook.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        import io

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Prefer Cairo (vector-quality PNG)
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(*size)
            drawer.drawOptions().addAtomIndices = False
            drawer.drawOptions().padding = 0.15
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            return drawer.GetDrawingText()          # already PNG bytes
        except Exception:
            pass

        # Fallback: PIL image → PNG bytes
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception:
        return None


def render_property_badges(props: dict):
    parts = []
    if props.get("valid") is None:
        parts.append(badge("RDKit unavailable", "warn"))
    elif not props.get("valid"):
        parts.append(badge("Invalid SMILES", "fail"))
        st.markdown('<div class="prop-row">' + " ".join(parts) + "</div>",
                    unsafe_allow_html=True)
        return

    # QED
    qed = props.get("qed")
    if qed is not None:
        kind = "pass" if qed >= 0.6 else ("warn" if qed >= 0.3 else "fail")
        parts.append(badge(f"QED {qed}", kind))

    # Lipinski
    if props.get("lipinski") is True:
        parts.append(badge("Lipinski ✓", "pass"))
    elif props.get("lipinski") is False:
        viols = ", ".join(props.get("lipinski_violations", []))
        parts.append(badge(f"Lipinski ✗ ({viols})", "fail"))

    # PAINS
    if props.get("pains") is True:
        parts.append(badge("PAINS-free ✓", "pass"))
    elif props.get("pains") is False:
        alerts = ", ".join(props.get("pains_alerts", []))
        parts.append(badge(f"PAINS ✗ {alerts}", "fail"))

    # Physicochemical
    for key, label in [("mw", "MW"), ("logp", "LogP"), ("hbd", "HBD"),
                        ("hba", "HBA"), ("tpsa", "TPSA"), ("rot_bonds", "RotBonds")]:
        if props.get(key) is not None:
            parts.append(badge(f"{label} {props[key]}", "gen"))

    st.markdown('<div class="prop-row">' + " ".join(parts) + "</div>",
                unsafe_allow_html=True)


def render_results(results: list[dict]):
    """
    Render history entries.
    Each entry has:
      label          – file name or "Text Input"
      pocket_results – list of {pocket_id, input, smiles, props}
      steps_html     – execution log HTML
    """
    if not results:
        st.info("No generation results yet.")
        return

    for idx, entry in enumerate(results):
        label   = entry.get("label", f"Run #{idx+1}")
        pockets = entry.get("pocket_results", [])

        st.markdown(f"### 💬 Query #{idx + 1}: `{label}`")
        st.caption(f"{len(pockets)} pocket(s) in this query")

        with st.expander("⏳ Execution Steps", expanded=False):
            st.markdown(entry.get("steps_html", ""), unsafe_allow_html=True)

        if not pockets:
            st.warning("No results for this query.")
            continue

        # ── Summary metrics across all pockets ──
        valid_rows  = [r for r in pockets if r["props"].get("valid")]
        lip_pass    = [r for r in valid_rows if r["props"].get("lipinski")]
        pains_free  = [r for r in valid_rows if r["props"].get("pains")]
        qed_vals    = [r["props"]["qed"] for r in valid_rows if r["props"].get("qed") is not None]
        avg_qed     = round(sum(qed_vals) / len(qed_vals), 4) if qed_vals else "—"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Pockets",       len(pockets))
        c2.metric("Valid SMILES",  len(valid_rows))
        c3.metric("Lipinski Pass", len(lip_pass))
        c4.metric("PAINS-Free",    len(pains_free))
        c5.metric("Avg QED",       avg_qed)

        st.divider()

        tab_table, tab_cards, tab_summary = st.tabs(
            ["📊 Table", "🧬 Molecule Cards", "📈 Summary"]
        )

        # ── TABLE ──────────────────────────────────────────────
        with tab_table:
            rows = []
            for r in pockets:
                p = r["props"]
                rows.append({
                    "Pocket":     r["pocket_id"],
                    "SMILES":     r["smiles"],
                    "Valid":      "✅" if p.get("valid") else "❌",
                    "QED":        p.get("qed",      "—"),
                    "MW":         p.get("mw",       "—"),
                    "LogP":       p.get("logp",     "—"),
                    "HBD":        p.get("hbd",      "—"),
                    "HBA":        p.get("hba",      "—"),
                    "TPSA":       p.get("tpsa",     "—"),
                    "RotBonds":   p.get("rot_bonds","—"),
                    "Lipinski":   "✅" if p.get("lipinski") else ("❌" if p.get("lipinski") is False else "—"),
                    "PAINS-Free": "✅" if p.get("pains")    else ("❌" if p.get("pains")    is False else "—"),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_bytes,
                file_name=f"denovo_results_query{idx+1}.csv",
                mime="text/csv",
                key=f"dl_{idx}",
            )

        # ── MOLECULE CARDS ─────────────────────────────────────
        with tab_cards:
            n_cols = min(3, len(pockets))
            cols   = st.columns(n_cols)
            for i, r in enumerate(pockets):
                col = cols[i % n_cols]
                with col:
                    st.markdown('<div class="mol-card">', unsafe_allow_html=True)
                    st.markdown(f"**{r['pocket_id']}**")

                    png = render_molecule_image(r["smiles"], size=(300, 200))
                    if png:
                        st.image(png, use_container_width=True)
                    else:
                        st.warning("Structure rendering unavailable")

                    st.code(r["smiles"] or "— no valid SMILES —", language=None)
                    render_property_badges(r["props"])
                    st.markdown("</div>", unsafe_allow_html=True)

        # ── SUMMARY ────────────────────────────────────────────
        with tab_summary:
            # Grid image of valid molecules (mirrors MolsToGridImage)
            if valid_rows:
                st.markdown('<div class="sec-hdr">🧬 Molecule Grid (Valid Only)</div>',
                            unsafe_allow_html=True)
                try:
                    from rdkit import Chem
                    from rdkit.Chem import Draw
                    import io

                    mols   = [Chem.MolFromSmiles(r["smiles"]) for r in valid_rows]
                    labels = [
                        f"{r['pocket_id']}  QED={r['props'].get('qed','—')}"
                        for r in valid_rows
                    ]
                    per_row  = min(4, len(mols))
                    grid_img = Draw.MolsToGridImage(
                        mols,
                        molsPerRow=per_row,
                        subImgSize=(250, 200),
                        legends=labels,
                    )
                    buf = io.BytesIO()
                    grid_img.save(buf, format="PNG")
                    st.image(buf.getvalue(), use_container_width=True)
                except Exception as e:
                    st.caption(f"Grid image unavailable: {e}")

            # QED bar chart
            if qed_vals:
                st.markdown('<div class="sec-hdr">QED per Pocket</div>', unsafe_allow_html=True)
                qed_df = pd.DataFrame({
                    "Pocket": [r["pocket_id"] for r in valid_rows if r["props"].get("qed") is not None],
                    "QED":    qed_vals,
                })
                st.bar_chart(qed_df.set_index("Pocket"))

            # Filter summary table
            st.markdown('<div class="sec-hdr">Filter Summary</div>', unsafe_allow_html=True)
            summary_data = {
                "Filter":  ["Validity", "Lipinski Ro5", "PAINS-Free"],
                "Pass":    [len(valid_rows), len(lip_pass), len(pains_free)],
                "Fail":    [len(pockets) - len(valid_rows),
                            len(valid_rows) - len(lip_pass),
                            len(valid_rows) - len(pains_free)],
                "Pass %":  [
                    f"{len(valid_rows)/len(pockets)*100:.1f}%",
                    f"{len(lip_pass)/len(valid_rows)*100:.1f}%" if valid_rows else "—",
                    f"{len(pains_free)/len(valid_rows)*100:.1f}%" if valid_rows else "—",
                ],
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧪 BioDrugDiscover")
    st.caption("De Novo Molecule Generation")
    st.divider()

    side_tab = st.radio("Navigation Options",
                        ["🗂️ Objectives", "🏗️ Architecture", "⚙️ Setup"],
                        label_visibility="collapsed")

    if "🗂️" in side_tab:
        st.markdown("**Objective: De Novo Generation**")
        for label, desc in [
            ("Pocket Input",      "Provide protein pocket description as free text or upload a .jsonl file"),
            ("Molecule Gen",      "MolT5 (LoRA fine-tuned, 50 epochs) generates candidate SMILES"),
            ("Validity Check",    "RDKit validates each generated SMILES string"),
            ("QED Score",         "Quantitative Estimate of Drug-likeness (0–1, higher = better)"),
            ("Lipinski Ro5",      "MW<500, LogP<5, HBD≤5, HBA≤10 — oral bioavailability filter"),
            ("PAINS Filter",      "Pan-Assay Interference Compounds — flags promiscuous binders"),
            ("Physicochemical",   "MW, LogP, HBD, HBA, TPSA, Rotatable Bonds"),
        ]:
            st.markdown(
                f'<span class="badge-gen" style="font-size:10px;padding:2px 8px;'
                f'margin:2px 0;display:inline-block">{label}</span> '
                f'<span style="font-size:11px;color:#666">{desc}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

        st.divider()
        st.markdown("**Next Objective →**")
        st.markdown(
            '<span class="badge-gen" style="font-size:10px;padding:2px 8px">ADMET Prediction</span>'
            ' <span style="font-size:11px;color:#666">Absorption, Distribution, Metabolism, Excretion, Toxicity</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="badge-gen" style="font-size:10px;padding:2px 8px">Docking Prep</span>'
            ' <span style="font-size:11px;color:#666">Feed generated molecules into AutoDock Vina</span>',
            unsafe_allow_html=True,
        )

    elif "🏗️" in side_tab:
        for icon, title, desc, env in [
            ("🧬", "MolT5 (LoRA)",
             "laituan245/molt5-base-smiles2caption fine-tuned 50 epochs on pocket→SMILES JSONL data.",
             "Python 3.10 / transformers + peft"),
            ("🔬", "RDKit",
             "Validates SMILES, computes QED, Lipinski, PAINS, MW, LogP, TPSA, RotBonds.",
             "rdkit-pypi"),
            ("📂", "Pocket Input",
             "Free-text pocket description or .jsonl (one JSON per line with 'input' key).",
             "File-based or text input"),
            ("📊", "Results",
             "Table + molecule cards + filter summary for each generation run.",
             "Session history"),
        ]:
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            st.code(env, language=None)
            st.markdown("")

    else:  # Setup
        for i, (title, cmd) in enumerate([
            ("Install deps",
             "pip install streamlit torch transformers peft rdkit-pypi"),
            ("Set model path",
             "# Edit MODEL_PATH in the sidebar input\n# Default: /home/icmr-proj/final_ligand_model_50_epoch_2"),
            ("Pocket JSONL format",
             '{"input": "protein pocket description", "output": "SMILES"}\n{"input": "...", "output": "..."}'),
            ("Run app",
             "streamlit run denovo_app.py"),
        ], 1):
            st.markdown(f"**{i}. {title}**")
            st.code(cmd, language="bash" if "pip" in cmd or "streamlit" in cmd else "json")

    st.divider()
    # Model path control — persisted in session
    if "model_path" not in st.session_state:
        st.session_state.model_path = "/home/icmr-proj/final_ligand_model_50_epoch_2"

    st.session_state.model_path = st.text_input(
        "Model Path",
        value=st.session_state.model_path,
        help="Path to your fine-tuned MolT5 LoRA checkpoint directory",
    )

    if st.button("🗑️ Clear History"):
        st.session_state.gen_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════

st.markdown("## 🧪 De Novo Molecule Generation")
st.caption("Generate novel drug-like molecules conditioned on protein pocket descriptions using fine-tuned MolT5.")

# ─── Session state ────────────────────────────────────────────
if "gen_history" not in st.session_state:
    st.session_state.gen_history = []

# ─── Input Panel ──────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Pocket Input</div>', unsafe_allow_html=True)

input_mode = st.radio("Input Mode",
                       ["✏️ Text Input", "📄 Upload .jsonl File"],
                       horizontal=True,
                       label_visibility="collapsed")

# Each job = one "query" (one button press / one file upload).
# job = {"label": str, "pockets": [{"pocket_id": str, "input": str}]}
gen_jobs: list[dict] = []

# shared generation params (defined once, used by both modes)
num_cands   = 5
temperature = 1.1
top_p       = 0.95

if "✏️" in input_mode:
    with st.form("gen_form"):
        pocket_text = st.text_area(
            "Pocket Description",
            placeholder=(
                "Paste a pocket description, e.g.:\n"
                "pocket_residues: ALA12 GLY45 LEU89 ... binding_site_volume: 412 A³"
            ),
            height=120,
            label_visibility="collapsed",
        )
        col_run, col_opts = st.columns([1, 3])
        with col_run:
            submitted = st.form_submit_button("▶ Generate Molecules",
                                               use_container_width=True)
        with col_opts:
            col_n, col_t, col_p = st.columns(3)
            num_cands   = col_n.number_input("Candidates", min_value=1, max_value=20, value=5)
            temperature = col_t.number_input("Temperature", min_value=0.5, max_value=2.0, value=1.1, step=0.05)
            top_p       = col_p.number_input("Top-p", min_value=0.1, max_value=1.0, value=0.95, step=0.05)

    if submitted and pocket_text.strip():
        # Single text input → one pocket, one query
        gen_jobs = [{
            "label":   "Text Input",
            "pockets": [{"pocket_id": "Pocket 0", "input": pocket_text.strip()}],
        }]

else:
    uploaded = st.file_uploader(
        "Upload .jsonl file — each line: {\"input\": \"...\"}  (all pockets treated as one query)",
        type=["jsonl", "json"],
        label_visibility="collapsed",
    )

    col_run2, col_opts2 = st.columns([1, 3])
    with col_opts2:
        col_n2, col_t2, col_p2 = st.columns(3)
        num_cands   = col_n2.number_input("Candidates per pocket", min_value=1, max_value=20, value=5, key="nc2")
        temperature = col_t2.number_input("Temperature", min_value=0.5, max_value=2.0, value=1.1,
                                           step=0.05, key="t2")
        top_p       = col_p2.number_input("Top-p", min_value=0.1, max_value=1.0, value=0.95,
                                           step=0.05, key="p2")

    max_rows = st.number_input("Max pockets from file", min_value=1, max_value=200, value=10)
    run_file = col_run2.button("▶ Generate from File", use_container_width=True)

    if run_file and uploaded is not None:
        content = uploaded.read().decode("utf-8", errors="replace")
        lines   = [l.strip() for l in content.splitlines() if l.strip()]
        pockets = []
        for i, line in enumerate(lines[:max_rows]):
            try:
                obj = json.loads(line)
                inp = obj.get("input", "")
                if inp:
                    pockets.append({"pocket_id": f"Pocket {i}", "input": inp})
            except Exception:
                pass
        if not pockets:
            st.warning("No valid 'input' keys found in the uploaded file.")
        else:
            # ALL pockets from this file → ONE single query / job
            gen_jobs = [{
                "label":   uploaded.name,
                "pockets": pockets,
            }]
    elif run_file and uploaded is None:
        st.warning("Please upload a .jsonl file first.")


# ─── Generation ───────────────────────────────────────────────
if gen_jobs:
    model_path = st.session_state.model_path.strip()
    if not model_path or not Path(model_path).exists():
        st.error(
            f"Model path not found: `{model_path}`\n\n"
            "Update the **Model Path** in the sidebar to point to your fine-tuned MolT5 checkpoint."
        )
        st.stop()

    try:
        tokenizer, model, device = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    for job in gen_jobs:
        log_steps = []

        def log(msg, status="ok"):
            icon = {"ok": "✅", "run": "⏳", "err": "❌"}.get(status, "✅")
            cls  = {"ok": "step-ok", "run": "step-run", "err": "step-err"}.get(status, "step-ok")
            log_steps.append(f'<div class="{cls}">{icon} {msg}</div>')

        log(f"Query: {job['label']}  ·  Pockets: {len(job['pockets'])}", "run")
        log(f"Device: {device}  ·  Candidates/pocket: {num_cands}  ·  Temp: {temperature}  ·  Top-p: {top_p}")

        # Generate one SMILES per pocket (all under one job/query)
        pocket_results = []   # list of {pocket_id, input, smiles, props}

        for pocket in job["pockets"]:
            pid   = pocket["pocket_id"]
            pinpt = pocket["input"]
            log(f"  Generating for {pid}…", "run")

            with st.spinner(f"Generating for {pid} ({job['label']})…"):
                try:
                    smiles_list = generate_molecules(
                        pinpt, tokenizer, model, device,
                        num_candidates=int(num_cands),
                        temperature=float(temperature),
                        top_p=float(top_p),
                    )
                    # Take the best (first valid) SMILES for this pocket
                    best_smi = smiles_list[0] if smiles_list else ""
                    log(f"    {pid} → {best_smi[:50] if best_smi else 'no output'}")
                except Exception as e:
                    log(f"    {pid} generation error: {e}", "err")
                    best_smi = ""

            props = compute_properties(best_smi) if best_smi else {"valid": False}
            status_str = "valid" if props.get("valid") else "invalid"
            log(f"    {pid} SMILES: {status_str}", "ok" if props.get("valid") else "err")

            pocket_results.append({
                "pocket_id": pid,
                "input":     pinpt,
                "smiles":    best_smi,
                "props":     props,
            })

        valid_n = sum(1 for r in pocket_results if r["props"].get("valid"))
        qed_vals = [r["props"]["qed"] for r in pocket_results if r["props"].get("qed") is not None]
        avg_qed  = round(sum(qed_vals) / len(qed_vals), 4) if qed_vals else "—"
        log(f"  Complete — Valid: {valid_n}/{len(pocket_results)}  Avg QED: {avg_qed}")

        st.session_state.gen_history.append({
            "label":          job["label"],
            "pocket_results": pocket_results,
            "steps_html":     "".join(log_steps),
        })


# ─── Render Results ───────────────────────────────────────────
if not st.session_state.gen_history:
    st.markdown("""
    <div style="margin-top:2.5rem;color:#888;font-size:14px;text-align:center">
        Enter a pocket description above (or upload a .jsonl file) and click <strong>▶ Generate Molecules</strong>.<br>
        Generated molecules with QED, Lipinski, PAINS, and other properties appear here.
    </div>""", unsafe_allow_html=True)
else:
    render_results(st.session_state.gen_history)
