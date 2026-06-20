"""
BioDrugDiscover — Streamlit UI
Wired directly to your real pipeline functions (no mock data).

Place this file in the SAME folder as orchestrator.ipynb and run_obj2_agent.py.

Run:
    conda activate ipykernel
    pip install streamlit nbformat
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json, os, re, sys
from pathlib import Path

# ─── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="BioDrugDiscover",
    page_icon="⚗️",
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
    background:#534AB7 !important; color:white !important; border:none !important;
    border-radius:8px !important; font-weight:500 !important; font-size:14px !important;
}
.stButton > button:hover { background:#3C3489 !important; }
.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid #E5E5E2; }
.stTabs [data-baseweb="tab"] { font-size:13px; font-weight:500; color:#888; padding:6px 14px; }
.stTabs [aria-selected="true"] { background:#EEEDFE !important; color:#534AB7 !important; }
.badge-kg   { display:inline-block; padding:3px 12px; border-radius:20px; background:#E1F5EE;
              color:#1D9E75; border:1px solid #9FE1CB; font-size:12px; font-weight:500; }
.badge-str  { display:inline-block; padding:3px 12px; border-radius:20px; background:#E6F1FB;
              color:#185FA5; border:1px solid #B5D4F4; font-size:12px; font-weight:500; }
.badge-pipe { display:inline-block; padding:3px 12px; border-radius:20px; background:#EEEDFE;
              color:#534AB7; border:1px solid #AFA9EC; font-size:12px; font-weight:500; }
.step-ok  { background:#E1F5EE; border:0.5px solid #9FE1CB; border-radius:7px; padding:5px 12px;
            font-size:12px; color:#1D9E75; font-family:'IBM Plex Mono',monospace; margin:3px 0; }
.step-run { background:#EEEDFE; border:0.5px solid #AFA9EC; border-radius:7px; padding:5px 12px;
            font-size:12px; color:#534AB7; font-family:'IBM Plex Mono',monospace; margin:3px 0; }
.step-err { background:#FCEBEB; border:0.5px solid #F09595; border-radius:7px; padding:5px 12px;
            font-size:12px; color:#A32D2D; font-family:'IBM Plex Mono',monospace; margin:3px 0; }
.sec-hdr  { font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:.07em;
            color:#888780; margin:1rem 0 .4rem; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD PIPELINE — exec the orchestrator notebook cells into ns
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading pipeline (KG + agents)…")
def load_pipeline():
    """
    Execute orchestrator.ipynb code cells into a shared namespace.
    This gives us: run(), classify_route(), _run_structure_agent_subprocess(),
    kg_agent, orchestrator, and all KG query functions.
    """
    import nbformat

    # Locate orchestrator notebook
    base = Path(__file__).parent
    nb_candidates = [
        base / "orchestrator.ipynb",
    ]
    nb_path = next((p for p in nb_candidates if p.exists()), None)
    if nb_path is None:
        return {"error": "orchestrator.ipynb not found next to app.py"}

    ns = {"__name__": "__main__"}

    # Read notebook with UTF-8
    with open(nb_path, encoding="utf-8", errors="replace") as f:
        nb = nbformat.read(f, as_version=4)

    skip_prefixes = ("%pip", "! ", "run(", "display(", "HTML(", "Markdown(")

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source.strip()
        if not src:
            continue
        # Skip install / demo execution cells
        if any(src.startswith(p) for p in skip_prefixes):
            continue
        # Skip cells that are purely run() calls (demo cells at bottom)
        if re.match(r'^run\(', src) and src.count('\n') < 2:
            continue
        try:
            exec(compile(src, str(nb_path), "exec"), ns)
        except Exception as e:
            # Non-fatal: some cells may fail if deps missing; continue
            ns.setdefault("_load_warnings", []).append(f"{type(e).__name__}: {e}")

    # Verify key functions loaded
    required = ["run_kg_agent", "classify_route", "_run_structure_agent_subprocess",
                "kg_agent", "detect_intent"]
    missing = [r for r in required if r not in ns]
    if missing:
        return {"error": f"Missing functions after load: {missing}",
                "warnings": ns.get("_load_warnings", [])}

    return ns


# ══════════════════════════════════════════════════════════════
# FILE HELPERS — read real output files
# ══════════════════════════════════════════════════════════════

BASE = Path(__file__).parent

def read_json(path):
    p = BASE / path
    if p.exists():
        with open(p, encoding="utf-8", errors="replace") as f:
            return json.load(f)
    return None

def find_csv(pattern):
    for p in BASE.glob(pattern):
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


# ══════════════════════════════════════════════════════════════
# PIPELINE FLOW DIAGRAM
# ══════════════════════════════════════════════════════════════

def pipeline_diagram(route: str = ""):
    active = {
        "kg":        ["query", "orchestrator", "kg", "output"],
        "structure": ["query", "orchestrator", "structure", "output"],
        "full":      ["query", "orchestrator", "kg", "structure", "output"],
    }.get(route, [])

    nodes = [
        ("query",        "🔍", "Query"),
        ("orchestrator", "🎛️", "Orchestrator"),
        ("kg",           "🧬", "KG Agent"),
        ("structure",    "🔬", "Structure"),
        ("output",       "📊", "Results"),
    ]

    parts, x = [], 0
    for nid, icon, label in nodes:
        is_active = nid in active or not route
        fill   = "#EEEDFE" if is_active else "#F5F5F3"
        stroke = "#534AB7" if is_active else "#D3D1C7"
        sw     = "2"   if is_active else "0.5"
        tc     = "#534AB7" if is_active else "#888780"
        op     = "1"   if (is_active or not route) else "0.3"

        parts.append(f"""
          <g transform="translate({x},0)" opacity="{op}">
            <circle cx="26" cy="26" r="26" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>
            <text x="26" y="34" text-anchor="middle" font-size="18">{icon}</text>
            <text x="26" y="66" text-anchor="middle" font-size="10" fill="{tc}"
                  font-family="Inter,sans-serif" font-weight="500">{label}</text>
          </g>""")

        if nid != "output":
            lc = "#534AB7" if is_active else "#D3D1C7"
            parts.append(
                f'<line x1="{x+52}" y1="26" x2="{x+70}" y2="26" '
                f'stroke="{lc}" stroke-width="1.5"/>'
            )
            x += 70

    svg = (f'<svg viewBox="0 0 {x+56} 80" xmlns="http://www.w3.org/2000/svg" '
           f'style="width:100%;max-width:560px;margin:0 auto;display:block;">'
           + "".join(parts) + "</svg>")
    st.markdown(svg, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# RESULT RENDERERS  (with structural type-safety fixes)
# ══════════════════════════════════════════════════════════════

# ── KG: DISEASE_TO_TARGET ────────────────────────────────────
def render_targets(kg_results: dict):
    ranked = kg_results.get("ranked_targets", {})
    if not ranked:
        st.warning("No ranked target data in results.")
        return

    has_methods = "method_3_structural_druggability" in ranked

    if has_methods:
        consensus = ranked.get("consensus_targets", [])
        if consensus:
            st.markdown('<div class="sec-hdr">⭐ Consensus Targets — top-5 in both ranking methods</div>',
                        unsafe_allow_html=True)
            rows = [{"Gene":            t.get("gene_name", ""),
                     "UniProt":         t.get("protein_id", ""),
                     "PDB Rank":        t.get("rank_pdb", ""),
                     "PubMed Rank":     t.get("rank_pubmed", ""),
                     "Structural Score":t.get("structural_score", ""),
                     "PDB Count":       t.get("pdb_count", ""),
                     "PubMed Count":    t.get("pubmed_count", "")}
                    for t in consensus]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        tab1, tab2 = st.tabs(["📐 Structural Druggability (RCSB PDB)",
                               "📚 Literature Evidence (PubMed)"])
        with tab1:
            st.caption("Score: 4=X-ray+drug ligand · 3=X-ray+any ligand · 2=X-ray only · 1=AlphaFold · 0=none")
            m3 = ranked.get("method_3_structural_druggability", [])
            if m3:
                rows = [{"Rank":              t.get("rank_pdb", i+1),
                         "Gene":              t.get("gene_name", ""),
                         "UniProt":           t.get("protein_id", ""),
                         "Protein Name":      t.get("protein_name", ""),
                         "Structural Score":  t.get("structural_score", ""),
                         "PDB Count":         t.get("pdb_count", ""),
                         "Best Res (Å)":      str(t.get("best_resolution")) if t.get("best_resolution") is not None else "N/A",
                         "Drug Ligand":       "✅" if t.get("has_drug_ligand") else "—"}
                        for i, t in enumerate(m3)]
                
                df_m3 = pd.DataFrame(rows)
                # Fix: Explicitly force "Best Res (Å)" to string format to handle combined string ('N/A') and floats seamlessly inside Arrow engine
                if "Best Res (Å)" in df_m3.columns:
                    df_m3["Best Res (Å)"] = df_m3["Best Res (Å)"].astype(str)
                st.dataframe(df_m3, use_container_width=True, hide_index=True)
                
        with tab2:
            st.caption("Score: 4=>100 papers · 3=21-100 · 2=6-20 · 1=1-5 · 0=none")
            m4 = ranked.get("method_4_literature_evidence", [])
            if m4:
                rows = [{"Rank":             t.get("rank_pubmed", i+1),
                         "Gene":             t.get("gene_name", ""),
                         "UniProt":          t.get("protein_id", ""),
                         "Protein Name":     t.get("protein_name", ""),
                         "Literature Score": t.get("literature_score", ""),
                         "PubMed Count":     t.get("pubmed_count", "")}
                        for i, t in enumerate(m4)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        tabs = st.tabs(["Degree", "PageRank", "Proximity", "Pathway"])
        method_keys = ["degree", "pagerank", "proximity", "pathway"]
        for tab, key in zip(tabs, method_keys):
            with tab:
                targets = ranked.get(key, [])
                if not targets:
                    st.info(f"No {key} ranking data.")
                    continue
                rows = [{"#":            i+1,
                         "Gene":         t.get("gene_name", ""),
                         "UniProt":      t.get("protein_id", ""),
                         "Protein Name": t.get("protein_name", "")}
                        for i, t in enumerate(targets)]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── KG: DISEASE_TO_PATHWAY / TARGET_TO_PATHWAY ───────────────
def render_pathways(kg_results: dict):
    pathways = kg_results.get("pathways", [])
    if not pathways:
        st.warning("No pathways found.")
        return
    rows = [{"#":           i+1,
             "Pathway Name":p.get("pathway_name", p.get("pathway_id", "")),
             "Source":      p.get("source", ""),
             "Pathway ID":  p.get("pathway_id", "")}
            for i, p in enumerate(pathways)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── KG: DISEASE_PPI_HUBS ─────────────────────────────────────
def render_ppi_hubs(kg_results: dict, ns: dict):
    hubs = kg_results.get("hub_proteins", [])
    if not hubs:
        st.warning("No hub proteins found.")
        return
    kg = ns.get("knowledge_graph")
    rows = []
    for i, h in enumerate(hubs):
        pid = h.get("protein_id", "")
        gene = h.get("gene_name", "")
        if not gene and kg and pid in kg.nodes:
            gene = kg.nodes[pid].get("gene_name", pid)
        rows.append({"#":           i+1,
                     "Gene":        gene or pid,
                     "UniProt":     pid,
                     "Centrality":  round(h.get("score", 0), 5)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── KG: DISEASE_PPI_NETWORK ──────────────────────────────────
def render_ppi_network(kg_results: dict, ns: dict):
    summary  = kg_results.get("ppi_network_summary", {})
    proteins = kg_results.get("ppi_network_proteins", [])
    c1, c2 = st.columns(2)
    c1.metric("Proteins in network", summary.get("num_proteins", len(proteins)))
    c2.metric("Interactions",         summary.get("num_interactions", "—"))

    disease_id = kg_results.get("disease", {}).get("disease_id", "")
    ppi_html = BASE / f"ppi_{disease_id}.html"
    if ppi_html.exists():
        with open(ppi_html, encoding="utf-8", errors="replace") as f:
            st.components.v1.html(f.read(), height=700, scrolling=True)
    elif proteins:
        kg = ns.get("knowledge_graph")
        rows = [{"Gene":    kg.nodes[p].get("gene_name", p) if kg and p in kg.nodes else p,
                 "UniProt": p}
                for p in proteins]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── KG: DRUG_TO_TARGET ───────────────────────────────────────
def render_drug_to_targets(kg_results: dict):
    targets = kg_results.get("targets", [])
    drug    = kg_results.get("drug", {})
    if drug:
        st.caption(f"Drug: **{drug.get('drug_name', drug.get('drug_id', ''))}**")
    if not targets:
        st.warning("No targets found for this drug.")
        return
    rows = [{"#":               i+1,
             "Gene":            t.get("gene_name", ""),
             "UniProt":         t.get("protein_id", ""),
             "Protein Name":    t.get("protein_name", ""),
             "Affinity Value":  t.get("affinity_value", ""),
             "Confidence Score":t.get("confidence_score", "")}
            for i, t in enumerate(targets)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── KG: MOLECULE_TO_TARGET ───────────────────────────────────
def render_molecule_to_targets(kg_results: dict):
    targets = kg_results.get("targets", [])
    mol     = kg_results.get("molecule", {})
    if mol:
        st.caption(f"Molecule: `{mol.get('molecule_id', '')}`")
    if not targets:
        st.warning("No targets found for this molecule.")
        return
    rows = [{"#":           i+1,
             "Gene":        t.get("gene_name", ""),
             "UniProt":     t.get("protein_id", ""),
             "Protein Name":t.get("protein_name", "")}
            for i, t in enumerate(targets)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── KG: CROSS_DISEASE_PATHWAYS ───────────────────────────────
def render_cross_disease_pathways(kg_results: dict):
    diseases = kg_results.get("diseases", [])
    shared   = kg_results.get("shared_pathways", [])
    total    = kg_results.get("total_shared", len(shared))
    st.caption(f"Diseases compared: {', '.join(str(d) for d in diseases)} · Shared: {total}")
    if not shared:
        st.warning("No shared pathways found.")
        return
    rows = [{"#":             i+1,
             "Pathway Name":  p.get("pathway_name", ""),
             "Source":        p.get("source", ""),
             "Shared by (N)": p.get("shared_count", ""),
             "Diseases":      str(p.get("shared_diseases", ""))}
            for i, p in enumerate(shared)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── KG: PATHWAY_HUB_ANALYSIS ─────────────────────────────────
def render_pathway_hub_analysis(kg_results: dict):
    hubs  = kg_results.get("pathway_hubs", [])
    total = kg_results.get("total_pathways_checked", "")
    if total:
        st.caption(f"Total pathways checked: {total}")
    if not hubs:
        st.warning("No pathway hub data.")
        return
    rows = [{"#":           i+1,
             "Pathway Name":p.get("pathway_name", ""),
             "Source":      p.get("source", ""),
             "Diseases":    p.get("disease_count", ""),
             "Proteins":    p.get("protein_count", ""),
             "Hub Score":   p.get("hub_score", "")}
            for i, p in enumerate(hubs)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── KG: PATHWAY_TO_DISEASES / PATHWAY_DISEASE_BURDEN ─────────
def render_pathway_to_diseases(kg_results: dict):
    pathway  = kg_results.get("pathway", {})
    diseases = kg_results.get("diseases", [])
    count    = kg_results.get("count", len(diseases))
    st.caption(f"Pathway: **{pathway.get('pathway_name', '')}** · {count} diseases")
    if not diseases:
        return
    rows = [{"#":           i+1,
             "Disease Name":d.get("disease_name", d) if isinstance(d, dict) else d}
            for i, d in enumerate(diseases)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── KG: PATHWAY_LITERATURE_VALIDATION ────────────────────────
def render_pathway_literature(kg_results: dict):
    pathways = kg_results.get("pathways", [])
    strong   = kg_results.get("strongly_validated", [])
    not_val  = kg_results.get("not_validated", [])
    c1, c2, c3 = st.columns(3)
    c1.metric("Total pathways",     len(pathways))
    c2.metric("Strongly validated", len(strong))
    c3.metric("Not validated",      len(not_val))
    if not pathways:
        return
    rows = [{"#":               i+1,
             "Pathway Name":    p.get("pathway_name", ""),
             "Source":          p.get("source", ""),
             "Publications":    p.get("combined_pub_count", ""),
             "Literature Score": p.get("literature_score", ""),
             "Validation":      p.get("validation", "")}
            for i, p in enumerate(pathways)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Structure: pocket table ───────────────────────────────────
def render_structure_summary(struct_output: dict):
    pdb_id  = struct_output.get("pdb_id", "")
    uniprot = struct_output.get("uniprot_id", "")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("UniProt",           uniprot or "—")
    c2.metric("PDB ID",            pdb_id  or "—")
    c3.metric("Pockets detected",  struct_output.get("pockets_detected",  "—"))
    c4.metric("Ligands retrieved", struct_output.get("ligands_retrieved", "—"))

    pockets_df = None
    if pdb_id:
        pockets_df = find_csv(f"pockets/{pdb_id}_clean_pockets.csv")
    if pockets_df is not None and not pockets_df.empty:
        st.markdown('<div class="sec-hdr">🔍 Detected Binding Pockets (P2Rank)</div>',
                    unsafe_allow_html=True)
        st.dataframe(pockets_df, use_container_width=True, hide_index=True)


# ── Structure: ligands + docking ─────────────────────────────
def render_ligands_docking(struct_output: dict):
    pdb_id  = struct_output.get("pdb_id", "")
    uniprot = struct_output.get("uniprot_id", "")

    ranked_df = find_csv(f"final_results/{pdb_id}_ranked_results.csv") if pdb_id else None
    best_df   = find_csv(f"final_results/{pdb_id}_best_per_pocket.csv") if pdb_id else None
    ligand_df = find_csv(f"ligand_outputs/ligands_{uniprot}.csv")       if uniprot else None
    explanations = read_json(f"final_results/{pdb_id}_pocket_explanations.json") if pdb_id else None

    if ranked_df is not None:
        valid = ranked_df.dropna(subset=["docking_score"]) if "docking_score" in ranked_df.columns else ranked_df
        c1, c2, c3 = st.columns(3)
        c1.metric("Total docking runs",  len(ranked_df))
        c2.metric("Successful docks",    len(valid))
        if not valid.empty and "docking_score" in valid.columns:
            c3.metric("Best score (kcal/mol)", round(valid["docking_score"].min(), 2))

    if best_df is not None and not best_df.empty:
        st.markdown('<div class="sec-hdr">🏆 Best Ligand per Pocket</div>', unsafe_allow_html=True)
        show_df = best_df.copy()
        if (ligand_df is not None
                and "ChEMBL_ID" in best_df.columns
                and "ChEMBL_ID" in ligand_df.columns):
            props = [c for c in ["ChEMBL_ID","MolWt","LogP","HBD","HBA"] if c in ligand_df.columns]
            show_df = show_df.merge(ligand_df[props], on="ChEMBL_ID", how="left")
        cols_order = [c for c in ["pocket","ChEMBL_ID","docking_score","SMILES","MolWt","LogP","HBD","HBA"]
                      if c in show_df.columns]
        st.dataframe(show_df[cols_order], use_container_width=True, hide_index=True)

    if ranked_df is not None and not ranked_df.empty:
        st.markdown('<div class="sec-hdr">📊 All Docking Results — Ranked by Score</div>',
                    unsafe_allow_html=True)
        if "pocket" in ranked_df.columns:
            for pocket in sorted(ranked_df["pocket"].unique()):
                pocket_df = ranked_df[ranked_df["pocket"] == pocket].copy()
                if "docking_score" in pocket_df.columns:
                    pocket_df = pocket_df.dropna(subset=["docking_score"]).sort_values("docking_score")
                    best = round(pocket_df["docking_score"].iloc[0], 2) if not pocket_df.empty else "—"
                else:
                    best = "—"
                with st.expander(f"🔬 {pocket}  ·  {len(pocket_df)} ligands  ·  best: {best} kcal/mol"):
                    cols_show = [c for c in ["ChEMBL_ID","docking_score","SMILES","similarity_score"]
                                 if c in pocket_df.columns]
                    st.dataframe(pocket_df[cols_show], use_container_width=True, hide_index=True)
                    if explanations and pocket in explanations:
                        exp = explanations[pocket]
                        st.markdown("**LLM Binding Explanation**")
                        st.info(exp.get("explanation", ""))
        else:
            cols_show = [c for c in ["ChEMBL_ID","docking_score","SMILES"] if c in ranked_df.columns]
            st.dataframe(ranked_df[cols_show], use_container_width=True, hide_index=True)

    if ligand_df is not None and not ligand_df.empty:
        with st.expander(f"💊 All Retrieved Ligands ({len(ligand_df)}) — Physicochemical Properties"):
            st.dataframe(ligand_df, use_container_width=True, hide_index=True)

    if ranked_df is None and best_df is None:
        st.info("Docking not yet run, or final_results/ files not found. Run the structure agent first.")


# ══════════════════════════════════════════════════════════════
# MAIN DISPATCHER
# ══════════════════════════════════════════════════════════════

def dispatch(intent: str, kg_results: dict, struct_output, explanation: str, ns: dict):
    """Render the correct tables based on intent."""

    if explanation:
        with st.expander("🧠 LLM Explanation", expanded=True):
            st.markdown(explanation)

    if intent == "DISEASE_TO_TARGET":
        render_targets(kg_results)

    elif intent in ("DISEASE_TO_PATHWAY", "MOLECULE_TO_PATHWAY"):
        render_pathways(kg_results)

    elif intent == "TARGET_TO_PATHWAY":
        render_pathways(kg_results)

    elif intent == "DISEASE_PPI_NETWORK":
        render_ppi_network(kg_results, ns)

    elif intent == "DISEASE_PPI_HUBS":
        render_ppi_hubs(kg_results, ns)

    elif intent == "DRUG_TO_TARGET":
        render_drug_to_targets(kg_results)

    elif intent in ("MOLECULE_TO_TARGET", "MOLECULE_TO_DISEASE"):
        render_molecule_to_targets(kg_results)

    elif intent == "CROSS_DISEASE_PATHWAYS":
        render_cross_disease_pathways(kg_results)

    elif intent == "PATHWAY_HUB_ANALYSIS":
        render_pathway_hub_analysis(kg_results)

    elif intent in ("PATHWAY_TO_DISEASES", "PATHWAY_DISEASE_BURDEN"):
        render_pathway_to_diseases(kg_results)

    elif intent == "PATHWAY_LITERATURE_VALIDATION":
        render_pathway_literature(kg_results)

    elif intent in ("TARGET_STRUCTURE", "TARGET_POCKET", "TARGET_DRUGGABILITY"):
        if struct_output and "error" not in struct_output:
            render_structure_summary(struct_output)
        else:
            st.info("No structure output found. Run the structure agent.")

    elif intent in ("TARGET_LIGAND", "TARGET_DOCKING",
                    "TARGET_VIRTUAL_SCREENING", "TARGET_TO_LIGAND_DISCOVERY"):
        if struct_output and "error" not in struct_output:
            render_structure_summary(struct_output)
            st.divider()
            render_ligands_docking(struct_output)
        else:
            st.info("No docking output found. Run the structure agent.")

    elif intent in ("DISEASE_DRUGGABLE_TARGETS", "FULL_PIPELINE"):
        if kg_results:
            st.markdown('<div class="sec-hdr">Phase 1 — Knowledge Graph Targets</div>',
                        unsafe_allow_html=True)
            render_targets(kg_results)

        if struct_output:
            st.divider()
            if isinstance(struct_output, dict) and not struct_output.get("pdb_id"):
                for uid, so in struct_output.items():
                    if "error" in so:
                        continue
                    st.markdown(f'<div class="sec-hdr">Phase 2 — Structure: {uid}</div>',
                                unsafe_allow_html=True)
                    render_structure_summary(so)
                    st.divider()
                    render_ligands_docking(so)
            else:
                st.markdown('<div class="sec-hdr">Phase 2 — Structure & Docking</div>',
                            unsafe_allow_html=True)
                render_structure_summary(struct_output)
                st.divider()
                render_ligands_docking(struct_output)

    else:
        if kg_results:
            st.json(kg_results)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚗️ BioDrugDiscover")
    st.caption("Multi-agent biomedical pipeline")
    st.divider()

    # Fix Accessibility: Provide explicit string label parameter names instead of an empty space ""
    side_tab = st.radio("Navigation Options", ["🗂️ Intents", "🏗️ Architecture", "⚙️ Setup"],
                        label_visibility="collapsed")

    if "🗂️" in side_tab:
        for group, badge, items in [
            ("Knowledge Graph Agent", "badge-kg", [
                "DISEASE_TO_TARGET","DISEASE_TO_PATHWAY","TARGET_TO_PATHWAY",
                "DRUG_TO_TARGET","MOLECULE_TO_TARGET","MOLECULE_TO_PATHWAY",
                "MOLECULE_TO_DISEASE","DISEASE_PPI_NETWORK","DISEASE_PPI_HUBS",
                "CROSS_DISEASE_PATHWAYS","PATHWAY_TO_DISEASES","PATHWAY_HUB_ANALYSIS",
                "PATHWAY_DISEASE_BURDEN","PATHWAY_LITERATURE_VALIDATION",
            ]),
            ("Structure Agent", "badge-str", [
                "TARGET_STRUCTURE","TARGET_POCKET","TARGET_DRUGGABILITY",
                "TARGET_LIGAND","TARGET_DOCKING","TARGET_VIRTUAL_SCREENING",
            ]),
            ("Integrated Pipeline", "badge-pipe", [
                "DISEASE_DRUGGABLE_TARGETS","FULL_PIPELINE","TARGET_TO_LIGAND_DISCOVERY",
            ]),
        ]:
            st.markdown(f"**{group}**")
            for intent in items:
                st.markdown(
                    f'<span class="{badge}" style="font-size:10px;padding:2px 8px;'
                    f'margin:2px 0;display:inline-block">{intent}</span>',
                    unsafe_allow_html=True,
                )

    elif "🏗️" in side_tab:
        for icon, title, desc, env in [
            ("🎛️","Orchestrator",
             "LangGraph state machine. Normalises query, detects intent, routes to KG or Structure agent.",
             "Python 3.8.6 / ipykernel"),
            ("🧬","KG Agent (Obj-1)",
             "NetworkX KG · Ollama llama3:8b · rapidfuzz entity correction · writes shared/kg_output.json",
             "Python 3.8.6 / ipykernel"),
            ("🔬","Structure Agent (Obj-2)",
             "RCSB/AlphaFold download · P2Rank pockets · ChEMBL ligands · Meeko · AutoDock Vina · LLM explanations",
             "Python 3.10 / meeko_env"),
            ("📂","Shared Memory",
             "shared/kg_output.json and shared/structure_output.json bridge the two agents.",
             "File-based IPC"),
        ]:
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            st.code(env, language=None)
            st.markdown("")

    else:  # Setup
        for i, (title, cmd) in enumerate([
            ("Start Ollama",                 "ollama serve"),
            ("Pull model",                   "ollama pull llama3:8b"),
            ("Install Streamlit deps",       "pip install streamlit nbformat"),
            ("Set MEEKO_PYTHON path",        '# in orchestrator.ipynb Cell A:\nMEEKO_PYTHON = r"C:\\...\\meeko_env\\python.exe"'),
            ("Set P2RANK & Vina paths",      "# in run_obj2_agent.py:\nP2RANK_PATH = ...\nVINA = ..."),
            ("Place KG file",                "data/knowledge_graph.gexf"),
            ("Run Streamlit",                "conda activate ipykernel\nstreamlit run app.py"),
        ], 1):
            st.markdown(f"**{i}. {title}**")
            st.code(cmd, language="bash")


# ══════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════

st.markdown("## Biomedical Drug Discovery Pipeline")

# ─── Initialize Chat History ──────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

EXAMPLES = [
    "What are targets for Lung cancer?",
    "Which pathways are involved in Parkinson's disease?",
    "Find binding pockets for Q01469",
    "Which proteins form the PPI network for Alzheimer disease?",
    "Which proteins are hub nodes in the Alzheimer PPI network?",
    "What proteins does Metformin act on?",
    "Which pathways are shared between lung cancer and breast cancer?",
    "Which pathways are the biggest hubs connecting the most diseases?",
    "Identify druggable targets in Alzheimer's disease",
    "Run full drug discovery pipeline for lung cancer",
    "Perform docking for protein P07471",
    "Find drug candidates for EGFR",
]

with st.form("query_form"):
    # Fix Accessibility: Provided robust non-empty widget titles coupled with label_visibility settings
    query = st.text_input(
        "Natural Language Pipeline Query",
        placeholder="Ask about diseases, targets, pathways, protein structures…",
        label_visibility="collapsed",
    )
    c_run, c_ex = st.columns([1, 4])
    with c_run:
        submitted = st.form_submit_button("▶ Run Pipeline", use_container_width=True)
    with c_ex:
        example = st.selectbox("Example Query Template Selector", [""] + EXAMPLES,
                               label_visibility="collapsed")

if example and not submitted:
    query     = example
    submitted = True

# ─── Load pipeline (cached after first run) ───────────────────
ns = load_pipeline()

if "error" in ns:
    st.error(f"Pipeline failed to load: {ns['error']}")
    for w in ns.get("warnings", []):
        st.warning(w)
    st.stop()

# ─── Process Query Pipeline execution into Session Memory ─────
if submitted and query:
    log_steps = []
    
    def log(msg, status="ok"):
        icon = {"ok": "✅", "run": "⏳", "err": "❌"}.get(status, "✅")
        cls  = {"ok": "step-ok", "run": "step-run", "err": "step-err"}.get(status, "step-ok")
        log_steps.append(f'<div class="{cls}">{icon} {msg}</div>')

    log(f"Query received: {query}", "run")

    # 1. Detect intent + route
    try:
        normalized_q  = ns["normalize_natural_query"](query)
        corrected_q   = ns["correct_query_spelling_and_entities"](normalized_q)
        intent        = ns["detect_intent"](corrected_q).strip()
        route         = ns["classify_route"](intent)
        log(f"Intent: {intent}  ·  Route: {route}")
    except Exception as e:
        log(f"Intent detection failed: {e}", "err")
        st.error(f"Execution terminated: {e}")
        st.stop()

    # 2. Execute KG agent
    kg_results  = {}
    explanation = ""
    if route in ("kg", "full"):
        log("Running KG Agent…", "run")
        try:
            from langchain_core.messages import HumanMessage
            KGState_init = {
                "query":             corrected_q,
                "corrected_query":   "",
                "intent":            intent,
                "entities":          {},
                "normalized":        {},
                "limit":             10,
                "kg_results":        {},
                "formatted_text":    "",
                "explanation":       "",
                "druggable_targets": [],
                "messages":          [HumanMessage(content=corrected_q)],
            }
            kg_final    = ns["kg_agent"].invoke(KGState_init)
            kg_results  = kg_final.get("kg_results", {})
            explanation = kg_final.get("explanation", "")
            log("KG Agent complete")
        except Exception as e:
            log(f"KG Agent error: {e}", "err")

    # 3. Execute Structure agent
    struct_output = {}
    if route in ("structure", "full"):
        log("Running Structure Agent…", "run")
        try:
            UNIPROT_PAT = re.compile(r"\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b")
            uni_match   = UNIPROT_PAT.search(corrected_q)

            if uni_match:
                uniprot_id   = uni_match.group()
                protein_name = uniprot_id
            else:
                druggable = []
                if kg_results:
                    ranked    = kg_results.get("ranked_targets", {})
                    druggable = (ranked.get("consensus_targets") or ranked.get("pagerank") or ranked.get("degree", []))
                if druggable:
                    uniprot_id   = druggable[0].get("protein_id", "")
                    protein_name = druggable[0].get("gene_name", uniprot_id)
                else:
                    entities = ns["extract_entities"](corrected_q)
                    proteins = entities.get("Protein", [])
                    if not proteins:
                        log("No UniProt ID found — structure agent skipped", "err")
                    else:
                        resolved     = ns["resolve_protein_kg_grounded"](proteins[0])
                        uniprot_id   = resolved["protein_id"] if resolved else proteins[0]
                        protein_name = resolved.get("gene_name", proteins[0]) if resolved else proteins[0]

            if 'uniprot_id' in locals():
                log(f"Structure Agent: {uniprot_id} ({protein_name})", "run")

                if route == "full":
                    all_structs = {}
                    ranked    = kg_results.get("ranked_targets", {})
                    druggable = (ranked.get("consensus_targets") or ranked.get("pagerank") or ranked.get("degree", []))[:3]
                    if not druggable:
                        druggable = [{"protein_id": uniprot_id, "gene_name": protein_name}]
                    for t in druggable:
                        uid   = t.get("protein_id", "")
                        gname = t.get("gene_name", uid)
                        log(f"  Structure: {uid} ({gname})", "run")
                        out = ns["_run_structure_agent_subprocess"](uid, gname)
                        all_structs[uid] = out
                        if "error" not in out:
                            log(f"  ... {uid} complete")
                        else:
                            log(f"  ... {uid}: {out.get('error','')}", "err")
                    struct_output = all_structs
                else:
                    struct_output = ns["_run_structure_agent_subprocess"](uniprot_id, protein_name)
                    if "error" not in struct_output:
                        log("Structure Agent complete")
                    else:
                        log(f"Structure Agent error: {struct_output.get('error','')}", "err")
        except Exception as e:
            log(f"Structure Agent exception: {e}", "err")

    # Commit step variables into memory history array
    st.session_state.chat_history.append({
        "query": query,
        "intent": intent,
        "route": route,
        "steps_html": "".join(log_steps),
        "kg_results": kg_results,
        "struct_output": struct_output,
        "explanation": explanation
    })

# ─── Render Logs and Dataframes ───────────────────────────────
if not st.session_state.chat_history:
    st.markdown("""
    <div style="margin-top:2.5rem;color:#888;font-size:14px;text-align:center">
        Type a query above and click <strong>▶ Run Pipeline</strong>.<br>
        Results appear here with tables matched to the detected intent.
    </div>""", unsafe_allow_html=True)
else:
    # Render all historic messages chronologically
    for idx, session in enumerate(reversed(st.session_state.chat_history)):
        total = len(st.session_state.chat_history)
        st.markdown(f"### 💬 Interaction #{total - idx}: {session['query']}")
        
        with st.expander("⏳ Pipeline Execution Steps", expanded=False):
            st.markdown(session['steps_html'], unsafe_allow_html=True)

        pipeline_diagram(session['route'])

        badge_cls = {"kg": "badge-kg", "structure": "badge-str"}.get(session['route'], "badge-pipe")
        badge_lbl = {"kg": "Knowledge Graph Agent", "structure": "Structure Agent"}.get(session['route'], "Full Pipeline")
        st.markdown(f'<span class="{badge_cls}">{badge_lbl} · {session["intent"]}</span>', unsafe_allow_html=True)
        st.divider()

        # Call dispatchers to render tables or dataframes tied to this item
        dispatch(session['intent'], session['kg_results'], session['struct_output'], session['explanation'], ns)
        st.markdown("<br><br>", unsafe_allow_html=True)