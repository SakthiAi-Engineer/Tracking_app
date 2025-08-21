# textile_tracker_app.py (PostgreSQL-integrated)
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import json

# ---------------- RAG / GenAI Imports ----------------
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from langchain_community.llms import Ollama

# ---------------- DB (PostgreSQL) ----------------
import psycopg2
import psycopg2.extras

# ---------------- App Config ----------------
st.set_page_config(page_title="Home Textile Tracker", layout="wide")

# ---------------- Constants ----------------
INDEX_DIR = "faiss_index"                    # FAISS vector store folder
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")

USERS = {
    "admin": "admin123",
    "user": "user123",
    "management": "manage123",
    "planning": "planning123",
    "procurement": "procure123",
    "stores": "stores123",
    "external_weaving": "extweave123",
    "quality": "quality123",
    "processing": "process123",
    "stitch_pack": "stitch123",
    "dispatch": "dispatch123"
}

SECTION_ACCESS = {
    "external_weaving": ["External Order", "Weaving"],
    "quality": ["Greige Inspection", "Inspection", "Final Inspection"],
    "processing": ["Processing"],
    "stitch_pack": ["Stitching", "Packing & Cartoning"],
    "dispatch": ["Shipment"],
}

process_options = [
    "External Order", "Weaving", "Greige Inspection", "Processing",
    "Inspection", "Stitching", "Final Inspection", "Packing & Cartoning", "Shipment"
]
import psycopg2
import streamlit as st

def get_connection():
    return psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        dbname=st.secrets["postgres"]["dbname"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        port=st.secrets["postgres"]["port"]
    )


# ======================= PostgreSQL Helpers =======================
@st.cache_resource(show_spinner=False)
def get_conn():
    """Create and cache a PostgreSQL connection using Streamlit secrets.
    st.secrets["postgres"] should contain host, port, dbname, user, password.
    """
    cfg = st.secrets.get("postgres", {})
    conn = psycopg2.connect(
        host=cfg.get("host", "localhost"),
        port=cfg.get("port", 5432),
        dbname=cfg.get("dbname", "textile_db"),
        user=cfg.get("user", "postgres"),
        password=cfg.get("password", "postgres"),
    )
    conn.autocommit = True
    return conn

@st.cache_resource(show_spinner=False)
def ensure_schema():
    conn = get_conn()
    with conn.cursor() as cur:
        # Plans (frozen monthly target plan)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS plans (
                id SERIAL PRIMARY KEY,
                "PO No" VARCHAR(100) NOT NULL,
                "Customer" VARCHAR(200),
                "External Order Start" DATE,   "External Order Finish" DATE,
                "Weaving Start" DATE,          "Weaving Finish" DATE,
                "Greige Inspection Start" DATE,"Greige Inspection Finish" DATE,
                "Processing Start" DATE,       "Processing Finish" DATE,
                "Inspection Start" DATE,       "Inspection Finish" DATE,
                "Stitching Start" DATE,        "Stitching Finish" DATE,
                "Final Inspection Start" DATE, "Final Inspection Finish" DATE,
                "Packing & Cartoning Start" DATE, "Packing & Cartoning Finish" DATE,
                "Shipment Start" DATE,         "Shipment Finish" DATE
            );
            """
        )
        # Daily status (actuals)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS status_updates (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "PO No" VARCHAR(100) NOT NULL,
                "Customer" VARCHAR(200),
                "Process" VARCHAR(50) NOT NULL,
                "Actual Start" DATE,
                "Actual Finish" DATE,
                "Remarks" TEXT,
                "Submitted By" VARCHAR(100)
            );
            """
        )
        # Requirements (Planning/Procurement/Stores combined)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS requirements (
                id SERIAL PRIMARY KEY,
                type VARCHAR(50) NOT NULL, -- Raw Material / Carton box / Accessories
                "Item Name" VARCHAR(300),
                quantity INT,
                "Requested In-house Date" DATE,
                status VARCHAR(30) DEFAULT 'Requested',
                "Procurement In-house Date" DATE,
                "Procured Quantity" INT,
                received VARCHAR(10), -- Yes/No
                "Received Quantity" INT,
                "Actual In-house Date" DATE
            );
            """
        )
    return True

# Simple query utils

def fetch_df(sql: str, params=None) -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql(sql, conn, params=params)

def execute(sql: str, params=None):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, params)

# Ensure DB schema
ensure_schema()

# ======================= Ollama Helpers =======================
# Initialize Ollama client once (keep your requirement)
if "ollama_client" not in st.session_state:
    try:
        st.session_state.ollama_client = Ollama(model=OLLAMA_MODEL)
    except Exception:
        st.error("⚠️ Ollama is not running or the model is missing.")
        st.code(f"ollama pull {OLLAMA_MODEL}", language="bash")
        st.stop()


def _strip_think(text: str) -> str:
    if not isinstance(text, str):
        return ""
    out, skip, i = [], False, 0
    while i < len(text):
        if text.startswith("<think>", i):
            skip, i = True, i + 7
            continue
        if text.startswith("</think>", i):
            skip, i = False, i + 8
            continue
        if not skip:
            out.append(text[i])
        i += 1
    return "".join(out).strip()

@st.cache_resource(show_spinner=False)
def _ollama_session():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s

def ask_ollama(prompt: str, max_tokens: int = 2048) -> tuple[bool, str]:
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                # Allow long, complete answers
                "num_predict": max_tokens,
                # Be a bit more verbose and deterministic
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        url = f"{OLLAMA_BASE_URL}/api/generate"
        resp = _ollama_session().post(url, json=payload, timeout=300)
        if resp.status_code != 200:
            return False, f"Ollama error {resp.status_code}: {resp.text}"
        data = resp.json()
        txt = data.get("response", "").strip()
        return True, _strip_think(txt)
    except Exception as e:
        return False, f"Failed to contact Ollama at {OLLAMA_BASE_URL}: {e}"

@st.cache_resource(show_spinner=False)
def _get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS from DB

def build_faiss_index_from_db() -> tuple[bool, str]:
    plan_df = fetch_df("SELECT * FROM plans")
    if plan_df.empty:
        return False, "No plans found. Upload on the Upload page first."

    texts = []
    for _, r in plan_df.iterrows():
        lines = [
            f"PO No: {r.get('PO No', '')}",
            f"Customer: {r.get('Customer', '')}",
        ]
        for proc in process_options:
            s_col = f"{proc} Start"; e_col = f"{proc} Finish"
            if s_col in plan_df.columns and pd.notna(r.get(s_col)):
                lines.append(f"{proc} Planned Start: {r.get(s_col)}")
            if e_col in plan_df.columns and pd.notna(r.get(e_col)):
                lines.append(f"{proc} Planned Finish: {r.get(e_col)}")
        texts.append("\n".join([str(x) for x in lines if str(x).strip() != ""]))

    act_df = fetch_df("SELECT * FROM status_updates")
    if not act_df.empty:
        for po, g in act_df.groupby("PO No"):
            lines = [f"PO No: {po}", "Actuals:"]
            for _, row in g.iterrows():
                lines.append(
                    f"- {row.get('Process','')}: Start={row.get('Actual Start','')}, Finish={row.get('Actual Finish','')}, Remarks={row.get('Remarks','')}"
                )
            texts.append("\n".join(lines))

    if not texts:
        return False, "No text content available to index."

    embeddings = _get_embeddings()
    db = FAISS.from_texts(texts, embeddings)
    db.save_local(INDEX_DIR)
    return True, f"FAISS index updated. Indexed {len(texts)} records."

# ---------------- Login ----------------
if "logged_in" not in st.session_state:
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state["role"] = username
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.sidebar.error("Invalid login")
    st.stop()

role = st.session_state["role"]

with st.sidebar:
    st.markdown("---")
    if st.button("Logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

# ---------------- Sidebar Navigation (role-based) ----------------
if role == "admin":
    page = st.sidebar.radio(
        "Go to",
        [
            "Dashboard", "Planning", "Procurement", "Stores", "Upload",
            "Daily Status Update", "Visualize", "Logs", "Unlock POs",
            "Unlock Process", "Pie chart view", "Ask AI"
        ]
    )
elif role == "user":
    page = st.sidebar.radio("Go to", ["Dashboard", "Daily Status Update", "Visualize", "Logs", "Ask AI"])
elif role == "management":
    page = st.sidebar.radio("Go to", ["Dashboard", "Visualize", "Pie chart view", "Ask AI"])
elif role in ["planning", "procurement", "stores"]:
    page = st.sidebar.radio("Go to", ["Dashboard", role.capitalize(), "Visualize", "Ask AI"])
elif role in SECTION_ACCESS:
    page = st.sidebar.radio("Go to", ["Dashboard", "Daily Status Update", "Visualize", "Ask AI"])
else:
    page = st.sidebar.radio("Go to", ["Dashboard"])  # minimal

# ---------------- Pages ----------------

# === Dashboard ===
if page == "Dashboard":
    st.title("🎯 Enhanced Textile Tracker Dashboard")
    st.markdown("### Complete End-to-End Process Visualization")

    types = ["Raw Material", "Carton box", "Accessories"]
    planning_counts, procured_counts, received_counts = {}, {}, {}

    req_df = fetch_df("SELECT * FROM requirements")
    for t in types:
        df_t = req_df[req_df["type"].str.lower() == t.lower()] if not req_df.empty else pd.DataFrame()
        planning_counts[t] = len(df_t)
        procured_counts[t] = df_t["status"].eq("Procured").sum() if not df_t.empty and "status" in df_t.columns else 0
        received_counts[t] = df_t["received"].str.upper().eq("YES").sum() if not df_t.empty and "received" in df_t.columns else 0

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Planned", sum(planning_counts.values()), delta=f"+{sum(planning_counts.values()) - sum(procured_counts.values())} pending")
    with c2:
        st.metric("Total Procured", sum(procured_counts.values()))
    with c3:
        st.metric("Total Received", sum(received_counts.values()))
    with c4:
        completion_rate = (sum(received_counts.values()) / max(sum(planning_counts.values()), 1)) * 100
        st.metric("Overall Completion", f"{completion_rate:.1f}%")

    st.markdown("---")
    st.subheader("🔍 Advanced Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        selected_materials = st.multiselect("Material Type", types, default=types)
    with f2:
        status_options = ["Requested", "Procured", "Received"]
        status_filter = st.multiselect("Status", status_options, default=status_options)
    with f3:
        show_details = st.checkbox("Show Detailed View", value=True)

    st.markdown("---")
    st.subheader("📊 Material Type Analysis")
    m1, m2 = st.columns(2)

    with m1:
        material_status = []
        if not req_df.empty:
            for t in selected_materials:
                df_t = req_df[req_df["type"].str.lower() == t.lower()]
                if not df_t.empty and 'status' in df_t.columns:
                    counts = df_t['status'].value_counts()
                    for status, count in counts.items():
                        if status in status_filter:
                            material_status.append({'Material': t, 'Status': status, 'Count': int(count)})
        if material_status:
            material_df = pd.DataFrame(material_status)
            fig_material = px.bar(material_df, x='Material', y='Count', color='Status', title="Material Status Distribution")
            st.plotly_chart(fig_material, use_container_width=True)

    with m2:
        plans_df = fetch_df("SELECT \"Customer\" FROM plans WHERE \"Customer\" IS NOT NULL")
        if not plans_df.empty and 'Customer' in plans_df.columns:
            customer_counts = plans_df['Customer'].value_counts().head(10)
            fig_customers = px.pie(values=customer_counts.values, names=customer_counts.index, title="Top 10 Customers by PO Count")
            st.plotly_chart(fig_customers, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Real-time Activity Feed")
    if not req_df.empty:
        recent_df = req_df.copy()
        recent_df["Material Type"] = recent_df["type"].str.title()
        if 'status' in recent_df.columns:
            recent_df = recent_df[recent_df['status'].isin([s for s in status_filter if s in ['Requested','Procured','Received']])]
        if 'Requested In-house Date' in recent_df.columns:
            recent_df['Requested In-house Date'] = pd.to_datetime(recent_df['Requested In-house Date'], errors='coerce')
            recent_df = recent_df.dropna(subset=['Requested In-house Date']).sort_values(by="Requested In-house Date", ascending=False).head(10)
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No recent activity found matching your filters.")

    st.markdown("---")
    st.subheader("🔄 Process Status Overview")
    try:
        frozen_df = fetch_df("SELECT DISTINCT \"PO No\", \"Customer\" FROM plans")
        status_df = fetch_df("SELECT * FROM status_updates")
        if not frozen_df.empty and not status_df.empty:
            common_pos = set(frozen_df['PO No'].dropna().unique()) & set(status_df['PO No'].dropna().unique())
            if common_pos:
                selected_po = st.selectbox("Select PO No for Process Timeline", sorted(list(common_pos)))
                po_target = fetch_df("SELECT * FROM plans WHERE \"PO No\"=%s ORDER BY id ASC LIMIT 1", [selected_po])
                po_actual = status_df[status_df['PO No'] == selected_po]
                if not po_target.empty and not po_actual.empty:
                    timeline_data = []
                    for process in process_options:
                        ts = po_target.get(f"{process} Start").iloc[0] if f"{process} Start" in po_target.columns else None
                        te = po_target.get(f"{process} Finish").iloc[0] if f"{process} Finish" in po_target.columns else None
                        ar = po_actual[po_actual['Process'] == process]
                        as_ = pd.to_datetime(ar['Actual Start'].iloc[0], errors='coerce') if not ar.empty else None
                        ae_ = pd.to_datetime(ar['Actual Finish'].iloc[0], errors='coerce') if not ar.empty else None
                        if pd.notna(ts):
                            timeline_data.append({'Process': process, 'Type': 'Target Start', 'Date': ts, 'PO': selected_po})
                        if pd.notna(te):
                            timeline_data.append({'Process': process, 'Type': 'Target End', 'Date': te, 'PO': selected_po})
                        if pd.notna(as_):
                            timeline_data.append({'Process': process, 'Type': 'Actual Start', 'Date': as_, 'PO': selected_po})
                        if pd.notna(ae_):
                            timeline_data.append({'Process': process, 'Type': 'Actual End', 'Date': ae_, 'PO': selected_po})
                    if timeline_data:
                        timeline_df = pd.DataFrame(timeline_data)
                        fig_timeline = px.line(timeline_df, x='Date', y='Process', color='Type', title=f"Process Timeline Comparison for PO: {selected_po}", markers=True)
                        fig_timeline.update_layout(xaxis_title="Timeline", yaxis_title="Process", height=400, showlegend=True)
                        st.plotly_chart(fig_timeline, use_container_width=True)
                        st.subheader("Process Summary")
                        summary_data = []
                        for process in process_options:
                            ts = po_target.get(f"{process} Start").iloc[0] if f"{process} Start" in po_target.columns else None
                            te = po_target.get(f"{process} Finish").iloc[0] if f"{process} Finish" in po_target.columns else None
                            ar = po_actual[po_actual['Process'] == process]
                            as_ = ar['Actual Start'].iloc[0] if not ar.empty else None
                            ae_ = ar['Actual Finish'].iloc[0] if not ar.empty else None
                            status = 'Completed' if pd.notna(ae_) else 'In Progress' if pd.notna(as_) else 'Not Started'
                            summary_data.append({'Process': process, 'Target Start': ts, 'Target End': te, 'Actual Start': as_, 'Actual End': ae_, 'Status': status})
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                    else:
                        st.info(f"No process data found for PO: {selected_po}")
            else:
                st.warning("No matching PO numbers found between frozen invoices and daily updates.")
        else:
            st.warning("No data available. Please ensure both frozen invoices and daily updates contain data.")
    except Exception as e:
        st.error(f"Error loading process status data: {str(e)}")
        st.info("Please check that your data tables are properly populated.")

    st.markdown("---")
    st.subheader("📤 Export Options")
    ec1, ec2 = st.columns(2)
    with ec1:
        export_df = fetch_df("SELECT * FROM requirements")
        if not export_df.empty:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download Current View (CSV)",
                data=csv,
                file_name=f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    with ec2:
        st.info("Full report generation feature coming soon…")

# === Upload ===
if page == "Upload":
    st.title("Upload Monthly Target Plan")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        # Normalize expected columns
        needed_cols = {"PO No", "Customer"}
        if not needed_cols.issubset(set(df.columns)):
            st.error(f"Excel must contain columns: {', '.join(sorted(needed_cols))}")
            st.stop()
        # Replace existing plans for same PO No (simple approach: clear table then insert)
        execute("DELETE FROM plans")

        # Insert rows
        cols = ["PO No", "Customer"] + [f"{p} Start" for p in process_options] + [f"{p} Finish" for p in process_options]
        for _, row in df.iterrows():
            values = [row.get(c, None) if c in df.columns else None for c in cols]
            placeholders = ",".join(["%s"] * len(cols))
            sql = f"INSERT INTO plans (\"PO No\", \"Customer\", " + ", ".join([f'"{c}"' for c in cols[2:]]) + f") VALUES (%s, %s, {placeholders[6:]})"
            # Build params explicitly to keep order
            params = [row.get("PO No"), row.get("Customer")] + [row.get(c, None) if c in df.columns else None for c in cols[2:]]
            execute(sql, params)
        st.success("Target Plan uploaded and frozen!")
        st.dataframe(df)
        with st.spinner("Building AI knowledge base from the latest plan…"):
            ok, msg = build_faiss_index_from_db()
        st.success(msg) if ok else st.warning(msg)

# === Daily Status Update ===
if page == "Daily Status Update":
    st.title("📤 Daily Status Update - Department Wise")
    frozen_df = fetch_df("SELECT DISTINCT \"PO No\", \"Customer\" FROM plans")
    if frozen_df.empty:
        st.error("No target plans uploaded and frozen yet.")
        st.stop()

    invoice_list = frozen_df["PO No"].dropna().unique().tolist()
    customer_lookup = dict(zip(frozen_df["PO No"], frozen_df["Customer"]))
    status_df = fetch_df("SELECT * FROM status_updates")

    selected_invoice = st.selectbox("Select PO No to update or edit", invoice_list)
    selected_customer = customer_lookup.get(selected_invoice, "")
    st.text_input("Customer Name", selected_customer, disabled=True)

    invoice_status = status_df[status_df["PO No"] == selected_invoice] if not status_df.empty else pd.DataFrame()

    allowed_sections = process_options if role not in SECTION_ACCESS else SECTION_ACCESS[role]

    for process in process_options:
        if process not in allowed_sections:
            continue
        row = invoice_status[invoice_status["Process"] == process]
        st.subheader(process)
        if not row.empty and pd.notna(row.iloc[0]["Actual Start"]):
            st.markdown(f"**Start Date:** {row.iloc[0]['Actual Start']}")
        if not row.empty and pd.notna(row.iloc[0]["Actual Finish"]):
            st.markdown(f"**End Date:** {row.iloc[0]['Actual Finish']}")

        st.markdown("**Enter or Modify Dates Below:**")
        submitted_by = st.text_input("Your Name (required)", key=f"submitter_{process}_{selected_invoice}", value=row.iloc[0]["Submitted By"] if not row.empty else "")
        start_default = pd.to_datetime(row.iloc[0]["Actual Start"]).date() if not row.empty and pd.notna(row.iloc[0]["Actual Start"]) else None
        finish_default = pd.to_datetime(row.iloc[0]["Actual Finish"]).date() if not row.empty and pd.notna(row.iloc[0]["Actual Finish"]) else None
        start = st.date_input(f"{process} Start", value=start_default, key=f"start_{process}_{selected_invoice}")
        finish = st.date_input(f"{process} Finish", value=finish_default, key=f"finish_{process}_{selected_invoice}")
        remarks = st.text_area("Remarks", key=f"remarks_{process}_{selected_invoice}", value=row.iloc[0]["Remarks"] if not row.empty else "")

        if st.button(f"Submit {process}", key=f"submit_{process}_{selected_invoice}"):
            if not submitted_by:
                st.error("Your name is mandatory to submit.")
            else:
                # Upsert: delete existing row for (PO, Process) then insert fresh
                execute("DELETE FROM status_updates WHERE \"PO No\"=%s AND \"Process\"=%s", [selected_invoice, process])
                execute(
                    """
                    INSERT INTO status_updates ("PO No", "Customer", "Process", "Actual Start", "Actual Finish", "Remarks", "Submitted By")
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    [selected_invoice, selected_customer, process, start, finish, remarks, submitted_by]
                )
                st.success(f"{process} updated for {selected_invoice}")

    if st.button("🔁 Refresh AI Knowledge Base (include latest actuals)"):
        with st.spinner("Rebuilding FAISS index…"):
            ok, msg = build_faiss_index_from_db()
        st.success(msg) if ok else st.warning(msg)

# === Visualize ===
if page == "Visualize":
    st.title("Process Completion Matrix with Status Comparison")
    df = fetch_df("SELECT * FROM plans")
    if df.empty:
        st.warning("Please upload data in Page 1.")
    else:
        actuals = fetch_df("SELECT * FROM status_updates")
        matrix = []
        allowed_sections = process_options if role not in SECTION_ACCESS else SECTION_ACCESS[role]
        for _, row in df.iterrows():
            status_row = {"PO No": row["PO No"], "Customer": row["Customer"]}
            for process in process_options:
                if process not in allowed_sections:
                    continue
                arow = actuals[(actuals["PO No"] == row["PO No"]) & (actuals["Process"] == process)] if not actuals.empty else pd.DataFrame()
                planned_end = row.get(f"{process} Finish")
                actual_end = arow["Actual Finish"].values[0] if not arow.empty else None
                if pd.notna(actual_end) and pd.notna(planned_end):
                    delay_days = (pd.to_datetime(actual_end) - pd.to_datetime(planned_end)).days
                    if delay_days > 0:
                        status = f"🟥 Delayed ({int(delay_days)}d)"
                    elif delay_days < 0:
                        status = f"🟩 Early ({abs(int(delay_days))}d)"
                    else:
                        status = "🟨 On Time"
                elif not arow.empty:
                    status = "🔵 In Progress"
                else:
                    status = "⬜ Not Started"
                status_row[process] = status
            matrix.append(status_row)
        status_df = pd.DataFrame(matrix)
        st.dataframe(status_df, use_container_width=True)
        st.subheader("Summary Table")
        po_selected = st.selectbox("Select PO No for Summary", status_df["PO No"].unique())
        summary = []
        actuals_po = actuals[actuals["PO No"] == po_selected] if not actuals.empty else pd.DataFrame()
        for process in process_options:
            if process not in allowed_sections:
                continue
            plan_end = df[df["PO No"] == po_selected][f"{process} Finish"].values[0] if f"{process} Finish" in df.columns and (df["PO No"] == po_selected).any() else None
            arow = actuals_po[actuals_po["Process"] == process]
            actual_end = arow["Actual Finish"].values[0] if not arow.empty else None
            submitted_by = arow["Submitted By"].values[0] if not arow.empty else ""
            remarks = arow["Remarks"].values[0] if not arow.empty else ""
            difference = None
            if pd.notna(plan_end) and actual_end is not None and str(actual_end) != "":
                difference = (pd.to_datetime(actual_end) - pd.to_datetime(plan_end)).days
            summary.append({
                "Process": process,
                "Planned Finish": plan_end,
                "Actual Finish": actual_end,
                "Delay (Days)": int(difference) if difference is not None else None,
                "Remarks": remarks,
                "Submitted By": submitted_by
            })
        summary_df = pd.DataFrame(summary)
        # Fix pyarrow issue by enforcing numeric type for Delay (Days)
        if 'Delay (Days)' in summary_df.columns:
            summary_df['Delay (Days)'] = pd.to_numeric(summary_df['Delay (Days)'], errors='coerce')
        st.dataframe(summary_df, use_container_width=True)

# === Pie chart view ===
if page == "Pie chart view":
    import io
    from plotly.io import to_image
    from PIL import Image
    from fpdf import FPDF

    st.title("📊 Pie Chart View - Planned vs Actual (by Days)")
    plan_df = fetch_df("SELECT * FROM plans")
    if plan_df.empty:
        st.warning("Please upload a target plan first.")
        st.stop()
    actual_df = fetch_df("SELECT * FROM status_updates")
    customer_list = plan_df["Customer"].dropna().unique().tolist()
    selected_customer = st.selectbox("Select Customer", customer_list)
    customer_orders = plan_df[plan_df["Customer"] == selected_customer]
    all_figs = []
    for idx, order in customer_orders.iterrows():
        po_no = order["PO No"]
        if idx > 0:
            st.markdown("---")
        st.markdown(f"### PO No: {po_no}")
        planned_durations, actual_durations = [], []
        for process in process_options:
            plan_start = order.get(f"{process} Start"); plan_end = order.get(f"{process} Finish")
            if pd.notna(plan_start) and pd.notna(plan_end):
                days = (pd.to_datetime(plan_end) - pd.to_datetime(plan_start)).days
                planned_durations.append({"Process": process, "Days": max(int(days), 0)})
            actual_rows = actual_df[(actual_df["PO No"] == po_no) & (actual_df["Process"] == process)] if not actual_df.empty else pd.DataFrame()
            if not actual_rows.empty:
                actual_start = actual_rows.iloc[0]["Actual Start"]
                actual_finish = actual_rows.iloc[0]["Actual Finish"]
                if pd.notna(actual_start) and pd.notna(actual_finish):
                    adays = (pd.to_datetime(actual_finish) - pd.to_datetime(actual_start)).days
                    actual_durations.append({"Process": process, "Days": max(int(adays), 0)})
        col1, col2 = st.columns(2)
        process_colors = px.colors.qualitative.Plotly[:len(process_options)]
        color_map = {proc: process_colors[i % len(process_colors)] for i, proc in enumerate(process_options)}
        fig1 = fig2 = None
        if planned_durations:
            with col1:
                pdfd = pd.DataFrame(planned_durations)
                fig1 = px.pie(pdfd, names="Process", values="Days", title="Planned Days", hover_data=['Days'], labels={'Days': 'Days'}, hole=0.3, color="Process", color_discrete_map=color_map)
                fig1.update_traces(textinfo='value', hovertemplate='%{label}: %{percent}')
                st.plotly_chart(fig1, use_container_width=True, key=f"{po_no}_planned")
                st.markdown(f"**Total Planned Days:** {int(pdfd['Days'].sum())} days")
        if actual_durations:
            with col2:
                adf = pd.DataFrame(actual_durations)
                fig2 = px.pie(adf, names="Process", values="Days", title="Actual Days", hover_data=['Days'], labels={'Days': 'Days'}, hole=0.3, color="Process", color_discrete_map=color_map)
                fig2.update_traces(textinfo='value', hovertemplate='%{label}: %{percent}')
                st.plotly_chart(fig2, use_container_width=True, key=f"{po_no}_actual")
                st.markdown(f"**Total Actual Days:** {int(adf['Days'].sum())} days")
        if fig1 and fig2:
            all_figs.append((po_no, fig1, fig2))
    if all_figs and st.button("📥 Download Combined PDF Report"):
        pdf_doc = FPDF(); img_paths = []
        for po_no, fig1, fig2 in all_figs:
            img1 = Image.open(io.BytesIO(to_image(fig1, format="png")))
            img2 = Image.open(io.BytesIO(to_image(fig2, format="png")))
            p1 = f"{po_no}_planned.png"; p2 = f"{po_no}_actual.png"
            img1.save(p1); img2.save(p2); img_paths.extend([p1, p2])
            for path in [p1, p2]:
                pdf_doc.add_page(); pdf_doc.image(path, x=10, y=20, w=180)
        pdf_path = f"{selected_customer}_Combined_Report.pdf"; pdf_doc.output(pdf_path)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", data=f, file_name=pdf_path, mime="application/pdf")
        for p in img_paths: os.remove(p)
        os.remove(pdf_path)

# === Planning ===
if page == "Planning":
    st.title("Planning Module")
    division = st.selectbox("Select Division", ["Procurement", "Production", "Dispatch"])
    if division == "Procurement":
        st.subheader("Procurement Requirement Entry")
        st.markdown("#### 📥 Bulk Upload Requirements (Excel/CSV)")
        bulk_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"], key="bulk_upload")
        if bulk_file:
            bulk_df = pd.read_excel(bulk_file) if bulk_file.name.endswith(".xlsx") else pd.read_csv(bulk_file)
            required_cols = {"Item Name", "Quantity", "Requested In-house Date", "Type"}
            if required_cols.issubset(set(bulk_df.columns)):
                execute("DELETE FROM requirements")
                for _, r in bulk_df.iterrows():
                    execute(
                        """
                        INSERT INTO requirements (type, "Item Name", quantity, "Requested In-house Date", status)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        [r.get("Type"), r.get("Item Name"), int(r.get("Quantity", 0) or 0), r.get("Requested In-house Date"), "Requested"]
                    )
                st.success("Bulk requirements uploaded and frozen by type!")
                st.dataframe(bulk_df)
            else:
                st.error(f"File must contain columns: {', '.join(required_cols)}")
        st.markdown("---"); st.markdown("#### Or add requirements individually:")
        req_type = st.selectbox("Select Requirement Type", ["Raw Material", "Carton box", "Accessories"])
        st.write(f"Add {req_type} Requirement")
        with st.form(f"add_{req_type}_form"):
            item_name = st.text_input("Item Name")
            quantity = st.number_input("Quantity", min_value=1, step=1)
            req_date = st.date_input("Requested In-house Date", value=datetime.today())
            submit = st.form_submit_button("Add Requirement")
        if submit and item_name:
            execute(
                """
                INSERT INTO requirements (type, "Item Name", quantity, "Requested In-house Date", status)
                VALUES (%s, %s, %s, %s, 'Requested')
                """,
                [req_type, item_name, int(quantity), req_date]
            )
            st.success(f"{req_type} requirement added.")
        df_view = fetch_df("SELECT * FROM requirements WHERE type=%s", [req_type])
        if not df_view.empty:
            st.dataframe(df_view)

# === Procurement ===
elif page == "Procurement":
    st.title("Procurement Module")
    req_type = st.selectbox("Select Requirement Type", ["Raw Material", "Carton box", "Accessories"])
    df = fetch_df("SELECT * FROM requirements WHERE type=%s", [req_type])
    if not df.empty:
        st.write("Pending Requirements:"); st.dataframe(df)
        idx = st.number_input("Select Row ID to Update", min_value=int(df['id'].min()), max_value=int(df['id'].max()), step=1)
        inhouse_date = st.date_input("Procurement In-house Date", value=datetime.today())
        default_qty = int(df[df['id'] == idx]["quantity"].values[0]) if (df['id'] == idx).any() else 1
        proc_qty = st.number_input("Procured Quantity", min_value=1, step=1, value=default_qty)
        if st.button("Update Procurement"):
            execute(
                """
                UPDATE requirements
                SET "Procurement In-house Date"=%s, "Procured Quantity"=%s, status='Procured'
                WHERE id=%s
                """,
                [inhouse_date, int(proc_qty), int(idx)]
            )
            st.success("Procurement info updated.")
    else:
        st.info("No requirements found. Please add in Planning module.")

# === Stores ===
elif page == "Stores":
    st.title("Stores Module")
    req_type = st.selectbox("Select Item Type", ["Raw Material", "Carton box", "Accessories"])
    df = fetch_df("SELECT * FROM requirements WHERE type=%s AND status IN ('Procured','Received','Requested')", [req_type])
    if not df.empty:
        st.write("Procured Items:"); st.dataframe(df)
        idx = st.number_input("Select Row ID to Confirm Receipt", min_value=int(df['id'].min()), max_value=int(df['id'].max()), step=1)
        received = st.selectbox("Received?", ["No", "Yes"])
        default_qty = int(df[df['id'] == idx]["quantity"].values[0]) if (df['id'] == idx).any() else 0
        rec_qty = st.number_input("Received Quantity", min_value=0, step=1, value=default_qty)
        rec_date = st.date_input("Actual In-house Date", value=datetime.today())
        if st.button("Confirm Receipt"):
            execute(
                """
                UPDATE requirements
                SET received=%s, "Received Quantity"=%s, "Actual In-house Date"=%s, status=CASE WHEN %s='Yes' THEN 'Received' ELSE status END
                WHERE id=%s
                """,
                [received, int(rec_qty), rec_date, received, int(idx)]
            )
            st.success("Receipt confirmed.")
    else:
        st.info("No procured items found.")

# === Logs ===
if page == "Logs":
    st.title("Daily Logs and Export")
    log_df = fetch_df("SELECT * FROM status_updates ORDER BY timestamp DESC")
    if not log_df.empty:
        st.dataframe(log_df)
        st.download_button("Download Log as CSV", log_df.to_csv(index=False), file_name="log_export.csv")
    else:
        st.warning("No logs available.")

# === Unlock POs / Unlock Process (admin placeholders retained) ===
if page == "Unlock POs" and role == "admin":
    st.title("Unlock PO Data for Editing")
    frozen_pos = fetch_df("SELECT DISTINCT \"PO No\" FROM plans")
    if not frozen_pos.empty:
        selected_unlock_po = st.selectbox("Select PO to Unlock", frozen_pos["PO No"].tolist())
        if st.button("Unlock Selected PO"):
            st.success(f"PO {selected_unlock_po} has been unlocked (placeholder logic).")
    else:
        st.info("No plan uploaded yet.")

if page == "Unlock Process" and role == "admin":
    st.title("Unlock Individual Process")
    po_list_df = fetch_df("SELECT DISTINCT \"PO No\" FROM plans")
    if not po_list_df.empty:
        selected_po = st.selectbox("Select PO", po_list_df["PO No"].tolist())
        selected_process = st.selectbox("Select Process", process_options)
        if st.button("Unlock Process"):
            st.success(f"Process '{selected_process}' for PO '{selected_po}' has been unlocked (placeholder logic).")
    else:
        st.info("No plan uploaded yet.")

