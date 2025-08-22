import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from sqlalchemy import create_engine, text
import json

# ---------------- Cloud Configuration ----------------
# Database configuration
SUPABASE_URL = str(st.secrets["SUPABASE_URL"])
SUPABASE_KEY = str(st.secrets["SUPABASE_KEY"])
DATABASE_URL = str(st.secrets["DATABASE_URL"])

# NVIDIA API configuration
NVIDIA_API_KEY = str(st.secrets["NVIDIA_API_KEY"])
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
st.write("DATABASE_URL secret:", st.secrets["DATABASE_URL"], type(st.secrets["DATABASE_URL"]))

def get_db_connection():
    # Handle both cases: nested dict or flat string
    raw_url = st.secrets["DATABASE_URL"]
    if isinstance(raw_url, dict):
        db_url = raw_url["DATABASE_URL"]
    else:
        db_url = raw_url

    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    engine = create_engine(db_url)
    return engine


#def get_db_connection():
   # """Create database connection using SQLAlchemy"""
   # engine = create_engine(DATABASE_URL)
   # return engine

def init_database():
    """Initialize database tables"""
    engine = get_db_connection()
    
    with engine.connect() as conn:
        # Create tables if they don't exist
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR(50) PRIMARY KEY,
                password VARCHAR(100),
                role VARCHAR(50)
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS frozen_plans (
                id SERIAL PRIMARY KEY,
                po_no VARCHAR(50),
                customer VARCHAR(100),
                data JSONB,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_status (
                id SERIAL PRIMARY KEY,
                po_no VARCHAR(50),
                customer VARCHAR(100),
                process VARCHAR(50),
                actual_start DATE,
                actual_finish DATE,
                remarks TEXT,
                submitted_by VARCHAR(100),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS procurement_data (
                id SERIAL PRIMARY KEY,
                type VARCHAR(50),
                item_name VARCHAR(200),
                quantity INTEGER,
                requested_in_house_date DATE,
                status VARCHAR(50),
                procurement_in_house_date DATE,
                procured_quantity INTEGER,
                received VARCHAR(10),
                received_quantity INTEGER,
                actual_in_house_date DATE
            )
        """))
        
        conn.commit()

def insert_default_users():
    """Insert default users"""
    engine = get_db_connection()
    
    with engine.connect() as conn:
        # Check if users already exist
        result = conn.execute(text("SELECT COUNT(*) FROM users"))
        if result.scalar() == 0:
            default_users = [
                ("admin", "admin123", "admin"),
                ("user", "user123", "user"),
                ("management", "manage123", "management"),
                ("planning", "planning123", "planning"),
                ("procurement", "procure123", "procurement"),
                ("stores", "stores123", "stores"),
                ("external_weaving", "extweave123", "external_weaving"),
                ("quality", "quality123", "quality"),
                ("processing", "process123", "processing"),
                ("stitch_pack", "stitch123", "stitch_pack"),
                ("dispatch", "dispatch123", "dispatch")
            ]
            
            for username, password, role in default_users:
                conn.execute(
                    text("INSERT INTO users (username, password, role) VALUES (:username, :password, :role)"),
                    {"username": username, "password": password, "role": role}
                )
            conn.commit()

# ---------------- App Config ----------------
st.set_page_config(page_title="Cloud Textile Tracker", layout="wide")

# Initialize database
init_database()
insert_default_users()

# ---------------- Constants ----------------
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

# ---------------- Database Operations ----------------
def get_user_role(username):
    """Get user role from database"""
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT role FROM users WHERE username = :username"),
            {"username": username}
        )
        return result.scalar()

def save_frozen_plan(df):
    """Save frozen plan to database"""
    engine = get_db_connection()
    with engine.connect() as conn:
        # Clear existing plans
        conn.execute(text("DELETE FROM frozen_plans"))
        
        # Insert new plans
        for _, row in df.iterrows():
            conn.execute(
                text("INSERT INTO frozen_plans (po_no, customer, data) VALUES (:po_no, :customer, :data)"),
                {
                    "po_no": str(row.get('PO No', '')),
                    "customer": str(row.get('Customer', '')),
                    "data": json.dumps(row.to_dict())
                }
            )
        conn.commit()

def get_frozen_plans():
    """Get all frozen plans"""
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM frozen_plans ORDER BY uploaded_at DESC"))
        return pd.DataFrame(result.fetchall())

def save_daily_status(data):
    """Save daily status update"""
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO daily_status 
                (po_no, customer, process, actual_start, actual_finish, remarks, submitted_by)
                VALUES (:po_no, :customer, :process, :actual_start, :actual_finish, :remarks, :submitted_by)
            """),
            data
        )
        conn.commit()

def get_daily_status():
    """Get all daily status updates"""
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM daily_status ORDER BY timestamp DESC"))
        return pd.DataFrame(result.fetchall())

def save_procurement_data(data_type, df):
    """Save procurement data"""
    engine = get_db_connection()
    with engine.connect() as conn:
        # Clear existing data for this type
        conn.execute(text("DELETE FROM procurement_data WHERE type = :type"), {"type": data_type})
        
        # Insert new data
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO procurement_data 
                    (type, item_name, quantity, requested_in_house_date, status, 
                     procurement_in_house_date, procured_quantity, received, 
                     received_quantity, actual_in_house_date)
                    VALUES (:type, :item_name, :quantity, :requested_in_house_date, :status,
                            :procurement_in_house_date, :procured_quantity, :received,
                            :received_quantity, :actual_in_house_date)
                """),
                {
                    "type": data_type,
                    "item_name": str(row.get('Item Name', '')),
                    "quantity": int(row.get('Quantity', 0)),
                    "requested_in_house_date": row.get('Requested In-house Date'),
                    "status": str(row.get('Status', 'Requested')),
                    "procurement_in_house_date": row.get('Procurement In-house Date'),
                    "procured_quantity": int(row.get('Procured Quantity', 0)),
                    "received": str(row.get('Received', 'No')),
                    "received_quantity": int(row.get('Received Quantity', 0)),
                    "actual_in_house_date": row.get('Actual In-house Date')
                }
            )
        conn.commit()

def get_procurement_data(data_type=None):
    """Get procurement data"""
    engine = get_db_connection()
    with engine.connect() as conn:
        if data_type:
            result = conn.execute(
                text("SELECT * FROM procurement_data WHERE type = :type"),
                {"type": data_type}
            )
        else:
            result = conn.execute(text("SELECT * FROM procurement_data"))
        return pd.DataFrame(result.fetchall())

# ---------------- AI Integration ----------------
def ask_nvidia_ai(prompt, context="", history=""):
    """Call NVIDIA API for AI responses"""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    full_prompt = f"""
You are an assistant for a home textile production tracker. Answer using the provided context. If the answer is not in context, say you don't have enough data.

Context:
{context}

Conversation so far:
{history}

Question: {prompt}
Answer in 4-6 concise sentences:
"""
    
    payload = {
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful textile production tracking assistant."},
            {"role": "user", "content": full_prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error connecting to NVIDIA API: {str(e)}"

# ---------------- Login System ----------------
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
        st.rerun()

# ---------------- Navigation ----------------
if role == "admin":
    page = st.sidebar.radio(
        "Go to",
        [
            "Dashboard", "Planning", "Procurement", "Stores", "Upload",
            "Daily Status Update", "Visualize", "Logs", "Ask AI"
        ]
    )
elif role == "user":
    page = st.sidebar.radio("Go to", ["Dashboard", "Daily Status Update", "Visualize", "Logs", "Ask AI"])
elif role == "management":
    page = st.sidebar.radio("Go to", ["Dashboard", "Visualize", "Ask AI"])
elif role in ["planning", "procurement", "stores"]:
    page = st.sidebar.radio("Go to", ["Dashboard", role.capitalize(), "Visualize", "Ask AI"]) 
elif role in SECTION_ACCESS:
    page = st.sidebar.radio("Go to", ["Dashboard", "Daily Status Update", "Visualize", "Ask AI"]) 
else:
    page = st.sidebar.radio("Go to", ["Dashboard"])

# ---------------- Dashboard ----------------
if page == "Dashboard":
    st.title("🎯 Cloud Textile Tracker Dashboard")
    
    # Get data from database
    procurement_df = get_procurement_data()
    frozen_plans_df = get_frozen_plans()
    daily_status_df = get_daily_status()
    
    # Calculate metrics
    total_planned = len(procurement_df) if not procurement_df.empty else 0
    total_procured = len(procurement_df[procurement_df['status'] == 'Procured']) if not procurement_df.empty else 0
    total_received = len(procurement_df[procurement_df['received'] == 'Yes']) if not procurement_df.empty else 0
    
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Planned", total_planned)
    with c2:
        st.metric("Total Procured", total_procured)
    with c3:
        st.metric("Total Received", total_received)
    with c4:
        completion_rate = (total_received / max(total_planned, 1)) * 100
        st.metric("Overall Completion", f"{completion_rate:.1f}%")
    
    # Data visualization
    if not procurement_df.empty:
        st.markdown("---")
        st.subheader("📊 Material Analysis")
        
        # Status distribution
        status_counts = procurement_df['status'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, title="Procurement Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Type distribution
        type_counts = procurement_df['type'].value_counts()
        fig2 = px.bar(x=type_counts.index, y=type_counts.values, title="Material Types")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- Upload ----------------
if page == "Upload":
    st.title("Upload Monthly Target Plan")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        save_frozen_plan(df)
        st.success("Target Plan uploaded and saved to cloud database!")
        st.dataframe(df)

# ---------------- Planning ----------------
if page == "Planning":
    st.title("Planning Module")
    division = st.selectbox("Select Division", ["Procurement", "Production", "Dispatch"])
    
    if division == "Procurement":
        st.subheader("Procurement Requirement Entry")
        
        # Bulk upload
        bulk_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"], key="bulk_upload")
        if bulk_file:
            bulk_df = pd.read_excel(bulk_file) if bulk_file.name.endswith(".xlsx") else pd.read_csv(bulk_file)
            required_cols = {"Item Name", "Quantity", "Requested In-house Date", "Type"}
            if required_cols.issubset(set(bulk_df.columns)):
                for req_type in ["Raw Material", "Carton box", "Accessories"]:
                    df_type = bulk_df[bulk_df["Type"].str.lower() == req_type.lower()]
                    if not df_type.empty:
                        df_type = df_type.copy()
                        df_type["Status"] = "Requested"
                        save_procurement_data(req_type, df_type)
                st.success("Bulk requirements uploaded!")
            else:
                st.error(f"File must contain columns: {', '.join(required_cols)}")
        
        # Individual entry
        req_type = st.selectbox("Select Requirement Type", ["Raw Material", "Carton box", "Accessories"])
        with st.form(f"add_{req_type}_form"):
            item_name = st.text_input("Item Name")
            quantity = st.number_input("Quantity", min_value=1, step=1)
            req_date = st.date_input("Requested In-house Date", value=datetime.today())
            submit = st.form_submit_button("Add Requirement")
        
        if submit and item_name:
            new_row = pd.DataFrame([{
                "Item Name": item_name,
                "Quantity": quantity,
                "Requested In-house Date": req_date,
                "Status": "Requested"
            }])
            save_procurement_data(req_type, new_row)
            st.success(f"{req_type} requirement added.")

# ---------------- Procurement ----------------
elif page == "Procurement":
    st.title("Procurement Module")
    req_type = st.selectbox("Select Requirement Type", ["Raw Material", "Carton box", "Accessories"])
    data = get_procurement_data(req_type)
    
    if not data.empty:
        st.dataframe(data)
        idx = st.number_input("Select Row to Update", min_value=0, max_value=len(data)-1, step=1)
        inhouse_date = st.date_input("Procurement In-house Date", value=datetime.today())
        proc_qty = st.number_input("Procured Quantity", min_value=1, step=1, value=int(data.iloc[idx]["Quantity"]))
        
        if st.button("Update Procurement"):
            data.loc[idx, "Procurement In-house Date"] = inhouse_date
            data.loc[idx, "Procured Quantity"] = proc_qty
            data.loc[idx, "Status"] = "Procured"
            save_procurement_data(req_type, data)
            st.success("Procurement info updated.")

# ---------------- Stores ----------------
elif page == "Stores":
    st.title("Stores Module")
    req_type = st.selectbox("Select Item Type", ["Raw Material", "Carton box", "Accessories"])
    data = get_procurement_data(req_type)
    
    if not data.empty:
        st.dataframe(data)
        idx = st.number_input("Select Row to Confirm Receipt", min_value=0, max_value=len(data)-1, step=1)
        received = st.selectbox("Received?", ["No", "Yes"])
        rec_qty = st.number_input("Received Quantity", min_value=0, step=1, value=int(data.iloc[idx]["Quantity"]))
        rec_date = st.date_input("Actual In-house Date", value=datetime.today())
        
        if st.button("Confirm Receipt"):
            data.loc[idx, "Received"] = received
            data.loc[idx, "Received Quantity"] = rec_qty
            data.loc[idx, "Actual In-house Date"] = rec_date
            save_procurement_data(req_type, data)
            st.success("Receipt confirmed.")

# ---------------- Daily Status Update ----------------
if page == "Daily Status Update":
    st.title("📤 Daily Status Update")
    
    frozen_plans = get_frozen_plans()
    if frozen_plans.empty:
        st.error("No target plans uploaded yet.")
        st.stop()
    
    # Get unique POs from frozen plans
    po_list = []
    for _, row in frozen_plans.iterrows():
        data = json.loads(row['data'])
        po_list.append(data.get('PO No', ''))
    
    selected_po = st.selectbox("Select PO No", list(set(po_list)))
    
    # Get customer for selected PO
    customer = ""
    for _, row in frozen_plans.iterrows():
        data = json.loads(row['data'])
        if data.get('PO No') == selected_po:
            customer = data.get('Customer', '')
            break
    
    st.text_input("Customer", customer, disabled=True)
    
    # Process updates
    allowed_sections = process_options if role not in SECTION_ACCESS else SECTION_ACCESS[role]
    
    for process in process_options:
        if process not in allowed_sections:
            continue
            
        st.subheader(process)
        submitted_by = st.text_input("Your Name", key=f"submitter_{process}")
        start = st.date_input(f"{process} Start", key=f"start_{process}")
        finish = st.date_input(f"{process} Finish", key=f"finish_{process}")
        remarks = st.text_area("Remarks", key=f"remarks_{process}")
        
        if st.button(f"Submit {process}", key=f"submit_{process}"):
            if not submitted_by:
                st.error("Your name is mandatory")
            else:
                save_daily_status({
                    "po_no": selected_po,
                    "customer": customer,
                    "process": process,
                    "actual_start": start,
                    "actual_finish": finish,
                    "remarks": remarks,
                    "submitted_by": submitted_by
                })
                st.success(f"{process} updated for {selected_po}")

# ---------------- Visualize ----------------
if page == "Visualize":
    st.title("Process Completion Matrix")
    
    frozen_plans = get_frozen_plans()
    daily_status = get_daily_status()
    
    if frozen_plans.empty:
        st.warning("Please upload a target plan first.")
    else:
        # Create visualization matrix
        matrix = []
        for _, row in frozen_plans.iterrows():
            data = json.loads(row['data'])
            status_row = {"PO No": data.get('PO No', ''), "Customer": data.get('Customer', '')}
            
            allowed_sections = process_options if role not in SECTION_ACCESS else SECTION_ACCESS[role]
            
            for process in process_options:
                if process not in allowed_sections:
                    continue
                    
                plan_end = data.get(f"{process} Finish")
                
                # Check actual status
                actual_rows = daily_status[
                    (daily_status["po_no"] == data.get('PO No', '')) & 
                    (daily_status["process"] == process)
                ]
                
                if not actual_rows.empty:
                    actual_end = actual_rows.iloc[0]["actual_finish"]
                    if pd.notna(actual_end) and pd.notna(plan_end):
                        delay_days = (pd.to_datetime(actual_end) - pd.to_datetime(plan_end)).days
                        if delay_days > 0:
                            status = f"🟥 Delayed ({delay_days}d)"
                        elif delay_days < 0:
                            status = f"🟩 Early ({abs(delay_days)}d)"
                        else:
                            status = "🟨 On Time"
                    else:
                        status = "🔵 In Progress"
                else:
                    status = "⬜ Not Started"
                
                status_row[process] = status
            matrix.append(status_row)
        
        status_df = pd.DataFrame(matrix)
        st.dataframe(status_df, use_container_width=True)

# ---------------- Logs ----------------
if page == "Logs":
    st.title("Daily Logs")
    daily_status = get_daily_status()
    
    if not daily_status.empty:
        st.dataframe(daily_status)
        csv = daily_status.to_csv(index=False)
        st.download_button(
            "Download Log as CSV",
            csv,
            "log_export.csv",
            "text/csv"
        )
    else:
        st.warning("No logs available.")

# === Unlock POs / Unlock Process (admin placeholders retained) ===
if page == "Unlock POs" and role == "admin":
    st.title("Unlock PO Data for Editing")
    if os.path.exists(FREEZE_FILE):
        locked_pos = [po for po in pd.read_csv(FREEZE_FILE)["PO No"].unique()]
        if locked_pos:
            selected_unlock_po = st.selectbox("Select PO to Unlock", locked_pos)
            if st.button("Unlock Selected PO"):
                st.success(f"PO {selected_unlock_po} has been unlocked (placeholder logic).")
        else:
            st.info("No locked POs to unlock.")
    else:
        st.info("No plan uploaded yet.")

if page == "Unlock Process" and role == "admin":
    st.title("Unlock Individual Process")
    if os.path.exists(FREEZE_FILE):
        po_list = pd.read_csv(FREEZE_FILE)["PO No"].unique()
        selected_po = st.selectbox("Select PO", po_list)
        selected_process = st.selectbox("Select Process", process_options)
        if st.button("Unlock Process"):
            st.success(f"Process '{selected_process}' for PO '{selected_po}' has been unlocked (placeholder logic).")
    else:
        st.info("No plan uploaded yet.")

# ---------------- Ask AI ----------------
if page == "Ask AI":
    st.title("🤖 Ask AI – Textile Tracker Assistant")
    
    if st.button("🧹 Clear Conversation"):
        st.session_state.pop("chat_history", None)
        st.rerun()
    
    # Chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Render chat history
    for role_msg, content in st.session_state.chat_history:
        st.chat_message("user" if role_msg == "user" else "assistant").write(content)
    
    # Chat input
    user_query = st.chat_input("Ask about POs, delays, processes, customers...")
    
    if user_query:
        st.chat_message("user").write(user_query)
        st.session_state.chat_history.append(["user", user_query])
        
        # Prepare context from database
        frozen_plans = get_frozen_plans()
        daily_status = get_daily_status()
        procurement_data = get_procurement_data()
        
        context = ""
        
        # Add frozen plans context
        if not frozen_plans.empty:
            context += "Frozen Plans:\n"
            for _, row in frozen_plans.head(5).iterrows():
                data = json.loads(row['data'])
                context += f"PO: {data.get('PO No', '')}, Customer: {data.get('Customer', '')}\n"
        
        # Add daily status context
        if not daily_status.empty:
            context += "\nRecent Status Updates:\n"
            for _, row in daily_status.head(5).iterrows():
                context += f"PO: {row['po_no']}, Process: {row['process']}, Status: {row['actual_finish']}\n"
        
        history_text = "\n".join([f"{r.upper()}: {c}" for r, c in st.session_state.chat_history[-4:]])
        
        with st.spinner("Thinking…"):
            answer = ask_nvidia_ai(user_query, context, history_text)
        
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append(["assistant", answer])
