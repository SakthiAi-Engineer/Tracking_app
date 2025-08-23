import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import psycopg2
from sqlalchemy import create_engine, text
import json
import requests
from passlib.context import CryptContext

# ---------------- Password Hashing Setup ----------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- Cloud Configuration ----------------
try:
    db_url = st.secrets["DATABASE_URL"]
    engine = create_engine(
        db_url,
        connect_args={"sslmode": "require"},
        pool_pre_ping=True
    )
    # Test connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT NOW();"))
        # Using a session state flag to show connection status only once
        if 'db_connected' not in st.session_state:
            st.success(f"✅ Database connected. Server time: {result.scalar()}")
            st.session_state.db_connected = True
except Exception as e:
    st.error(f"❌ DATABASE CONNECTION FAILED: {e}")
    st.error("Please ensure your DATABASE_URL is correctly configured in Streamlit secrets.")
    st.stop()

# Database and API configurations from secrets
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# ---------------- Database Initialization ----------------
def get_db_connection():
    """Returns a new SQLAlchemy engine instance."""
    return create_engine(
        st.secrets["DATABASE_URL"],
        connect_args={"sslmode": "require"},
        pool_pre_ping=True
    )

def init_database():
    # """Initialize database tables if they don't exist."""
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR(50) PRIMARY KEY,
                hashed_password VARCHAR(255),
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
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS unlocked_status (
                id SERIAL PRIMARY KEY,
                po_no VARCHAR(50),
                process VARCHAR(50),
                unlocked BOOLEAN DEFAULT FALSE,
                UNIQUE (po_no, process)
            )
        """))
        conn.commit()

def insert_default_users():
    # """Insert default users with HASHED passwords if no users exist."""
    engine = get_db_connection()
    with engine.connect() as conn:
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
                hashed_password = pwd_context.hash(password)
                conn.execute(
                    text("INSERT INTO users (username, hashed_password, role) VALUES (:username, :hashed_password, :role)"),
                    {"username": username, "hashed_password": hashed_password, "role": role}
                )
            conn.commit()

# ---------------- App Config ----------------
st.set_page_config(page_title="Cloud Textile Tracker", layout="wide")
init_database()
insert_default_users()

# ---------------- Constants ----------------
SECTION_ACCESS = {
    "external_weaving": ["External Order", "Weaving"],
    "quality": ["Greige Inspection", "Inspection", "Final Inspection"],
    "processing": ["Processing"],
    "stitch_pack": ["Stitching", "Packing & Cartoning"],
    "dispatch": ["Shipment"],
}
PROCESS_OPTIONS = [
    "External Order", "Weaving", "Greige Inspection", "Processing",
    "Inspection", "Stitching", "Final Inspection", "Packing & Cartoning", "Shipment"
]

# ---------------- Security & User Management ----------------
def verify_user(username, password):
    # """Verify user against the database with hashed passwords."""
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT hashed_password FROM users WHERE username = :username"),
            {"username": username}
        )
        user_row = result.fetchone()
        if user_row and pwd_context.verify(password, user_row[0]):
            role_result = conn.execute(
                text("SELECT role FROM users WHERE username = :username"),
                {"username": username}
            )
            return role_result.scalar()
    return None

# ---------------- Database Operations ----------------
def save_frozen_plan(df):
    # """Save frozen plan, replacing the old one. Handles date serialization."""
    engine = get_db_connection()
    with engine.connect() as conn:
        # This function replaces the entire plan, so deleting first is the intended behavior.
        conn.execute(text("DELETE FROM frozen_plans"))
        conn.commit() # Commit the delete before inserting new data
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            # Convert any datetime/timestamp objects to ISO strings for JSON compatibility
            for key, value in row_dict.items():
                if isinstance(value, (datetime, pd.Timestamp)):
                    row_dict[key] = value.isoformat()
            
            conn.execute(
                text("INSERT INTO frozen_plans (po_no, customer, data) VALUES (:po_no, :customer, :data)"),
                {
                    "po_no": str(row.get('PO No', '')),
                    "customer": str(row.get('Customer', '')),
                    "data": json.dumps(row_dict)
                }
            )
        conn.commit()

def get_frozen_plans():
    # """Get all frozen plans."""
    engine = get_db_connection()
    query = text("SELECT * FROM frozen_plans ORDER BY uploaded_at DESC")
    return pd.read_sql(query, engine)

def save_daily_status(data):
    # """Save a single daily status update."""
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO daily_status 
            (po_no, customer, process, actual_start, actual_finish, remarks, submitted_by)
            VALUES (:po_no, :customer, :process, :actual_start, :actual_finish, :remarks, :submitted_by)
        """), data)
        conn.commit()

def get_daily_status():
    # """Get all daily status updates."""
    engine = get_db_connection()
    query = text("SELECT * FROM daily_status ORDER BY timestamp DESC")
    return pd.read_sql(query, engine)

def add_procurement_item(item_data):
    # """Adds a single new procurement item to the database."""
    engine = get_db_connection()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO procurement_data 
            (type, item_name, quantity, requested_in_house_date, status)
            VALUES (:type, :item_name, :quantity, :requested_in_house_date, 'Requested')
        """), item_data)
        conn.commit()

def save_bulk_procurement_data(df):
    # """Saves a DataFrame of new procurement items."""
    engine = get_db_connection()
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO procurement_data 
                (type, item_name, quantity, requested_in_house_date, status)
                VALUES (:type, :item_name, :quantity, :requested_in_house_date, :status)
            """), {
                "type": row.get('Type', ''),
                "item_name": str(row.get('Item Name', '')),
                "quantity": int(row.get('Quantity', 0)),
                "requested_in_house_date": row.get('Requested In-house Date'),
                "status": "Requested"
            })
        conn.commit()

def update_procurement_item(item_id, updates):
    # """Updates a specific procurement item by its ID."""
    engine = get_db_connection()
    set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
    updates['id'] = item_id
    with engine.connect() as conn:
        conn.execute(text(f"UPDATE procurement_data SET {set_clause} WHERE id = :id"), updates)
        conn.commit()

def get_procurement_data(data_type=None):
    # """Get procurement data, optionally filtered by type."""
    engine = get_db_connection()
    if data_type:
        query = text("SELECT * FROM procurement_data WHERE type = :type ORDER BY id")
        params = {"type": data_type}
    else:
        query = text("SELECT * FROM procurement_data ORDER BY id")
        params = {}
    return pd.read_sql(query, engine, params=params)
def unlock_process(po_no, process):
    engine = get_db_connection()
    with engine.connect() as conn:
        # Upsert unlock status
        conn.execute(text("""
            INSERT INTO unlocked_status (po_no, process, unlocked)
            VALUES (:po_no, :process, TRUE)
            ON CONFLICT (po_no, process)
            DO UPDATE SET unlocked = TRUE
        """), {"po_no": po_no, "process": process})
        conn.commit()

def is_process_unlocked(po_no, process):
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT unlocked FROM unlocked_status
            WHERE po_no = :po_no AND process = :process
        """), {"po_no": po_no, "process": process})
        result_row = result.fetchone()
        return result_row is not None and result_row[0] == True

def unlock_po(po_no):
    engine = get_db_connection()
    with engine.connect() as conn:
        for process in PROCESS_OPTIONS:
            conn.execute(text("""
                INSERT INTO unlocked_status (po_no, process, unlocked)
                VALUES (:po_no, :process, TRUE)
                ON CONFLICT (po_no, process)
                DO UPDATE SET unlocked = TRUE
            """), {"po_no": po_no, "process": process})
        conn.commit()
def get_all_unlocked():
    engine_local = get_db_connection()
    query = text("SELECT po_no, process FROM unlocked_status WHERE unlocked = TRUE")
    df = pd.read_sql(query, engine_local)
    return set((row.po_no, row.process) for row in df.itertuples())



# ---------------- AI Integration ----------------
def ask_nvidia_ai(prompt, context="", history=""):
    # """Call NVIDIA API for AI responses."""
    if not NVIDIA_API_KEY:
        return "Error: NVIDIA_API_KEY is not configured in secrets."
    
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
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error connecting to NVIDIA API: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# ---------------- Login System ----------------
if "logged_in" not in st.session_state:
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        role = verify_user(username, password)
        if role:
            st.session_state["role"] = role
            st.session_state["username"] = username
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password")
    st.stop()

role = st.session_state["role"]

with st.sidebar:
    st.write(f"Logged in as: **{st.session_state['username']}** (`{role}`)")
    st.markdown("---")
    if st.button("Logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
# For unlocking functionality (POs and processes)
if "enable_edits" not in st.session_state:
    st.session_state.enable_edits = {}  # example: {po_no: True}
if "unlocked_process" not in st.session_state:
    st.session_state.unlocked_process = {}  # example: {(po_no, process): True}
# ---------------- Navigation ----------------
if role == "admin":
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Upload", "Planning", "Procurement", "Stores", 
         "Daily Status Update", "Visualize", "Logs", "Ask AI" , "Unlock" ]
    )
elif role == "planning":
    page = st.sidebar.radio("Go to", ["Dashboard", "Planning", "Visualize", "Ask AI"])
elif role == "procurement":
    page = st.sidebar.radio("Go to", ["Dashboard", "Procurement", "Visualize", "Ask AI"])
elif role == "stores":
    page = st.sidebar.radio("Go to", ["Dashboard", "Stores", "Visualize", "Ask AI"])
elif role == "management":
    page = st.sidebar.radio("Go to", ["Dashboard", "Visualize", "Logs", "Ask AI"])
elif role in SECTION_ACCESS:
    page = st.sidebar.radio("Go to", ["Dashboard", "Daily Status Update", "Visualize", "Ask AI"])
else: # Includes 'user' role
    page = st.sidebar.radio("Go to", ["Dashboard", "Visualize", "Logs", "Ask AI"])

# ---------------- Main App Pages ----------------

if page == "Dashboard":
    st.title("🎯 Cloud Textile Tracker Dashboard")
    procurement_df = get_procurement_data()
    
    if not procurement_df.empty:
        total_planned = len(procurement_df)
        total_procured = len(procurement_df[procurement_df['status'] == 'Procured'])
        total_received = len(procurement_df[procurement_df['received'] == 'Yes'])
        completion_rate = (total_received / max(total_planned, 1)) * 100
    else:
        total_planned = total_procured = total_received = 0
        completion_rate = 0

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Planned Items", total_planned)
    c2.metric("Items Procured", total_procured)
    c3.metric("Items Received", total_received)
    c4.metric("Overall Completion", f"{completion_rate:.1f}%")

    if not procurement_df.empty:
        st.markdown("---")
        st.subheader("📊 Material Analysis")
        col1, col2 = st.columns(2)
        with col1:
            status_counts = procurement_df['status'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index, title="Procurement Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            type_counts = procurement_df['type'].value_counts()
            fig2 = px.bar(x=type_counts.index, y=type_counts.values, title="Material Types", labels={'x': 'Type', 'y': 'Count'})
            st.plotly_chart(fig2, use_container_width=True)

elif page == "Upload":
    st.title("📤 Upload Monthly Target Plan")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            save_frozen_plan(df)
            st.success("Target Plan uploaded and saved to cloud database!")
            st.subheader("📋 Preview of Uploaded Data")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

elif page == "Planning":
    st.title("Planning Module")
    st.subheader("Procurement Requirement Entry")
    
    # Bulk upload
    with st.expander("Bulk Upload from Excel/CSV"):
        bulk_file = st.file_uploader("Upload File", type=["xlsx", "csv"], key="bulk_upload")
        if bulk_file:
            bulk_df = pd.read_excel(bulk_file) if bulk_file.name.endswith(".xlsx") else pd.read_csv(bulk_file)
            required_cols = {"Item Name", "Quantity", "Requested In-house Date", "Type"}
            if required_cols.issubset(set(bulk_df.columns)):
                save_bulk_procurement_data(bulk_df)
                st.success("Bulk requirements uploaded!")
                st.dataframe(bulk_df.head())
            else:
                st.error(f"File must contain columns: {', '.join(required_cols)}")

    # Individual entry
    st.markdown("---")
    st.subheader("Add Individual Requirement")
    req_type = st.selectbox("Select Requirement Type", ["Raw Material", "Carton box", "Accessories"])
    with st.form(f"add_{req_type}_form", clear_on_submit=True):
        item_name = st.text_input("Item Name")
        quantity = st.number_input("Quantity", min_value=1, step=1)
        req_date = st.date_input("Requested In-house Date", value=datetime.today())
        submit = st.form_submit_button("Add Requirement")
        if submit and item_name:
            add_procurement_item({
                "type": req_type,
                "item_name": item_name,
                "quantity": quantity,
                "requested_in_house_date": req_date
            })
            st.success(f"{req_type} requirement for '{item_name}' added.")

elif page == "Procurement":
    st.title("Procurement Module")
    req_type = st.selectbox("Select Requirement Type", ["Raw Material", "Carton box", "Accessories"])
    data = get_procurement_data(req_type)
    
    if not data.empty:
        st.dataframe(data)
        item_ids = data['id'].tolist()
        selected_id = st.selectbox("Select Item ID to Update", item_ids)
        
        selected_row = data[data['id'] == selected_id].iloc[0]
        
        with st.form(f"update_proc_{selected_id}"):
            inhouse_date = st.date_input("Procurement In-house Date", value=pd.to_datetime(selected_row['procurement_in_house_date'] or datetime.today()).date())
            proc_qty = st.number_input("Procured Quantity", min_value=0, step=1, value=int(selected_row['procured_quantity'] or selected_row['quantity']))
            if st.form_submit_button("Update Procurement Status to 'Procured'"):
                update_procurement_item(selected_id, {
                    "procurement_in_house_date": inhouse_date,
                    "procured_quantity": proc_qty,
                    "status": "Procured"
                })
                st.success(f"Procurement info updated for item ID {selected_id}.")
                st.rerun()
    else:
        st.info(f"No procurement requests for '{req_type}'.")

elif page == "Stores":
    st.title("Stores Module")
    req_type = st.selectbox("Select Item Type", ["Raw Material", "Carton box", "Accessories"])
    data = get_procurement_data(req_type)
    
    if not data.empty:
        st.dataframe(data)
        item_ids = data['id'].tolist()
        selected_id = st.selectbox("Select Item ID to Confirm Receipt", item_ids)

        selected_row = data[data['id'] == selected_id].iloc[0]

        with st.form(f"update_stores_{selected_id}"):
            received = st.selectbox("Received?", ["No", "Yes"], index=1 if selected_row['received'] == 'Yes' else 0)
            rec_qty = st.number_input("Received Quantity", min_value=0, step=1, value=int(selected_row['received_quantity'] or selected_row['procured_quantity'] or 0))
            rec_date = st.date_input("Actual In-house Date", value=pd.to_datetime(selected_row['actual_in_house_date'] or datetime.today()).date())
            if st.form_submit_button("Confirm Receipt"):
                update_procurement_item(selected_id, {
                    "received": received,
                    "received_quantity": rec_qty,
                    "actual_in_house_date": rec_date
                })
                st.success(f"Receipt confirmed for item ID {selected_id}.")
                st.rerun()
    else:
        st.info(f"No items of type '{req_type}' to receive.")

elif page == "Daily Status Update":
    st.title("📤 Daily Status Update")
    frozen_plans_df = get_frozen_plans()
    if frozen_plans_df.empty:
        st.error("No target plans uploaded yet. Please go to the 'Upload' page.")
        st.stop()
    
    po_list = sorted(frozen_plans_df['po_no'].unique().tolist())
    selected_po = st.selectbox("Select PO No", po_list)
    
    customer = frozen_plans_df[frozen_plans_df['po_no'] == selected_po]['customer'].iloc[0]
    st.text_input("Customer", customer, disabled=True)
    
    allowed_sections = PROCESS_OPTIONS if role not in SECTION_ACCESS else SECTION_ACCESS[role]
    daily_status = get_daily_status()
    
    for process in allowed_sections:
       with st.expander(f"Update Status for: {process}"):
        # Fetch existing status for this PO+process from DB
        existing_status = daily_status[
            (daily_status["po_no"] == selected_po) & 
            (daily_status["process"] == process)
        ]
        
        # Retrieve existing start/finish dates if available
        existing_start = existing_status.iloc[0]['actual_start'] if not existing_status.empty else None
        existing_finish = existing_status.iloc['actual_finish'] if not existing_status.empty else None
        
        # Check unlock status for the process
        unlocked_proc = is_process_unlocked(selected_po, process)

        # Determine disabled state for each date input and remarks
        start_disabled = (existing_start is not None) and (not unlocked_proc)
        finish_disabled = (existing_finish is not None) and (not unlocked_proc)
        remarks_disabled = (existing_status.empty == False) and (not unlocked_proc)

        # Show date inputs with proper default value or None (blank)
        start = st.date_input(
            "Actual Start Date",
            value=existing_start if existing_start is not None else None,
            disabled=start_disabled,
            key=f"start_{process}_{selected_po}",
            help="Start date is frozen if already submitted and not unlocked"
        )
        finish = st.date_input(
            "Actual Finish Date",
            value=existing_finish if existing_finish is not None else None,
            disabled=finish_disabled,
            key=f"finish_{process}_{selected_po}",
            help="Finish date is frozen if already submitted and not unlocked"
        )

        remarks = st.text_area(
            "Remarks",
            value=existing_status.iloc[0]['remarks'] if not existing_status.empty else "",
            disabled=remarks_disabled,
            key=f"remarks_{process}_{selected_po}"
        )

        # If form fields are frozen (disabled), disable submit button too
        submit_disabled = start_disabled and finish_disabled and remarks_disabled

        if not submit_disabled:
            if st.button(f"Submit for {process}", key=f"submit_{process}_{selected_po}"):
                save_daily_status({
                    "po_no": selected_po,
                    "customer": customer,
                    "process": process,
                    "actual_start": start,
                    "actual_finish": finish,
                    "remarks": remarks,
                    "submitted_by": st.session_state["username"]
                })
                st.success(f"{process} status updated for PO {selected_po}")
                st.experimental_rerun()  # Refresh to reflect freezing changes
        else:
            st.info("This status is frozen and cannot be edited.")

    # for process in allowed_sections:
    #     with st.expander(f"Update Status for: {process}"):
    #            # 1. Check if status for this PO+process is already in daily_status
    #      existing_status = daily_status[
    #          (daily_status["po_no"] == selected_po) &
    #          (daily_status["process"] == process)
    #      ]
    #      unlocked_proc = is_process_unlocked(selected_po, process)

    #      if not existing_status.empty and not unlocked_proc:
    #         # Display "frozen" info (the existing date)
    #         latest = existing_status.iloc[0]
    #         st.success(f"Already submitted:\n\n- Start: {latest['actual_start']}\n- Finish: {latest['actual_finish']}\n- Remarks: {latest['remarks']}")
    #         st.info("This status is frozen and cannot be edited.")
    #         # Optionally show read-only input fields:
    #         st.date_input("Actual Start Date", value=latest['actual_start'], disabled=True)
    #         st.date_input("Actual Finish Date", value=latest['actual_finish'], disabled=True)
    #         st.text_area("Remarks", value=latest['remarks'], disabled=True)
    #      else:
    #         # Show editable form as in your original code
    #         with st.form(key=f"form_{process}_{selected_po}", clear_on_submit=True):
    #             start = st.date_input("Actual Start Date", key=f"start_{process}")
    #             finish = st.date_input("Actual Finish Date", key=f"finish_{process}")
    #             remarks = st.text_area("Remarks", key=f"remarks_{process}")
    #             if st.form_submit_button(f"Submit for {process}"):
    #                 save_daily_status({
    #                     "po_no": selected_po, "customer": customer, "process": process,
    #                     "actual_start": start, "actual_finish": finish, "remarks": remarks,
    #                     "submitted_by": st.session_state["username"]
    #                 })
    #                 st.success(f"{process} status updated for PO {selected_po}")
            # with st.form(key=f"form_{process}_{selected_po}", clear_on_submit=True):
            #     start = st.date_input("Actual Start Date", key=f"start_{process}")
            #     finish = st.date_input("Actual Finish Date", key=f"finish_{process}")
            #     remarks = st.text_area("Remarks", key=f"remarks_{process}")
            #     if st.form_submit_button(f"Submit for {process}"):
            #         save_daily_status({
            #             "po_no": selected_po, "customer": customer, "process": process,
            #             "actual_start": start, "actual_finish": finish, "remarks": remarks,
            #             "submitted_by": st.session_state["username"]
            #         })
            #         st.success(f"{process} status updated for PO {selected_po}")

elif page == "Visualize":
    st.title("📊 Process Completion Matrix")
    frozen_plans = get_frozen_plans()
    daily_status = get_daily_status()
    
    if frozen_plans.empty:
        st.warning("Please upload a target plan first.")
    else:
        matrix = []
        for _, plan_row in frozen_plans.iterrows():
            # FIX: Do not use json.loads() here. 'data' is already a dict.
            plan_data = plan_row['data'] 
            status_row = {"PO No": plan_data.get('PO No', ''), "Customer": plan_data.get('Customer', '')}
            
            for process in PROCESS_OPTIONS:
                plan_end_str = plan_data.get(f"{process} Finish")
                
                actual_updates = daily_status[
                    (daily_status["po_no"] == plan_data.get('PO No', '')) & 
                    (daily_status["process"] == process)
                ]
                
                status = "⬜ Not Started"
                if not actual_updates.empty:
                    latest_update = actual_updates.iloc[0]
                    actual_end = latest_update["actual_finish"]
                    if actual_end and plan_end_str:
                        try:
                            delay_days = (pd.to_datetime(actual_end).date() - pd.to_datetime(plan_end_str).date()).days
                            if delay_days > 0: status = f"🟥 Delayed ({delay_days}d)"
                            elif delay_days < 0: status = f"🟩 Early ({abs(delay_days)}d)"
                            else: status = "🟨 On Time"
                        except (ValueError, TypeError):
                            status = "🔵 In Progress (Date Error)"
                    else:
                        status = "🔵 In Progress"
                status_row[process] = status
            matrix.append(status_row)
            
        status_df = pd.DataFrame(matrix)
        st.dataframe(status_df, use_container_width=True)

elif page == "Logs":
    st.title("📋 Daily Status Logs")
    daily_status = get_daily_status()
    if not daily_status.empty:
        st.dataframe(daily_status)
        csv = daily_status.to_csv(index=False).encode('utf-8')
        st.download_button("Download Log as CSV", csv, "log_export.csv", "text/csv")
    else:
        st.warning("No logs available.")

elif page == "Ask AI":
    st.title("🤖 Ask AI – Textile Tracker Assistant")
    
    if st.button("🧹 Clear Conversation"):
        st.session_state.pop("chat_history", None)
        st.rerun()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role_msg, content in st.session_state.chat_history:
        with st.chat_message("user" if role_msg == "user" else "assistant"):
            st.write(content)
            
    if user_query := st.chat_input("Ask about POs, delays, processes, customers..."):
        st.chat_message("user").write(user_query)
        st.session_state.chat_history.append(["user", user_query])
        
        with st.spinner("Thinking…"):
            # Prepare context
            frozen_plans = get_frozen_plans()
            daily_status = get_daily_status()
            
            context = "Frozen Plans Summary:\n"
            if not frozen_plans.empty:
                for _, row in frozen_plans.head(5).iterrows():
                    # FIX: Do not use json.loads() here. 'data' is already a dict.
                    data = row['data'] 
                    context += f"- PO: {data.get('PO No', 'N/A')}, Customer: {data.get('Customer', 'N/A')}\n"
            
            context += "\nRecent Status Updates:\n"
            if not daily_status.empty:
                for _, row in daily_status.head(5).iterrows():
                    finish_date = row['actual_finish'] or "In Progress"
                    context += f"- PO: {row['po_no']}, Process: {row['process']}, Finish Date: {finish_date}\n"
            
            history_text = "\n".join([f"{r.upper()}: {c}" for r, c in st.session_state.chat_history[-4:]])
            
            answer = ask_nvidia_ai(user_query, context, history_text)
        
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append(["assistant", answer])
#Unlock Page
elif page == "Unlock" and role == "admin":
    st.title("🔓 Unlock PO or Process for Editing")

    frozen_plans_df = get_frozen_plans()
    all_pos = sorted(frozen_plans_df['po_no'].unique().tolist())

    # Fetch all unlocked PO-process pairs once
    unlocked_set = get_all_unlocked()

    # Compute locked POs efficiently
    locked_pos = []
    for po in all_pos:
        # If no process for this PO is unlocked, then it is locked
        if not any((po, p) in unlocked_set for p in PROCESS_OPTIONS):
            locked_pos.append(po)

    if locked_pos:
        selected_unlock_po = st.selectbox("Select PO to Unlock", locked_pos)
        if st.button("Unlock Selected PO"):
            unlock_po(selected_unlock_po)
            st.success(f"PO {selected_unlock_po} has been unlocked for editing.")
    else:
        st.info("All POs are currently unlocked")

    st.markdown("---")

    st.subheader("Unlock Individual Process")
    selected_po = st.selectbox("Select PO", all_pos, key="unlock_proc_po")
    selected_process = st.selectbox("Select Process", PROCESS_OPTIONS, key="unlock_proc_process")
    if st.button("Unlock Process"):
        unlock_process(selected_po, selected_process)
        st.success(f"Process '{selected_process}' for PO '{selected_po}' has been unlocked for editing.")
