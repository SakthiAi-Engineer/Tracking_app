# textile_tracker_app.py - Improved Version with Enhanced Security & Code Quality
import streamlit as st
import pandas as pd
import os
import re
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import psutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# App Config
st.set_page_config(
    page_title="Home Textile Tracker", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Security Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = ['.xlsx', '.xls']
MAX_UPLOAD_ATTEMPTS = 3
SESSION_TIMEOUT = 30  # minutes

# File Constants
UPLOAD_DIR = Path("uploaded_plans")
FREEZE_FILE = Path("frozen_invoices.csv")
STATUS_FILE = Path("daily_status_updates.csv")
DAILY_LOG_DIR = Path("daily_logs")

# Process workflow constants
PROCESS_OPTIONS = [
    "Warping", "Sizing", "Weaving", "Greige Inspection", "Wet Processing",
    "Inspection", "Stitching", "Final Inspection", "Packing & Cartooning", "Shipment"
]

class SecurityManager:
    """Handles security-related operations"""
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 100) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(text, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';\\]', '', text)
        # Limit length
        sanitized = sanitized[:max_length]
        # Strip whitespace
        return sanitized.strip()
    
    @staticmethod
    def validate_po_number(po_no: str) -> bool:
        """Validate PO number format"""
        if not po_no:
            return False
        # Allow alphanumeric, hyphens, underscores
        return bool(re.match(r'^[A-Za-z0-9_-]+$', po_no))
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password for secure storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def validate_file_upload(uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded file for security"""
        if not uploaded_file:
            return False, "No file uploaded"
        
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            return False, f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024:.1f}MB"
        
        # Check file extension
        file_ext = Path(uploaded_file.name).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        
        return True, "Valid file"

class DataManager:
    """Handles data operations with error handling"""
    
    @staticmethod
    def safe_read_csv(file_path: Path) -> Optional[pd.DataFrame]:
        """Safely read CSV file with error handling"""
        try:
            if file_path.exists():
                return pd.read_csv(file_path)
            return None
        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {e}")
            st.error(f"Error reading file: {file_path.name}")
            return None
    
    @staticmethod
    def safe_write_csv(df: pd.DataFrame, file_path: Path, mode: str = 'w') -> bool:
        """Safely write CSV file with error handling"""
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if mode == 'a':
                df.to_csv(file_path, mode='a', index=False, header=not file_path.exists())
            else:
                df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Error writing CSV {file_path}: {e}")
            st.error(f"Error saving file: {file_path.name}")
            return False
    
    @staticmethod
    def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
        """Validate DataFrame has required columns"""
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        return True, "Valid structure"

class SessionManager:
    """Manages user sessions and authentication"""
    
    @staticmethod
    def initialize_session():
        """Initialize session state variables"""
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
        if "login_time" not in st.session_state:
            st.session_state.login_time = None
        if "failed_attempts" not in st.session_state:
            st.session_state.failed_attempts = 0
        if "enable_edits" not in st.session_state:
            st.session_state.enable_edits = {}
        if "unlocked_process" not in st.session_state:
            st.session_state.unlocked_process = {}
    
    @staticmethod
    def check_session_timeout() -> bool:
        """Check if session has timed out"""
        if not st.session_state.get("login_time"):
            return True
        
        time_diff = datetime.now() - st.session_state.login_time
        return time_diff > timedelta(minutes=SESSION_TIMEOUT)
    
    @staticmethod
    def logout():
        """Safely logout user"""
        logger.info(f"User {st.session_state.get('role', 'unknown')} logged out")
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def create_directories():
    """Create necessary directories"""
    try:
        UPLOAD_DIR.mkdir(exist_ok=True)
        DAILY_LOG_DIR.mkdir(exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        st.error("Error initializing application directories")

def authenticate_user() -> Optional[str]:
    """Handle user authentication with security measures"""
    SessionManager.initialize_session()
    
    # Check for session timeout
    if st.session_state.logged_in and SessionManager.check_session_timeout():
        st.warning("Session expired. Please login again.")
        SessionManager.logout()
    
    if not st.session_state.logged_in:
        st.sidebar.title("🔐 Login")
        
        # Check for too many failed attempts
        if st.session_state.failed_attempts >= MAX_UPLOAD_ATTEMPTS:
            st.sidebar.error("Too many failed attempts. Please refresh the page.")
            st.stop()
        
        username = st.sidebar.text_input("Username", max_chars=50)
        password = st.sidebar.text_input("Password", type="password", max_chars=100)
        login_btn = st.sidebar.button("Login")
        
        if login_btn:
            username = SecurityManager.sanitize_input(username)
            
            try:
                users = dict(st.secrets["users"])
                if username in users and users[username] == password:
                    st.session_state.role = username
                    st.session_state.logged_in = True
                    st.session_state.login_time = datetime.now()
                    st.session_state.failed_attempts = 0
                    logger.info(f"User {username} logged in successfully")
                    st.rerun()
                else:
                    st.session_state.failed_attempts += 1
                    logger.warning(f"Failed login attempt for user: {username}")
                    st.sidebar.error("Invalid credentials")
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                st.sidebar.error("Authentication system error")
        
        st.stop()
    
    return st.session_state.role

def render_sidebar_navigation(role: str) -> str:
    """Render sidebar navigation based on user role"""
    with st.sidebar:
        st.markdown("---")
        st.write(f"👤 Logged in as: **{role}**")
        
        if st.button("🚪 Logout"):
            SessionManager.logout()
        
        st.markdown("---")
        
        # Role-based navigation
        if role == "admin":
            pages = ["Upload", "Daily Status Update", "Visualize", "Logs", 
                    "Unlock POs", "Unlock Process", "Pie chart view"]
        elif role == "user":
            pages = ["Daily Status Update", "Visualize", "Logs"]
        elif role == "management":
            pages = ["Visualize", "Pie chart view"]
        else:
            pages = ["Visualize"]
        
        return st.radio("📋 Navigate to:", pages)

def render_system_monitor():
    """Render system monitoring information"""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        memory_mb = mem_info.rss / (1024 ** 2)
        
        st.sidebar.markdown("---")
        st.sidebar.title("📊 System Monitor")
        st.sidebar.metric("Memory Usage", f"{memory_mb:.1f} MB")
        
        # Add CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        st.sidebar.metric("CPU Usage", f"{cpu_percent:.1f}%")
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")

def handle_file_upload():
    """Handle secure file upload"""
    st.title("📤 Upload Target Plan")
    
    uploaded_file = st.file_uploader(
        "Upload Excel File", 
        type=["xlsx", "xls"],
        help="Upload your target plan in Excel format (max 10MB)"
    )
    
    if uploaded_file:
        # Validate file
        is_valid, message = SecurityManager.validate_file_upload(uploaded_file)
        
        if not is_valid:
            st.error(f"Upload failed: {message}")
            return
        
        try:
            # Read and validate Excel file
            df = pd.read_excel(uploaded_file)
            
            # Validate required columns
            required_cols = ["PO No", "Customer"]
            is_valid_structure, structure_message = DataManager.validate_dataframe_structure(df, required_cols)
            
            if not is_valid_structure:
                st.error(f"Invalid file structure: {structure_message}")
                return
            
            # Sanitize data
            df["PO No"] = df["PO No"].astype(str).apply(lambda x: SecurityManager.sanitize_input(x, 50))
            df["Customer"] = df["Customer"].astype(str).apply(lambda x: SecurityManager.sanitize_input(x, 100))
            
            # Save file
            if DataManager.safe_write_csv(df, FREEZE_FILE):
                st.success("✅ Target Plan uploaded and frozen successfully!")
                st.dataframe(df, use_container_width=True)
                logger.info(f"File uploaded successfully by {st.session_state.role}")
            
        except Exception as e:
            logger.error(f"File upload error: {e}")
            st.error("Error processing uploaded file. Please check the file format.")

def handle_daily_status_update():
    """Handle daily status updates with validation"""
    st.title("📤 Daily Status Update - Department Wise")
    
    # Check if frozen file exists
    frozen_df = DataManager.safe_read_csv(FREEZE_FILE)
    if frozen_df is None:
        st.error("❌ No target plans uploaded yet. Please upload a plan first.")
        return
    
    # Load status data
    status_df = DataManager.safe_read_csv(STATUS_FILE)
    if status_df is None:
        status_df = pd.DataFrame(columns=[
            "Timestamp", "PO No", "Customer", "Process", 
            "Actual Start", "Actual Finish", "Remarks", "Submitted By"
        ])
    
    # Get PO list and customer lookup
    invoice_list = frozen_df["PO No"].dropna().unique().tolist()
    customer_lookup = frozen_df.set_index("PO No")["Customer"].to_dict()
    
    if not invoice_list:
        st.warning("No PO numbers found in uploaded plan.")
        return
    
    # PO selection
    selected_invoice = st.selectbox("Select PO No to update:", invoice_list)
    selected_customer = customer_lookup.get(selected_invoice, "Unknown")
    st.text_input("Customer Name:", selected_customer, disabled=True)
    
    # Process updates
    invoice_status = status_df[status_df["PO No"] == selected_invoice]
    role = st.session_state.role
    
    for i, process in enumerate(PROCESS_OPTIONS):
        with st.expander(f"🔧 {process}", expanded=False):
            row = invoice_status[invoice_status["Process"] == process]
            
            # Check if previous process is complete
            prev_process_complete = True
            if i > 0:
                prev_process = PROCESS_OPTIONS[i - 1]
                prev_row = invoice_status[invoice_status["Process"] == prev_process]
                prev_process_complete = not prev_row.empty and pd.notna(prev_row.iloc[0]["Actual Finish"])
            
            # Check edit permissions
            can_edit = (
                role == "admin" or 
                selected_invoice in st.session_state.enable_edits or
                (selected_invoice, process) in st.session_state.unlocked_process
            )
            
            if not prev_process_complete:
                st.warning(f"⚠️ Complete previous process first: {PROCESS_OPTIONS[i-1]}")
                continue
            
            if not can_edit and role != "admin":
                st.info("🔒 This process is locked. Contact admin to unlock.")
                continue
            
            # Input fields
            col1, col2 = st.columns(2)
            
            with col1:
                submitted_by = st.text_input(
                    "Your Name (required):", 
                    key=f"submitter_{process}_{selected_invoice}",
                    value=row.iloc[0]["Submitted By"] if not row.empty else "",
                    max_chars=100
                )
                
                start_default = None
                if not row.empty and pd.notna(row.iloc[0]["Actual Start"]):
                    start_default = pd.to_datetime(row.iloc[0]["Actual Start"]).date()
                
                actual_start = st.date_input(
                    "Actual Start Date:",
                    key=f"start_{process}_{selected_invoice}",
                    value=start_default
                )
            
            with col2:
                finish_default = None
                if not row.empty and pd.notna(row.iloc[0]["Actual Finish"]):
                    finish_default = pd.to_datetime(row.iloc[0]["Actual Finish"]).date()
                
                actual_finish = st.date_input(
                    "Actual Finish Date:",
                    key=f"finish_{process}_{selected_invoice}",
                    value=finish_default
                )
            
            remarks = st.text_area(
                "Remarks:",
                key=f"remarks_{process}_{selected_invoice}",
                value=row.iloc[0]["Remarks"] if not row.empty else "",
                max_chars=500
            )
            
            # Submit button
            if st.button(f"💾 Submit {process}", key=f"submit_{process}_{selected_invoice}"):
                # Validate inputs
                submitted_by = SecurityManager.sanitize_input(submitted_by, 100)
                remarks = SecurityManager.sanitize_input(remarks, 500)
                
                if not submitted_by:
                    st.error("❌ Your name is required to submit.")
                    continue
                
                if actual_start and actual_finish and actual_start > actual_finish:
                    st.error("❌ Start date cannot be after finish date.")
                    continue
                
                # Create new record
                new_row = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "PO No": selected_invoice,
                    "Customer": selected_customer,
                    "Process": process,
                    "Actual Start": actual_start.strftime("%Y-%m-%d") if actual_start else "",
                    "Actual Finish": actual_finish.strftime("%Y-%m-%d") if actual_finish else "",
                    "Remarks": remarks,
                    "Submitted By": submitted_by
                }
                
                # Update status file
                if DataManager.safe_write_csv(pd.DataFrame([new_row]), STATUS_FILE, mode='a'):
                    st.success(f"✅ {process} updated successfully for {selected_invoice}")
                    logger.info(f"Process {process} updated for PO {selected_invoice} by {submitted_by}")
                    st.rerun()

def main():
    """Main application function"""
    try:
        # Initialize
        create_directories()
        
        # Authenticate user
        role = authenticate_user()
        if not role:
            return
        
        # Render navigation
        page = render_sidebar_navigation(role)
        
        # Render system monitor
        render_system_monitor()
        
        # Route to appropriate page
        if page == "Upload" and role == "admin":
            handle_file_upload()
        elif page == "Daily Status Update":
            handle_daily_status_update()
        elif page == "Visualize":
            st.title("📊 Visualization")
            st.info("Visualization features will be implemented here")
        elif page == "Logs":
            st.title("📋 Logs")
            st.info("Logs features will be implemented here")
        elif page == "Pie chart view":
            st.title("🥧 Pie Chart View")
            st.info("Pie chart features will be implemented here")
        elif page == "Unlock POs" and role == "admin":
            st.title("🔓 Unlock POs")
            st.info("PO unlock features will be implemented here")
        elif page == "Unlock Process" and role == "admin":
            st.title("🔓 Unlock Process")
            st.info("Process unlock features will be implemented here")
        else:
            st.error("❌ Access denied or page not found")
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please contact support.")

if __name__ == "__main__":
    main()
