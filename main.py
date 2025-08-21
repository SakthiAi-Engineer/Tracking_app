# textile_tracker_app.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import psutil

# App Config
st.set_page_config(page_title="Home Textile Tracker", layout="wide")

# Constants
UPLOAD_DIR = "uploaded_plans"
FREEZE_FILE = "frozen_invoices.csv"
STATUS_FILE = "daily_status_updates.csv"
DAILY_LOG_DIR = "daily_logs"
USERS = dict(st.secrets["users"])

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DAILY_LOG_DIR, exist_ok=True)

# Login
if "logged_in" not in st.session_state:
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    login_btn = st.sidebar.button("Login")

    if login_btn:
        if username in USERS and USERS[username] == password:
            st.session_state["role"] = username
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.sidebar.error("Invalid login")
    st.stop()

role = st.session_state["role"]

# Logout Button
with st.sidebar:
    st.markdown("---")
    if st.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# Sidebar Navigation
if role == "admin":
    page = st.sidebar.radio("Go to", [
        "Upload", "Daily Status Update", "Visualize", "Logs", "Unlock POs", "Unlock Process"
    , "Pie chart view"])
elif role == "user":
    page = st.sidebar.radio("Go to", [
        "Daily Status Update", "Visualize", "Logs"
    ])
elif role == "management":
    page = st.sidebar.radio("Go to", [
        "Visualize", "Pie chart view"
    ])

process_options = [
    "Warping", "Sizing", "Weaving", "Greige Inspection", "Wet Processing",
    "Inspection", "Stitching", "Final Inspection", "Packing & Cartooning", "Shipment"
]

# Enable editing for admin
if "enable_edits" not in st.session_state:
    st.session_state.enable_edits = {}
if "unlocked_process" not in st.session_state:
    st.session_state.unlocked_process = {}
if role == "admin":
    st.sidebar.markdown("---")
    toggle_po = st.sidebar.text_input("Enable PO for Editing")
    if st.sidebar.button("Enable PO") and toggle_po:
        st.session_state.enable_edits[toggle_po] = True
    toggle_process = st.sidebar.text_input("Enable Process for Editing (PO_No:Process)")
    if st.sidebar.button("Enable Process") and toggle_process:
        po, proc = toggle_process.split(":")
        st.session_state.unlocked_process[(po.strip(), proc.strip())] = True

# ---------------- PAGE 1: UPLOAD -----------------
if page == "Upload":
    st.title("Upload Target Plan")
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df.to_csv(FREEZE_FILE, index=False)
        st.success("Target Plan uploaded and frozen!")
        st.dataframe(df)

# ---------------- PAGE 2: DAILY STATUS UPDATE -----------------
if page == "Daily Status Update":
    st.title("📤 Daily Status Update - Department Wise")
    if not os.path.exists(FREEZE_FILE):
        st.error("No target plans uploaded and frozen yet.")
        st.stop()

    frozen_df = pd.read_csv(FREEZE_FILE)
    invoice_list = frozen_df["PO No"].dropna().unique().tolist()
    customer_lookup = frozen_df.set_index("PO No")["Customer"].to_dict()
    status_df = pd.read_csv(STATUS_FILE) if os.path.exists(STATUS_FILE) else pd.DataFrame(
        columns=["Timestamp", "PO No", "Customer", "Process", "Actual Start", "Actual Finish", "Remarks", "Submitted By"]
    )

    selected_invoice = st.selectbox("Select PO No to update or edit", invoice_list)
    selected_customer = customer_lookup.get(selected_invoice, "")
    st.text_input("Customer Name", selected_customer, disabled=True)

    invoice_status = status_df[status_df["PO No"] == selected_invoice]
    can_edit_po = selected_invoice in st.session_state.enable_edits or role == "admin"

    for i, process in enumerate(process_options):
        row = invoice_status[invoice_status["Process"] == process]
        prev_process_complete = True
        if i > 0:
            prev = process_options[i - 1]
            prev_row = invoice_status[invoice_status["Process"] == prev]
            prev_process_complete = not prev_row.empty and pd.notna(prev_row.iloc[0]["Actual Finish"])

        is_editable = ((selected_invoice, process) in st.session_state.unlocked_process) or \
                      (can_edit_po and (row.empty or pd.isna(row.iloc[0]["Actual Start"]) or pd.isna(row.iloc[0]["Actual Finish"])) and prev_process_complete)

        st.subheader(process)

        if not row.empty and pd.notna(row.iloc[0]["Actual Start"]):
            st.markdown(f"**Start Date:** {row.iloc[0]['Actual Start']}")
        if not row.empty and pd.notna(row.iloc[0]["Actual Finish"]):
            st.markdown(f"**End Date:** {row.iloc[0]['Actual Finish']}")

        if not row.empty and not is_editable:
            st.info("Submitted data is frozen. Contact admin to unlock.")
            continue

        # Always show input fields if unlocked, even if dates already exist
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
                new_row = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "PO No": selected_invoice,
                    "Customer": selected_customer,
                    "Process": process,
                    "Actual Start": start,
                    "Actual Finish": finish,
                    "Remarks": remarks,
                    "Submitted By": submitted_by
                }
                if row.empty:
                    status_df = pd.concat([status_df, pd.DataFrame([new_row])], ignore_index=True)
                else:
                    status_df.loc[row.index[0]] = new_row
                status_df.to_csv(STATUS_FILE, index=False)
                log_path = os.path.join(DAILY_LOG_DIR, f"log_{datetime.today().strftime('%Y-%m-%d')}.csv")
                pd.DataFrame([new_row]).to_csv(log_path, mode='a', index=False, header=not os.path.exists(log_path))
                st.success(f"{process} updated for {selected_invoice}")

# ---------------- PAGE 3: VISUALIZE -----------------
if page == "Visualize":
    st.title("Process Completion Matrix with Status Comparison")

    if not os.path.exists(FREEZE_FILE):
        st.warning("Please upload data in Page 1.")
    else:
        df = pd.read_csv(FREEZE_FILE)
        actuals = pd.read_csv(STATUS_FILE) if os.path.exists(STATUS_FILE) else pd.DataFrame()

        matrix = []

        for _, row in df.iterrows():
            status_row = {"PO No": row["PO No"], "Customer": row["Customer"]}

            for process in process_options:
                actual_row = actuals[(actuals["PO No"] == row["PO No"]) & (actuals["Process"] == process)]
                planned_end = row.get(f"{process} Finish")
                actual_end = actual_row["Actual Finish"].values[0] if not actual_row.empty else None

                if pd.notna(actual_end) and pd.notna(planned_end):
                    delay_days = (pd.to_datetime(actual_end) - pd.to_datetime(planned_end)).days
                    if delay_days > 0:
                        status = f"🟥 Delayed ({delay_days}d)"
                    elif delay_days < 0:
                        status = f"🟩 Early ({abs(delay_days)}d)"
                    else:
                        status = "🟨 On Time"
                elif not actual_row.empty:
                    status = "🔵 In Progress"
                else:
                    status = "⬜ Not Started"

                status_row[process] = status

            matrix.append(status_row)

        status_df = pd.DataFrame(matrix)
        st.dataframe(status_df, use_container_width=True)

        # Summary View
        st.subheader("Summary Table")
        po_selected = st.selectbox("Select PO No for Summary", status_df["PO No"].unique())
        summary = []
        for process in process_options:
            plan_end = df[df["PO No"] == po_selected][f"{process} Finish"].values[0] if f"{process} Finish" in df.columns else ""
            actual_row = actuals[(actuals["PO No"] == po_selected) & (actuals["Process"] == process)]
            actual_end = actual_row["Actual Finish"].values[0] if not actual_row.empty else ""
            submitted_by = actual_row["Submitted By"].values[0] if not actual_row.empty else ""
            remarks = actual_row["Remarks"].values[0] if not actual_row.empty else ""
            difference = ""
            if pd.notna(plan_end) and actual_end:
                difference = (pd.to_datetime(actual_end) - pd.to_datetime(plan_end)).days
            summary.append({
                "Process": process,
                "Planned Finish": plan_end,
                "Actual Finish": actual_end,
                "Delay (Days)": difference,
                "Remarks": remarks,
                "Submitted By": submitted_by
            })
        st.dataframe(pd.DataFrame(summary))
# ---------------- PAGE: PIE CHART VIEW -----------------
if page == "Pie chart view":
    import io
    from plotly.io import to_image
    from PIL import Image
    from fpdf import FPDF

    st.title("📊 Pie Chart View - Planned vs Actual (by Days)")

    if not os.path.exists(FREEZE_FILE):
        st.warning("Please upload a target plan first.")
        st.stop()

    plan_df = pd.read_csv(FREEZE_FILE)
    actual_df = pd.read_csv(STATUS_FILE) if os.path.exists(STATUS_FILE) else pd.DataFrame()

    customer_list = plan_df["Customer"].dropna().unique().tolist()
    selected_customer = st.selectbox("Select Customer", customer_list)

    customer_orders = plan_df[plan_df["Customer"] == selected_customer]

    all_figs = []
    for idx, order in customer_orders.iterrows():
        po_no = order["PO No"]
        if idx > 0:
            st.markdown("---")
        st.markdown(f"### PO No: {po_no}")

        planned_durations = []
        actual_durations = []

        for process in process_options:
            plan_start = order.get(f"{process} Start")
            plan_end = order.get(f"{process} Finish")
            if pd.notna(plan_start) and pd.notna(plan_end):
                days = (pd.to_datetime(plan_end) - pd.to_datetime(plan_start)).days
                planned_durations.append({"Process": process, "Days": max(days, 0)})

            actual_rows = actual_df[(actual_df["PO No"] == po_no) & (actual_df["Process"] == process)]
            if not actual_rows.empty:
                actual_start = actual_rows.iloc[0]["Actual Start"]
                actual_finish = actual_rows.iloc[0]["Actual Finish"]
                if pd.notna(actual_start) and pd.notna(actual_finish):
                    adays = (pd.to_datetime(actual_finish) - pd.to_datetime(actual_start)).days
                    actual_durations.append({"Process": process, "Days": max(adays, 0)})

        col1, col2 = st.columns(2)

        process_colors = px.colors.qualitative.Plotly[:len(process_options)]
        color_map = {proc: process_colors[i % len(process_colors)] for i, proc in enumerate(process_options)}

        fig1, fig2 = None, None
        if planned_durations:
            with col1:
                pdf = pd.DataFrame(planned_durations)
                fig1 = px.pie(pdf, names="Process", values="Days", title="Planned Days",
                              hover_data=['Days'], labels={'Days': 'Days'}, hole=0.3,
                              color="Process", color_discrete_map=color_map)
                fig1.update_traces(textinfo='value', hovertemplate='%{label}: %{percent}')
                st.plotly_chart(fig1, use_container_width=True, key=f"{po_no}_planned")
                st.markdown(f"**Total Planned Days:** {sum(pdf['Days'])} days")

        if actual_durations:
            with col2:
                adf = pd.DataFrame(actual_durations)
                fig2 = px.pie(adf, names="Process", values="Days", title="Actual Days",
                              hover_data=['Days'], labels={'Days': 'Days'}, hole=0.3,
                              color="Process", color_discrete_map=color_map)
                fig2.update_traces(textinfo='value', hovertemplate='%{label}: %{percent}')
                st.plotly_chart(fig2, use_container_width=True, key=f"{po_no}_actual")
                st.markdown(f"**Total Actual Days:** {sum(adf['Days'])} days")

        if fig1 and fig2:
            all_figs.append((po_no, fig1, fig2))

    if all_figs and st.button("📥 Download Combined PDF Report"):
        pdf_doc = FPDF()
        img_paths = []

        for po_no, fig1, fig2 in all_figs:
            img1 = Image.open(io.BytesIO(to_image(fig1, format="png")))
            img2 = Image.open(io.BytesIO(to_image(fig2, format="png")))
            path1 = f"{po_no}_planned.png"
            path2 = f"{po_no}_actual.png"
            img1.save(path1)
            img2.save(path2)
            img_paths.extend([path1, path2])

            for path in [path1, path2]:
                pdf_doc.add_page()
                pdf_doc.image(path, x=10, y=20, w=180)

        pdf_path = f"{selected_customer}_Combined_Report.pdf"
        pdf_doc.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", data=f, file_name=pdf_path, mime="application/pdf")

        for path in img_paths:
            os.remove(path)
        os.remove(pdf_path)
# ---------------- PAGE 4: LOGS -----------------
if page == "Logs":
    st.title("Daily Logs and Export")
    if os.path.exists(STATUS_FILE):
        log_df = pd.read_csv(STATUS_FILE)
        st.dataframe(log_df)
        st.download_button("Download Log as CSV", log_df.to_csv(index=False), file_name="log_export.csv")
    else:
        st.warning("No logs available.")

# ---------------- PAGE 5: UNLOCK -----------------
if page == "Unlock POs" and role == "admin":
    st.title("Unlock PO Data for Editing")
    locked_pos = [po for po in pd.read_csv(FREEZE_FILE)["PO No"].unique() if po not in st.session_state.enable_edits]
    if locked_pos:
        selected_unlock_po = st.selectbox("Select PO to Unlock", locked_pos)
        if st.button("Unlock Selected PO"):
            st.session_state.enable_edits[selected_unlock_po] = True
            st.success(f"PO {selected_unlock_po} has been unlocked.")
    else:
        st.info("No locked POs to unlock.")

if page == "Unlock Process" and role == "admin":
    st.title("Unlock Individual Process")
    po_list = pd.read_csv(FREEZE_FILE)["PO No"].unique()
    selected_po = st.selectbox("Select PO", po_list)
    selected_process = st.selectbox("Select Process", process_options)
    if st.button("Unlock Process"):
        st.session_state.unlocked_process[(selected_po, selected_process)] = True
        st.success(f"Process '{selected_process}' for PO '{selected_po}' has been unlocked.")
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Memory usage in MB

st.sidebar.title("System Monitor")
st.sidebar.write(f"Memory Usage: {get_memory_usage():.2f} MB")
