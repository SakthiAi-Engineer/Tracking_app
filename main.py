import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Google Sheets Setup
gsheets_url = st.secrets["gsheets_url"]
conn = st.connection("gsheets")

def load_data(sheet_name):
    data = conn.read(worksheet=sheet_name)
    return data.dropna(how="all")

def save_data(df, sheet_name):
    conn.update(worksheet=sheet_name, data=df)

# Load Data
frozen_df = load_data("freeze")
status_df = load_data("status")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Update Plan", "Daily Status", "Visualize", "Logs", "PO Unlock", "Process Unlock", "Summary Pie Chart"])

# Page 1: Update Plan
if page == "Update Plan":
    st.title("🔄 Update Planning Freeze")

    with st.form("freeze_form"):
        po_number = st.text_input("PO Number")
        customer = st.text_input("Customer Name")
        process = st.selectbox("Process", [
            "Warping", "Sizing", "Weaving", "Greige Inspection", "Processing",
            "Inspection", "Stitching", "Final Inspection", "Packing & Cartooning", "Shipment"])
        planned_date = st.date_input("Planned Date")
        submit = st.form_submit_button("Add/Update Freeze Plan")

        if submit:
            new_row = pd.DataFrame({
                "PO Number": [po_number],
                "Customer Name": [customer],
                "Process": [process],
                "Planned Date": [planned_date.strftime("%Y-%m-%d")]
            })
            frozen_df = pd.concat([frozen_df, new_row], ignore_index=True)
            save_data(frozen_df, "freeze")
            st.success("Planning data updated successfully.")

    st.write("### Current Planning Freeze")
    st.dataframe(frozen_df)

# Page 2: Daily Status
elif page == "Daily Status":
    st.title("🗓️ Daily Process Update")

    with st.form("status_form"):
        date = st.date_input("Date", value=datetime.today())
        po_number = st.text_input("PO Number")
        customer = st.text_input("Customer Name")
        process = st.selectbox("Process", [
            "Warping", "Sizing", "Weaving", "Greige Inspection", "Processing",
            "Inspection", "Stitching", "Final Inspection", "Packing & Cartooning", "Shipment"])
        status = st.selectbox("Status", ["Not Started", "On Progress", "Completed"])
        actual_date = st.date_input("Actual Date")
        remarks = st.text_input("Remarks")
        submit = st.form_submit_button("Submit Status")

        if submit:
            new_status = pd.DataFrame({
                "Date": [date.strftime("%Y-%m-%d")],
                "PO Number": [po_number],
                "Customer Name": [customer],
                "Process": [process],
                "Status": [status],
                "Planned Date": [""],
                "Actual Date": [actual_date.strftime("%Y-%m-%d")],
                "Remarks": [remarks]
            })
            status_df = pd.concat([status_df, new_status], ignore_index=True)
            save_data(status_df, "status")
            st.success("Status updated successfully.")

    st.write("### Daily Status Records")
    st.dataframe(status_df)

# Page 3: Visualize
elif page == "Visualize":
    st.title("📊 Process Tracking Dashboard")

    merged_df = pd.merge(status_df, frozen_df, on=["PO Number", "Customer Name", "Process"], how="outer", suffixes=("_Status", "_Planned"))
    merged_df.fillna("", inplace=True)

    fig = px.timeline(
        merged_df,
        x_start="Planned Date",
        x_end="Actual Date",
        y="PO Number",
        color="Status",
        hover_data=["Process", "Customer Name"],
        color_discrete_map={
            "Not Started": "gray",
            "On Progress": "blue",
            "Completed": "green"
        }
    )
    fig.update_yaxes(categoryorder="category ascending")
    st.plotly_chart(fig, use_container_width=True)

# Page 4: Logs
elif page == "Logs":
    st.title("📝 Logs")
    st.write("### Status Log")
    st.dataframe(status_df)
    st.write("### Freeze Log")
    st.dataframe(frozen_df)

# Page 5: PO Unlock
elif page == "PO Unlock":
    st.title("🔓 Unlock PO")
    unlock_po = st.selectbox("Select PO to Unlock", frozen_df["PO Number"].unique())
    if st.button("Unlock Selected PO"):
        frozen_df = frozen_df[frozen_df["PO Number"] != unlock_po]
        save_data(frozen_df, "freeze")
        st.success(f"PO {unlock_po} unlocked.")

# Page 6: Process Unlock
elif page == "Process Unlock":
    st.title("🔓 Unlock Process Entry")
    unlock_entry = st.selectbox("Select entry to remove", status_df.apply(lambda x: f'{x["PO Number"]} - {x["Process"]} - {x["Date"]}', axis=1))
    if st.button("Remove Selected Entry"):
        idx_to_remove = status_df.apply(lambda x: f'{x["PO Number"]} - {x["Process"]} - {x["Date"]}', axis=1) == unlock_entry
        status_df = status_df[~idx_to_remove]
        save_data(status_df, "status")
        st.success("Entry removed.")

# Page 7: Summary Pie Chart
elif page == "Summary Pie Chart":
    st.title("📈 Status Distribution Pie Chart")
    status_counts = status_df["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    fig = px.pie(status_counts, names="Status", values="Count", title="Overall Status Distribution",
                 color_discrete_map={
                     "Not Started": "gray",
                     "On Progress": "blue",
                     "Completed": "green"
                 })
    st.plotly_chart(fig, use_container_width=True)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Memory usage in MB

st.sidebar.title("System Monitor")
st.sidebar.write(f"Memory Usage: {get_memory_usage():.2f} MB")
