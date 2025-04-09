
import streamlit as st
import requests
import json
import urllib.parse
import urllib3
import certifi
import pandas as pd  
from bs4 import BeautifulSoup
from datetime import datetime
import re
import logging
import os
from dotenv import load_dotenv
import io

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# WatsonX configuration
WATSONX_API_URL = os.getenv("WATSONX_API_URL")
MODEL_ID = os.getenv("MODEL_ID")
PROJECT_ID = os.getenv("PROJECT_ID")
API_KEY = os.getenv("API_KEY")

# Check environment variables
if not all([API_KEY, WATSONX_API_URL, MODEL_ID, PROJECT_ID]):
    st.error("❌ Required environment variables (API_KEY, WATSONX_API_URL, MODEL_ID, PROJECT_ID) missing!")
    logger.error("Missing one or more required environment variables")
    st.stop()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API Endpoints
LOGIN_URL = "https://dms.asite.com/apilogin/"
SEARCH_URL = "https://adoddleak.asite.com/commonapi/formsearchapi/search"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Function to generate access token
def get_access_token(API_KEY):
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": API_KEY}
    try:
        response = requests.post(IAM_TOKEN_URL, headers=headers, data=data, verify=certifi.where(), timeout=50)
        if response.status_code == 200:
            token_info = response.json()
            logger.info("Access token generated successfully")
            return token_info['access_token']
        else:
            logger.error(f"Failed to get access token: {response.status_code} - {response.text}")
            st.error(f"❌ Failed to get access token: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Exception getting access token: {str(e)}")
        st.error(f"❌ Error getting access token: {str(e)}")
        return None

# Login Function
def login_to_asite(email, password):
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    payload = {"emailId": email, "password": password}
    response = requests.post(LOGIN_URL, headers=headers, data=payload, verify=certifi.where(), timeout=50)
    if response.status_code == 200:
        try:
            session_id = response.json().get("UserProfile", {}).get("Sessionid")
            logger.info(f"Login successful, Session ID: {session_id}")
            return session_id
        except json.JSONDecodeError:
            logger.error("JSONDecodeError during login")
            st.error("❌ Failed to parse login response")
            return None
    logger.error(f"Login failed: {response.status_code}")
    st.error(f"❌ Login failed: {response.status_code}")
    return None

# Fetch Data Function
def fetch_project_data(session_id, project_name, form_name, record_limit=1000):
    headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded", "Cookie": f"ASessionID={session_id}"}
    all_data = []
    start_record = 1
    total_records = None

    with st.spinner("Fetching data from Asite..."):
        while True:
            search_criteria = {"criteria": [{"field": "ProjectName", "operator": 1, "values": [project_name]}, {"field": "FormName", "operator": 1, "values": [form_name]}], "recordStart": start_record, "recordLimit": record_limit}
            search_criteria_str = json.dumps(search_criteria)
            encoded_payload = f"searchCriteria={urllib.parse.quote(search_criteria_str)}"
            response = requests.post(SEARCH_URL, headers=headers, data=encoded_payload, verify=certifi.where(), timeout=50)

            try:
                response_json = response.json()
                if total_records is None:
                    total_records = response_json.get("responseHeader", {}).get("results-total", 0)
                all_data.extend(response_json.get("FormList", {}).get("Form", []))
                st.info(f"🔄 Fetched {len(all_data)} / {total_records} records")
                if start_record + record_limit - 1 >= total_records:
                    break
                start_record += record_limit
            except Exception as e:
                logger.error(f"Error fetching data: {str(e)}")
                st.error(f"❌ Error fetching data: {str(e)}")
                break

    return {"responseHeader": {"results": len(all_data), "total_results": total_records}}, all_data, encoded_payload

# Process JSON Data
def process_json_data(json_data):
    data = []
    for item in json_data:
        form_details = item.get('FormDetails', {})
        created_date = form_details.get('FormCreationDate', None)
        expected_close_date = form_details.get('UpdateDate', None)
        form_status = form_details.get('FormStatus', None)
        
        discipline = None
        description = None
        custom_fields = form_details.get('CustomFields', {}).get('CustomField', [])
        for field in custom_fields:
            if field.get('FieldName') == 'CFID_DD_DISC':
                discipline = field.get('FieldValue', None)
            elif field.get('FieldName') == 'CFID_RTA_DES':
                description = BeautifulSoup(field.get('FieldValue', None) or '', "html.parser").get_text()

        days_diff = None
        if created_date and expected_close_date:
            try:
                created_date_obj = datetime.strptime(created_date.split('#')[0], "%d-%b-%Y")
                expected_close_date_obj = datetime.strptime(expected_close_date.split('#')[0], "%d-%b-%Y")
                days_diff = (expected_close_date_obj - created_date_obj).days
            except Exception as e:
                logger.error(f"Error calculating days difference: {str(e)}")
                days_diff = None

        data.append([days_diff, created_date, expected_close_date, description, form_status, discipline])

    df = pd.DataFrame(data, columns=['Days', 'Created Date (WET)', 'Expected Close Date (WET)', 'Description', 'Status', 'Discipline'])
    df['Created Date (WET)'] = pd.to_datetime(df['Created Date (WET)'].str.split('#').str[0], format="%d-%b-%Y", errors='coerce')
    df['Expected Close Date (WET)'] = pd.to_datetime(df['Expected Close Date (WET)'].str.split('#').str[0], format="%d-%b-%Y", errors='coerce')
    logger.debug(f"DataFrame columns after processing: {df.columns.tolist()}")  # Debug column names
    if df.empty:
        logger.warning("DataFrame is empty after processing")
        st.warning("⚠️ No data processed. Check the API response.")
    return df

# Generate NCR Report
def generate_ncr_report(df, report_type, start_date=None, end_date=None):
    with st.spinner(f"Generating {report_type} NCR Report..."):
        # Filter based on Created Date (WET) range and pre-calculated Days > 21
        if report_type == "Closed":
            filtered_df = df[
                (df['Status'] == 'Closed') &
                (df['Expected Close Date (WET)'] >= pd.to_datetime(start_date)) &
                (df['Created Date (WET)'] <= pd.to_datetime(end_date)) &
                (df['Days'] > 21)
            ].copy()  # Use .copy() to avoid SettingWithCopyWarning
        else:  # Open
            from datetime import datetime
            today = pd.to_datetime(datetime.today().strftime('%Y/%m/01'))
            print(today,"hari")  # Convert to Pandas Timestamp
            filtered_df = df[
                (df['Status'] == 'Open') &
                (df['Created Date (WET)'].notna())  # Ensure Created Date exists
            ].copy()
            # Calculate days from Created Date to today
            filtered_df.loc[:, 'Days_From_Today'] = (today - pd.to_datetime(filtered_df['Created Date (WET)'])).dt.days
            filtered_df = filtered_df[filtered_df['Days_From_Today'] > 21].copy()

        if filtered_df.empty:
            return {"error": f"No {report_type} records found with duration > 21 days"}, ""

        # Use .loc to avoid SettingWithCopyWarning
        filtered_df.loc[:, 'Created Date (WET)'] = filtered_df['Created Date (WET)'].astype(str)
        filtered_df.loc[:, 'Expected Close Date (WET)'] = filtered_df['Expected Close Date (WET)'].astype(str)

        processed_data = filtered_df.to_dict(orient="records")
        
        # Inside generate_ncr_report, after the cleaned_data loop
        cleaned_data = []
        for record in processed_data:
            cleaned_record = {
                "Description": str(record.get("Description", "")),
                "Discipline": str(record.get("Discipline", "")),
                "Created Date (WET)": str(record.get("Created Date (WET)", "")),
                "Expected Close Date (WET)": str(record.get("Expected Close Date (WET)", "")),
                "Status": str(record.get("Status", "")),
                "Days": record.get("Days", 0),
                "Tower": "External Development"
            }

            description = cleaned_record["Description"]
            if any(phrase in description.lower() for phrase in ["veridia clubhouse", "veridia-clubhouse", "veridia club"]):
                cleaned_record["Tower"] = "Veridia-Club"
                logger.debug(f"Matched 'Veridia Clubhouse' variant in description: {description}, setting Tower to Veridia-Club")
            else:
                tower_match = re.search(r"(Tower|T)\s*-?\s*(\d+)", description, re.IGNORECASE)
                cleaned_record["Tower"] = f"Veridia- Tower-{tower_match.group(2).zfill(2)}" if tower_match else None
                logger.debug(f"Tower match result: {tower_match}, Tower set to {cleaned_record['Tower']}")

            discipline = cleaned_record["Discipline"].strip().lower()
            if "structure" in discipline or "sw" in discipline:
                cleaned_record["Discipline_Category"] = "SW"
            elif "civil" in discipline or "finishing" in discipline or "fw" in discipline:
                cleaned_record["Discipline_Category"] = "FW"
            else:
                cleaned_record["Discipline_Category"] = "MEP"

            cleaned_data.append(cleaned_record)
            logger.debug(f"Processed record: {json.dumps(cleaned_record, indent=2)}")

        # Filter out records with no Tower
        cleaned_data = [record for record in cleaned_data if record["Tower"] is not None]

        st.write(f"Debug - Total {report_type} records to process: {len(cleaned_data)}")
        logger.debug(f"Processed data for {report_type}: {json.dumps(cleaned_data, indent=2)}")

        if not cleaned_data:
            return {report_type: {"Sites": {}, "Grand_Total": 0}}, ""

        access_token = get_access_token(API_KEY)
        if not access_token:
            return {"error": "Failed to obtain access token"}, ""

        result = {report_type: {"Sites": {}, "Grand_Total": 0}}
        
        for record in cleaned_data:
            tower = record["Tower"]
            discipline = record["Discipline_Category"]
            
            if tower not in result[report_type]["Sites"]:
                result[report_type]["Sites"][tower] = {"SW": 0, "FW": 0, "MEP": 0, "Total": 0}
            
            result[report_type]["Sites"][tower][discipline] += 1
            result[report_type]["Sites"][tower]["Total"] += 1
            result[report_type]["Grand_Total"] += 1
        
        chunk_size = 200
        all_results = {report_type: {"Sites": {}, "Grand_Total": 0}}

        for i in range(0, len(cleaned_data), chunk_size):
            chunk = cleaned_data[i:i + chunk_size]
            st.write(f"Processing chunk {i // chunk_size + 1}: Records {i} to {min(i + chunk_size, len(cleaned_data))}")
            logger.debug(f"Chunk data sent to WatsonX: {json.dumps(chunk, indent=2)}")

            prompt = (
                "IMPORTANT: YOU MUST RETURN ONLY A SINGLE VALID JSON OBJECT. DO NOT INCLUDE ANY TEXT, EXPLANATION, OR MULTIPLE RESPONSES.\n\n"
                f"Task: For {report_type} NCRs, count ALL records in the provided data by site ('Tower' field) and discipline category ('Discipline_Category' field) EXACTLY AS PROVIDED IN THE 'Tower' AND 'Discipline_Category' FIELDS OF THE DATA. DO NOT SKIP ANY RECORDS, REINTERPRET, OR MODIFY THE 'Tower' OR 'Discipline_Category' VALUES BASED ON THE DESCRIPTION. ENSURE EVERY RECORD WITH 'Days' > 21 AND THE CORRECT 'Status' IS COUNTED.\n"
                f"Condition: Only include records where 'Days' > 21 (pre-calculated planned duration from Created Date to Expected Close Date for Closed, or from Created Date to today for Open) and "
                f"{'Created Date (WET) is between Closed Start Date and Closed End Date' if report_type == 'Closed' else 'Created Date (WET) is before today'}.\n"
                "Use the 'Tower' values exactly as they appear in the data (e.g., 'Veridia-Club', 'Tower-06', 'Tower-01') and the 'Discipline_Category' values (e.g., 'SW', 'FW', 'MEP') without any changes.\n\n"
                "REQUIRED OUTPUT FORMAT (exactly this structure with actual counts):\n"
                "{\n"
                f'  "{report_type}": {{\n'
                '    "Sites": {\n'
                '      "Site_Name1": {\n'
                '        "SW": number,\n'
                '        "FW": number,\n'
                '        "MEP": number,\n'
                '        "Total": number\n'
                '      },\n'
                '      "Site_Name2": {\n'
                '        "SW": number,\n'
                '        "FW": number,\n'
                '        "MEP": number,\n'
                '        "Total": number\n'
                '      }\n'
                '    },\n'
                '    "Grand_Total": number\n'
                '  }\n'
                '}\n\n'
                f"Data: {json.dumps(chunk)}\n"
                f"Return the result strictly as a single JSON object—no code, no explanations, no string literal like this ```, only the JSON."
            )

            payload = {"input": prompt, "parameters": {"decoding_method": "greedy", "max_new_tokens": 8100, "min_new_tokens": 0, "temperature": 0.01}, "model_id": MODEL_ID, "project_id": PROJECT_ID}
            headers = {"Accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}

            try:
                response = requests.post(WATSONX_API_URL, headers=headers, json=payload, verify=certifi.where(), timeout=600)
                logger.info(f"WatsonX API response status code: {response.status_code}")
                logger.debug(f"WatsonX API response text: {response.text}")
                st.write(f"Debug - Response status code: {response.status_code}")

                if response.status_code == 200:
                    api_result = response.json()
                    generated_text = api_result.get("results", [{}])[0].get("generated_text", "").strip()
                    st.write(f"Debug - Raw response: {generated_text}")
                    logger.debug(f"Parsed generated text: {generated_text}")

                    if generated_text:
                        json_match = re.search(r'({[\s\S]*?})(?=\n\{|\Z)', generated_text)
                        if json_match:
                            json_str = json_match.group(1)
                            try:
                                parsed_json = json.loads(json_str)
                                chunk_result = parsed_json.get(report_type, {})
                                chunk_sites = chunk_result.get("Sites", {})
                                chunk_grand_total = chunk_result.get("Grand_Total", 0)
                                logger.debug(f"Parsed JSON result: {json.dumps(parsed_json, indent=2)}")
                                logger.info(f"WatsonX results for comparison: {json.dumps(parsed_json, indent=2)}")
                            except json.JSONDecodeError as e:
                                logger.error(f"JSONDecodeError: {str(e)} - Raw response: {generated_text}")
                                st.error(f"❌ Failed to parse JSON: {str(e)}")
                                st.write("Falling back to local count due to JSON parsing error")
                        else:
                            logger.error("No valid JSON found in response")
                            st.error("❌ No valid JSON found in response")
                            st.write("Falling back to local count due to invalid JSON response")
                    else:
                        logger.error("Empty generated_text from WatsonX")
                        st.error("❌ Empty response from WatsonX")
                        st.write("Falling back to local count due to empty WatsonX response")
                else:
                    error_msg = f"❌ WatsonX API error: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    st.write("Falling back to local count due to WatsonX API error")
            except Exception as e:
                error_msg = f"❌ Exception during WatsonX call: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg)
                st.write("Falling back to local count due to exception")

        return result, json.dumps(result)

def generate_consolidated_ncr_excel(combined_result, report_title="NCR"):
    # Create a new Excel writer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        title_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': 'yellow',
            'border': 1,
            'font_size': 12
        })
        
        header_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'text_wrap': True
        })
        
        subheader_format = workbook.add_format({
            'bold': True,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        site_format = workbook.add_format({
            'align': 'left',
            'valign': 'vcenter',
            'border': 1
        })
        
        # Create worksheet
        worksheet = workbook.add_worksheet('NCR Report')
        
        # Set column widths
        worksheet.set_column('A:A', 20)  # Site column
        worksheet.set_column('B:H', 12)  # Data columns
        
        # Get data from both sections
        resolved_data = combined_result.get("NCR resolved beyond 21 days", {})
        open_data = combined_result.get("NCR open beyond 21 days", {})
        
        if not isinstance(resolved_data, dict) or "error" in resolved_data:
            resolved_data = {"Sites": {}}
        if not isinstance(open_data, dict) or "error" in open_data:
            open_data = {"Sites": {}}
            
        resolved_sites = resolved_data.get("Sites", {})
        open_sites = open_data.get("Sites", {})
        
        # Define only the standard sites you want to include
        standard_sites = [
            "Veridia-Club",
            "Veridia- Tower 01",
            "Veridia- Tower 02",
            "Veridia- Tower 03",
            "Veridia- Tower 04",
            "Veridia- Tower 05",
            "Veridia- Tower 06",
            "Veridia- Tower 07",
            "Veridia-Commercial",
            "External Development"
        ]
        
        # Normalize JSON site names to match standard_sites format
        def normalize_site_name(site):
            if site in ["Veridia-Club", "Veridia-Commercial"]:
                return site
            match = re.search(r'(?:tower|t)[- ]?(\d+)', site, re.IGNORECASE)
            if match:
                num = match.group(1).zfill(2)
                return f"Veridia- Tower {num}"
            return site

        # Create a reverse mapping for original keys to normalized names
        site_mapping = {k: normalize_site_name(k) for k in (resolved_sites.keys() | open_sites.keys())}
        
        # Sort the standard sites
        sorted_sites = sorted(standard_sites)
        
        # Title row
        worksheet.merge_range('A1:H1', report_title, title_format)
        
        # Header row
        row = 1
        worksheet.write(row, 0, 'Site', header_format)
        worksheet.merge_range(row, 1, row, 3, 'NCR resolved beyond 21 days', header_format)
        worksheet.merge_range(row, 4, row, 6, 'NCR open beyond 21 days', header_format)
        worksheet.write(row, 7, 'Total', header_format)
        
        # Subheaders
        row = 2
        categories = ['Civil Finishing', 'MEP', 'Structure']
        worksheet.write(row, 0, '', header_format)
        
        # Resolved subheaders
        for i, cat in enumerate(categories):
            worksheet.write(row, i+1, cat, subheader_format)
            
        # Open subheaders
        for i, cat in enumerate(categories):
            worksheet.write(row, i+4, cat, subheader_format)
            
        worksheet.write(row, 7, '', header_format)
        
        # Map our categories to the JSON data categories
        category_map = {
            'Civil Finishing': 'FW',
            'MEP': 'MEP',
            'Structure': 'SW'
        }
        
        # Data rows
        row = 3
        site_totals = {}
        
        for site in sorted_sites:
            worksheet.write(row, 0, site, site_format)
            
            # Find original key that maps to this normalized site
            original_resolved_key = next((k for k, v in site_mapping.items() if v == site), None)
            original_open_key = next((k for k, v in site_mapping.items() if v == site), None)
            
            site_total = 0
            
            # Resolved data
            for i, (display_cat, json_cat) in enumerate(category_map.items()):
                value = 0
                if original_resolved_key and original_resolved_key in resolved_sites:
                    value = resolved_sites[original_resolved_key].get(json_cat, 0)
                worksheet.write(row, i+1, value, cell_format)
                site_total += value
                
            # Open data
            for i, (display_cat, json_cat) in enumerate(category_map.items()):
                value = 0
                if original_open_key and original_open_key in open_sites:
                    value = open_sites[original_open_key].get(json_cat, 0)
                worksheet.write(row, i+4, value, cell_format)
                site_total += value
                
            # Total for this site
            worksheet.write(row, 7, site_total, cell_format)
            site_totals[site] = site_total
            row += 1
        
        # Return the Excel file
        output.seek(0)
        return output

# Streamlit UI
st.title("Asite NCR Reporter")

# Initialize session state
if "df" not in st.session_state:
    st.session_state["df"] = None

# Login Section
st.sidebar.title("🔒 Asite Login")
email = st.sidebar.text_input("Email", "impwatson@gadieltechnologies.com")
password = st.sidebar.text_input("Password", "Srihari@790$", type="password")
if st.sidebar.button("Login"):
    session_id = login_to_asite(email, password)
    if session_id:
        st.session_state["session_id"] = session_id
        st.sidebar.success("✅ Login Successful")

# Data Fetch Section
st.sidebar.title("📂 Project Data")
project_name = st.sidebar.text_input("Project Name", "Wave Oakwood, Wave City")
form_name = st.sidebar.text_input("Form Name", "Non Conformity Report")
if "session_id" in st.session_state and st.sidebar.button("Fetch Data"):
    header, data, payload = fetch_project_data(st.session_state["session_id"], project_name, form_name)
    st.json(header)
    if data:
        df = process_json_data(data)
        st.session_state["df"] = df  # Store DataFrame in session state
        st.dataframe(df)
        st.success("✅ Data fetched and processed successfully!")

# Report Generation Section
if st.session_state["df"] is not None:
    df = st.session_state["df"]
    
    st.sidebar.title("📋 Combined NCR Report")
    closed_start = st.sidebar.date_input("Closed Start Date", df['Expected Close Date (WET)'].min())
    closed_end = st.sidebar.date_input("Closed End Date", df['Expected Close Date (WET)'].max(), key="closed_end")
    open_end = st.sidebar.date_input("Open Until Date", df['Expected Close Date (WET)'].max(), key="open_end")
    
    if st.sidebar.button("Generate Combined NCR Report", key="generate_report_button"):
        month_name = closed_end.strftime("%B")
        report_title = f"NCR: {month_name}"
        
        closed_result, closed_raw = generate_ncr_report(df, "Closed", closed_start, closed_end)
        open_result, open_raw = generate_ncr_report(df, "Open", end_date=open_end)

        combined_result = {}
        if "error" not in closed_result:
            combined_result["NCR resolved beyond 21 days"] = closed_result["Closed"]
        else:
            combined_result["NCR resolved beyond 21 days"] = {"error": closed_result["error"]}
        
        if "error" not in open_result:
            combined_result["NCR open beyond 21 days"] = open_result["Open"]
        else:
            combined_result["NCR open beyond 21 days"] = {"error": open_result["error"]}

        st.subheader("Combined NCR Report (JSON)")
        st.json(combined_result)
        
        excel_file = generate_consolidated_ncr_excel(combined_result, report_title)
        st.download_button(
            label="📥 Download Excel Report",
            data=excel_file,
            file_name=f"NCR_Report_{month_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


