import streamlit as st
import pandas as pd
import json
import io
import plotly.express as px
from backend import RegexClassifier

def main():
    st.set_page_config(page_title="PII Discovery", layout="wide")
    st.title("🛡️ Segmento Sense")

    if 'classifier' not in st.session_state:
        st.session_state.classifier = RegexClassifier()
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 0
    
    classifier = st.session_state.classifier

    # --- SIDEBAR: CATEGORIZED SOURCE SELECTION ---
    with st.sidebar:
        st.header("1. Source Selection")
        
        # UPDATE: Split Databases into two categories
        category = st.selectbox("Select Category", 
            [
                "Traditional Sources", 
                "Relational Databases", 
                "Non-Relational Databases", 
                "Cloud Storage"
            ]
        )
        
        source = None
        file_sub_type = None

        # 1. TRADITIONAL SOURCES
        if category == "Traditional Sources":
            file_sub_type = st.radio("Select File Format", ["PDF Document", "CSV Spreadsheet", "JSON Data"])
            source = "File Upload"
        
        # 2. RELATIONAL DATABASES (SQL)
        elif category == "Relational Databases":
            db_icons = {
                "PostgreSQL": "🐘 PostgreSQL", 
                "MySQL": "🐬 MySQL"
            }
            source = st.selectbox(
                "Database Type", 
                options=["PostgreSQL", "MySQL"], 
                format_func=lambda x: db_icons.get(x)
            )
        
        # 3. NON-RELATIONAL DATABASES (NoSQL)
        elif category == "Non-Relational Databases":
            db_icons = {
                "MongoDB": "🍃 MongoDB"
            }
            source = st.selectbox(
                "Database Type", 
                options=["MongoDB"], 
                format_func=lambda x: db_icons.get(x)
            )
        
        # 4. CLOUD STORAGE
        elif category == "Cloud Storage":
            source = st.selectbox("Service", ["Google Drive"])

        st.divider()
        st.header("2. Patterns")
        patterns = classifier.list_patterns()
        
        ordered_keys = ["EMAIL", "FIRST_NAME", "LAST_NAME", "PHONE", "SSN", "CREDIT_CARD"]
        display_patterns = {k: patterns.get(k, "NLTK") for k in ordered_keys if k in patterns or k in ["FIRST_NAME", "LAST_NAME"]}
        for k, v in patterns.items():
            if k not in display_patterns: display_patterns[k] = v     
        st.dataframe(pd.DataFrame(list(display_patterns.items()), columns=["Name", "Regex/Method"]), hide_index=True)
        
        with st.expander("➕ Add Pattern"):
            new_name = st.text_input("Name")
            new_regex = st.text_input("Regex")
            if st.button("Add"):
                classifier.add_pattern(new_name, new_regex)
                st.rerun()

        with st.expander("🗑️ Remove Pattern"):
            pattern_to_remove = st.selectbox("Select Pattern", options=list(patterns.keys()))
            if st.button("Remove Selected"):
                classifier.remove_pattern(pattern_to_remove)
                st.rerun()

    # --- HELPER: RENDER LOGOS ---
    def render_source_header(title, logo_url):
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            if logo_url: st.image(logo_url, width=50)
        with col2: st.subheader(title)
        st.divider()

    # --- HELPER: ANALYTICS DASHBOARD ---
    def render_analytics(count_df):
        st.markdown("### 📊 Analytics Dashboard")
        if count_df.empty:
            st.info("No PII data detected to visualize.")
            return

        c1, c2 = st.columns([1, 1])
        with c1:
            st.caption("Distribution of Detected PII")
            fig = px.pie(count_df, values='Count', names='PII Type', hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.caption("Detailed Counts")
            st.dataframe(count_df, hide_index=True, use_container_width=True)

    # --- MAIN LOGIC ---

    # A. TRADITIONAL SOURCES
    if source == "File Upload":
        ext_map = {"PDF Document": ["pdf"], "CSV Spreadsheet": ["csv"], "JSON Data": ["json"]}
        accepted_exts = ext_map[file_sub_type]
        
        st.subheader(f"📂 {file_sub_type} Analysis")
        uploaded_file = st.file_uploader(f"Upload {file_sub_type}", type=accepted_exts)
        
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            mask_mode = st.checkbox("🔒 Enable PII Masking")

            if file_type == 'pdf':
                file_bytes = uploaded_file.getvalue()
                current_text = classifier.get_pdf_page_text(file_bytes, st.session_state.page_number)
                count_df = classifier.get_pii_counts(current_text)
                render_analytics(count_df)
                
                total_pages = classifier.get_pdf_total_pages(file_bytes)
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1:
                    if st.button("⬅️ Prev") and st.session_state.page_number > 0: st.session_state.page_number -= 1
                with c3:
                    if st.button("Next ➡️") and st.session_state.page_number < total_pages - 1: st.session_state.page_number += 1
                
                st.markdown(f"**Viewing Page {st.session_state.page_number + 1} of {total_pages}**")
                if mask_mode: st.warning("Visual masking for PDF pending. Showing detection boxes.")
                img_data = classifier.get_labeled_pdf_image(file_bytes, st.session_state.page_number)
                if img_data: st.image(img_data, use_container_width=True)

            elif file_type in ['csv', 'json']:
                if file_type == 'csv': df = pd.read_csv(uploaded_file)
                else: df = classifier.get_json_data(uploaded_file)
                render_analytics(classifier.get_pii_counts_dataframe(df))
                
                if mask_mode: 
                    st.success("Data Masked: All PII replaced with ******")
                    st.dataframe(classifier.mask_dataframe(df).head(50))
                else: 
                    st.markdown(classifier.scan_dataframe_with_html(df.head(50)).to_html(escape=False), unsafe_allow_html=True)

    # B. RELATIONAL DATABASES (SQL)
    elif category == "Relational Databases":
        db_logos = {
            "PostgreSQL": "https://upload.wikimedia.org/wikipedia/commons/2/29/Postgresql_elephant.svg",
            "MySQL": "https://upload.wikimedia.org/wikipedia/commons/b/b2/Database-mysql.svg"
        }
        render_source_header(f"Connect to {source}", db_logos.get(source, ""))

        c1, c2 = st.columns(2)
        host = c1.text_input("Host", "localhost")
        port = c2.text_input("Port", "3306" if source == "MySQL" else "5432")
        user = c1.text_input("User", "root" if source == "MySQL" else "postgres")
        pw = c2.text_input("Password", type="password")
        db = c1.text_input("Database Name")
        table = c2.text_input("Table Name")

        if st.button("Connect & Scan"):
            try:
                if source == "MySQL": df = classifier.get_mysql_data(host, port, db, user, pw, table)
                else: df = classifier.get_postgres_data(host, port, db, user, pw, table)
                st.session_state.db_data = df
                st.rerun()
            except Exception as e: st.error(f"Connection Failed: {e}")

        if 'db_data' in st.session_state:
            df = st.session_state.db_data
            if df.empty:
                st.warning("Database connected but returned no records.")
            else:
                count_df = classifier.get_pii_counts_dataframe(df)
                render_analytics(count_df)
                mask_mode = st.checkbox("🔒 Mask PII Results")
                if mask_mode: 
                    st.success("Data Masked")
                    st.dataframe(classifier.mask_dataframe(df))
                else: 
                    st.markdown(classifier.scan_dataframe_with_html(df).to_html(escape=False), unsafe_allow_html=True)

    # C. NON-RELATIONAL DATABASES (NoSQL)
    elif category == "Non-Relational Databases":
        render_source_header("Connect to MongoDB", "https://upload.wikimedia.org/wikipedia/commons/9/93/MongoDB_Logo.svg")

        c1, c2 = st.columns(2)
        host = c1.text_input("Host", "localhost")
        port = c2.text_input("Port", "27017")
        user = c1.text_input("User (Optional)", "")
        pw = c2.text_input("Password (Optional)", type="password")
        db = c1.text_input("Database Name")
        collection = c2.text_input("Collection Name")

        if st.button("Connect & Scan"):
            try:
                df = classifier.get_mongodb_data(host, port, db, user, pw, collection)
                st.session_state.db_data = df
                st.rerun()
            except Exception as e: st.error(f"Connection Failed: {e}")

        if 'db_data' in st.session_state:
            df = st.session_state.db_data
            if df.empty:
                st.warning("Collection empty or connection failed.")
            else:
                count_df = classifier.get_pii_counts_dataframe(df)
                render_analytics(count_df)
                mask_mode = st.checkbox("🔒 Mask PII Results")
                if mask_mode: 
                    st.success("Data Masked")
                    st.dataframe(classifier.mask_dataframe(df))
                else: 
                    st.markdown(classifier.scan_dataframe_with_html(df).to_html(escape=False), unsafe_allow_html=True)

    # D. CLOUD STORAGE
    elif category == "Cloud Storage":
        render_source_header("Google Drive Import", "https://upload.wikimedia.org/wikipedia/commons/d/da/Google_Drive_logo.png")
        
        st.info("Upload your Service Account JSON to connect dynamically.")
        creds_file = st.file_uploader("Upload credentials.json", type=['json'])
        
        if creds_file:
            creds_dict = json.load(creds_file)
            st.session_state.creds_dict = creds_dict
            st.success("Credentials Loaded!")
            
            if st.button("📂 List Files from Drive"):
                st.session_state.drive_files = classifier.get_google_drive_files(creds_dict)
        
        if 'drive_files' in st.session_state:
            files = st.session_state.drive_files
            if not files: st.warning("No files found or Authentication failed.")
            else:
                file_map = {f['name']: f for f in files}
                selected_name = st.selectbox("Select File", list(file_map.keys()))
                
                if st.button("⬇️ Scan File"):
                    sel_file = file_map[selected_name]
                    content = classifier.download_drive_file(
                        sel_file['id'], 
                        sel_file.get('mimeType', ''), 
                        st.session_state.creds_dict
                    )
                    
                    if not content: st.error("Failed to read file.")
                    else:
                        st.success(f"Scanning {selected_name}...")
                        mask_mode = st.checkbox("🔒 Mask Results", value=False, key="drive_mask")
                        
                        mime = sel_file.get('mimeType', '').lower()
                        is_csv = "csv" in mime or "spreadsheet" in mime
                        is_pdf = "pdf" in mime or "document" in mime or "presentation" in mime
                        is_json = "json" in mime

                        if is_pdf:
                            text = classifier.get_pdf_page_text(content, 0)
                            render_analytics(classifier.get_pii_counts(text))
                            img = classifier.get_labeled_pdf_image(content, 0)
                            if img: st.image(img, caption="Page 1 Preview")
                        elif is_csv:
                            df = pd.read_csv(io.BytesIO(content))
                            render_analytics(classifier.get_pii_counts_dataframe(df))
                            if mask_mode: st.dataframe(classifier.mask_dataframe(df))
                            else: st.markdown(classifier.scan_dataframe_with_html(df).to_html(escape=False), unsafe_allow_html=True)
                        elif is_json:
                            df = classifier.get_json_data(io.BytesIO(content))
                            render_analytics(classifier.get_pii_counts_dataframe(df))
                            if mask_mode: st.dataframe(classifier.mask_dataframe(df))
                            else: st.markdown(classifier.scan_dataframe_with_html(df).to_html(escape=False), unsafe_allow_html=True)

if __name__ == "__main__":
    main()