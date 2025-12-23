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
    
    # Session state to track "Last Seen" accuracy
    if 'last_accuracy' not in st.session_state:
        st.session_state.last_accuracy = {"🛠️ Regex": 0, "🧠 NLTK": 0, "🤖 SpaCy": 0}

    classifier = st.session_state.classifier

    # --- SIDEBAR: NESTED NAVIGATION ---
    with st.sidebar:
        st.header("1. Source Selection")
        
        main_category = st.selectbox("Select System", 
            ["File System", "Databases", "Cloud Storage"]
        )
        
        source = None
        file_sub_type = None
        
        # --- A. FILE SYSTEM ---
        if main_category == "File System":
            struct_type = st.radio("Data Type", ["Structured Data", "Unstructured Data"])
            if struct_type == "Structured Data":
                file_sub_type = st.selectbox("File Format", ["CSV", "JSON", "Parquet"])
            else:
                file_sub_type = st.selectbox("File Format", ["PDF"])
            source = "File Upload"

        # --- B. DATABASES ---
        elif main_category == "Databases":
            db_type = st.radio("Database Type", ["Relational (SQL)", "Non-Relational (NoSQL)"])
            if db_type == "Relational (SQL)":
                db_icons = {"PostgreSQL": "🐘 PostgreSQL", "MySQL": "🐬 MySQL"}
                source = st.selectbox("Select Database", ["PostgreSQL", "MySQL"], format_func=lambda x: db_icons.get(x))
            else:
                db_icons = {"MongoDB": "🍃 MongoDB"}
                source = st.selectbox("Select Database", ["MongoDB"], format_func=lambda x: db_icons.get(x))

        # --- C. CLOUD STORAGE ---
        elif main_category == "Cloud Storage":
            source = st.selectbox("Service", ["Google Drive", "AWS S3"])

        st.divider()
        
        # --- PATTERN MANAGEMENT ---
        st.header("2. Patterns")
        patterns = classifier.list_patterns()
        
        ordered_keys = ["EMAIL", "FIRST_NAME", "LAST_NAME", "PHONE", "SSN", "CREDIT_CARD"]
        display_patterns = {k: patterns.get(k, "NLTK/SpaCy") for k in ordered_keys if k in patterns or k in ["FIRST_NAME", "LAST_NAME"]}
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

    # --- HELPER: INSPECTOR DASHBOARD ---
    def render_inspector(raw_text):
        if not raw_text: return
        st.divider()
        st.markdown("### 🕵️ Inspector: Behind the Scenes")
        with st.expander("Show Detailed Model Performance", expanded=True):
            results_df = classifier.run_full_inspection(raw_text)
            if results_df.empty:
                st.info("No PII detected by any model.")
                return
            display_df = results_df[["Model", "Detected PII", "Missed PII"]]
            st.table(display_df)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Model Accuracy Graph**")
                fig = px.bar(results_df, x="Accuracy", y="Model", orientation='h', color="Model", text_auto='.2%', range_x=[0,1])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("**Efficiency Gain**")
                for index, row in results_df.iterrows():
                    model = row['Model']
                    current_acc = row['Accuracy']
                    last_acc = st.session_state.last_accuracy.get(model, 0)
                    delta = current_acc - last_acc
                    st.metric(label=model, value=f"{current_acc:.1%}", delta=f"{delta:.1%}")
                    st.session_state.last_accuracy[model] = current_acc

    # --- HELPER: ANALYTICS DASHBOARD ---
    def render_analytics(count_df, source_df=None):
        if source_df is not None and not source_df.empty:
            st.markdown("### 🧬 Data Schema Detected")
            with st.expander("View Column Types & Samples", expanded=False):
                schema_df = classifier.get_data_schema(source_df)
                st.dataframe(schema_df, use_container_width=True, hide_index=True)
            st.divider()
        st.markdown("### 📊 PII Analytics")
        if count_df.empty:
            st.info("No PII data detected to visualize.")
            return
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.pie(count_df, values='Count', names='PII Type', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(count_df, hide_index=True, use_container_width=True)

    # --- MAIN LOGIC ---

    # A. FILE UPLOADS
    if source == "File Upload":
        ext_map = {"PDF": ["pdf"], "CSV": ["csv"], "JSON": ["json"], "Parquet": ["parquet", "pqt"]}
        accepted_exts = ext_map.get(file_sub_type, [])
        st.subheader(f"📂 {file_sub_type} Analysis")
        uploaded_file = st.file_uploader(f"Upload {file_sub_type}", type=accepted_exts)
        
        if uploaded_file:
            mask_mode = st.checkbox("🔒 Enable PII Masking")
            if file_sub_type == 'PDF':
                file_bytes = uploaded_file.getvalue()
                current_text = classifier.get_pdf_page_text(file_bytes, st.session_state.page_number)
                count_df = classifier.get_pii_counts(current_text)
                render_analytics(count_df, None)
                render_inspector(current_text)
                
                total_pages = classifier.get_pdf_total_pages(file_bytes)
                c1, c2, c3 = st.columns([1, 2, 1])
                with c1: 
                    if st.button("⬅️ Prev") and st.session_state.page_number > 0: st.session_state.page_number -= 1
                with c3:
                    if st.button("Next ➡️") and st.session_state.page_number < total_pages - 1: st.session_state.page_number += 1
                
                st.markdown(f"**Viewing Page {st.session_state.page_number + 1} of {total_pages}**")
                img = classifier.get_labeled_pdf_image(file_bytes, st.session_state.page_number)
                if img: st.image(img, use_container_width=True)
            else:
                if file_sub_type == 'Parquet': df = classifier.get_parquet_data(uploaded_file.getvalue())
                elif file_sub_type == 'CSV': df = pd.read_csv(uploaded_file)
                else: df = classifier.get_json_data(uploaded_file) 

                render_analytics(classifier.get_pii_counts_dataframe(df), df)
                sample_text = df.head(10).to_string()
                render_inspector(sample_text)

                if mask_mode: st.dataframe(classifier.mask_dataframe(df).head(50))
                else: st.markdown(classifier.scan_dataframe_with_html(df.head(50)).to_html(escape=False), unsafe_allow_html=True)

    # B. DATABASES
    elif source in ["PostgreSQL", "MySQL", "MongoDB"]:
        db_logos = {
            "PostgreSQL": "https://upload.wikimedia.org/wikipedia/commons/2/29/Postgresql_elephant.svg",
            "MySQL": "https://upload.wikimedia.org/wikipedia/commons/b/b2/Database-mysql.svg",
            "MongoDB": "https://upload.wikimedia.org/wikipedia/commons/9/93/MongoDB_Logo.svg"
        }
        render_source_header(f"Connect to {source}", db_logos.get(source, ""))
        c1, c2 = st.columns(2)
        host = c1.text_input("Host", "localhost")
        default_port = "27017" if source == "MongoDB" else ("3306" if source == "MySQL" else "5432")
        port = c2.text_input("Port", default_port)
        user = c1.text_input("User", "root" if source == "MySQL" else "postgres")
        pw = c2.text_input("Password", type="password")
        db = c1.text_input("Database Name")
        table_label = "Collection Name" if source == "MongoDB" else "Table Name"
        table = c2.text_input(table_label)

        if st.button("Connect & Scan"):
            try:
                if source == "MongoDB": df = classifier.get_mongodb_data(host, port, db, user, pw, table)
                elif source == "MySQL": df = classifier.get_mysql_data(host, port, db, user, pw, table)
                else: df = classifier.get_postgres_data(host, port, db, user, pw, table)
                st.session_state.db_data = df
                st.rerun()
            except Exception as e: st.error(f"Connection Failed: {e}")

        if 'db_data' in st.session_state:
            df = st.session_state.db_data
            render_analytics(classifier.get_pii_counts_dataframe(df), df)
            sample_text = df.head(10).to_string()
            render_inspector(sample_text)
            st.dataframe(classifier.mask_dataframe(df))

    # C. CLOUD STORAGE
    elif source == "Google Drive":
        render_source_header("Google Drive Import", "https://upload.wikimedia.org/wikipedia/commons/d/da/Google_Drive_logo.png")
        st.info("Upload your Service Account JSON to connect dynamically.")
        creds_file = st.file_uploader("Upload credentials.json", type=['json'])
        if creds_file:
            creds_dict = json.load(creds_file)
            st.session_state.creds_dict = creds_dict
            st.success("Credentials Loaded!")
            if st.button("📂 List Files"):
                st.session_state.drive_files = classifier.get_google_drive_files(creds_dict)
        if 'drive_files' in st.session_state:
            files = st.session_state.drive_files
            if not files: st.warning("No files found.")
            else:
                file_map = {f['name']: f for f in files}
                selected_name = st.selectbox("Select File", list(file_map.keys()))
                if st.button("⬇️ Scan File"):
                    sel_file = file_map[selected_name]
                    content = classifier.download_drive_file(sel_file['id'], sel_file.get('mimeType', ''), st.session_state.creds_dict)
                    if not content: st.error("Failed to read.")
                    else:
                        st.success(f"Scanning {selected_name}...")
                        mask_mode = st.checkbox("🔒 Mask Results", value=False, key="drive_mask")
                        mime = sel_file.get('mimeType', '').lower()
                        is_pdf = "pdf" in mime or "document" in mime
                        is_csv = "csv" in mime or "spreadsheet" in mime
                        is_json = "json" in mime
                        if is_pdf:
                            text = classifier.get_pdf_page_text(content, 0)
                            render_analytics(classifier.get_pii_counts(text), None)
                            render_inspector(text)
                            img = classifier.get_labeled_pdf_image(content, 0)
                            if img: st.image(img, caption="Page 1 Preview")
                        elif is_csv:
                            df = pd.read_csv(io.BytesIO(content))
                            render_analytics(classifier.get_pii_counts_dataframe(df), df)
                            render_inspector(df.head(10).to_string())
                            if mask_mode: st.dataframe(classifier.mask_dataframe(df))
                            else: st.markdown(classifier.scan_dataframe_with_html(df).to_html(escape=False), unsafe_allow_html=True)
                        elif is_json:
                            df = classifier.get_json_data(io.BytesIO(content))
                            render_analytics(classifier.get_pii_counts_dataframe(df), df)
                            render_inspector(df.head(10).to_string())
                            if mask_mode: st.dataframe(classifier.mask_dataframe(df))
                            else: st.markdown(classifier.scan_dataframe_with_html(df).to_html(escape=False), unsafe_allow_html=True)

    # D. AWS S3 (NEW)
    elif source == "AWS S3":
        render_source_header("AWS S3 Import", "https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg")
        
        c1, c2, c3 = st.columns(3)
        aws_access = c1.text_input("Access Key ID")
        aws_secret = c2.text_input("Secret Access Key", type="password")
        aws_region = c3.text_input("Region", "us-east-1")

        if st.button("🔗 Connect to AWS"):
            buckets = classifier.get_s3_buckets(aws_access, aws_secret, aws_region)
            if buckets:
                st.session_state.aws_creds = (aws_access, aws_secret, aws_region)
                st.session_state.s3_buckets = buckets
                st.success(f"Connected! Found {len(buckets)} buckets.")
            else:
                st.error("Connection Failed or No Buckets found.")

        if 's3_buckets' in st.session_state:
            selected_bucket = st.selectbox("Select Bucket", st.session_state.s3_buckets)
            
            if st.button("📂 List Files in Bucket"):
                creds = st.session_state.aws_creds
                st.session_state.s3_files = classifier.get_s3_files(creds[0], creds[1], creds[2], selected_bucket)
            
            if 's3_files' in st.session_state and st.session_state.s3_files:
                selected_file = st.selectbox("Select File", st.session_state.s3_files)
                
                if st.button("⬇️ Download & Scan"):
                    creds = st.session_state.aws_creds
                    file_content = classifier.download_s3_file(creds[0], creds[1], creds[2], selected_bucket, selected_file)
                    
                    if not file_content:
                        st.error("Failed to download file.")
                    else:
                        st.success(f"Scanning {selected_file}...")
                        mask_mode = st.checkbox("🔒 Mask Results", value=False, key="s3_mask")
                        
                        # Determine type by extension
                        ext = selected_file.split('.')[-1].lower()
                        
                        if ext == 'pdf':
                            text = classifier.get_pdf_page_text(file_content, 0)
                            render_analytics(classifier.get_pii_counts(text), None)
                            render_inspector(text)
                            img = classifier.get_labeled_pdf_image(file_content, 0)
                            if img: st.image(img, caption="Page 1 Preview")
                        elif ext == 'csv':
                            df = pd.read_csv(io.BytesIO(file_content))
                            render_analytics(classifier.get_pii_counts_dataframe(df), df)
                            render_inspector(df.head(10).to_string())
                            if mask_mode: st.dataframe(classifier.mask_dataframe(df))
                            else: st.markdown(classifier.scan_dataframe_with_html(df).to_html(escape=False), unsafe_allow_html=True)
                        elif ext == 'json':
                            df = classifier.get_json_data(io.BytesIO(file_content))
                            render_analytics(classifier.get_pii_counts_dataframe(df), df)
                            render_inspector(df.head(10).to_string())
                            if mask_mode: st.dataframe(classifier.mask_dataframe(df))
                            else: st.markdown(classifier.scan_dataframe_with_html(df).to_html(escape=False), unsafe_allow_html=True)
                        elif ext in ['parquet', 'pqt']:
                            df = classifier.get_parquet_data(file_content)
                            render_analytics(classifier.get_pii_counts_dataframe(df), df)
                            render_inspector(df.head(10).to_string())
                            if mask_mode: st.dataframe(classifier.mask_dataframe(df))
                            else: st.markdown(classifier.scan_dataframe_with_html(df).to_html(escape=False), unsafe_allow_html=True)
            elif 's3_files' in st.session_state:
                st.warning("Bucket is empty.")

if __name__ == "__main__":
    main()