import streamlit as st
import re
import pandas as pd
import json
from typing import Dict, List
from pypdf import PdfReader
from io import StringIO

# ==========================================
# 1. CORE LOGIC (The Brains)
# ==========================================

class RegexClassifier:
    """
    The engine that scans files for PII using Regex.
    Now supports CSV, PDF, and JSON.
    """
    def __init__(self):
        # Default patterns
        self.patterns: Dict[str, str] = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "PHONE": r"\b\d{10}\b",
            "CREDIT_CARD": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b"
        }

    # --- Pattern Management ---
    def add_pattern(self, name: str, regex: str):
        self.patterns[name.upper()] = regex

    def remove_pattern(self, name: str):
        self.patterns.pop(name.upper(), None)

    def list_patterns(self) -> Dict[str, str]:
        return self.patterns

    # --- File Scanners ---
    
    def scan_csv(self, uploaded_file) -> List[dict]:
        """Reads a CSV file object from Streamlit."""
        detections = []
        try:
            # Streamlit uploads are file-like objects, pandas can read them directly
            df = pd.read_csv(uploaded_file)
            
            for row_index, row in df.iterrows():
                for column_name, cell_value in row.items():
                    if pd.isna(cell_value): continue
                    
                    self._check_text(
                        text=str(cell_value),
                        location=f"Row {row_index}, Col '{column_name}'",
                        file_type="CSV",
                        detections_list=detections
                    )
        except Exception as e:
            st.error(f"Error scanning CSV: {e}")
        return detections

    def scan_pdf(self, uploaded_file) -> List[dict]:
        """Reads a PDF file object from Streamlit."""
        detections = []
        try:
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    self._check_text(
                        text=text,
                        location=f"Page {page_num + 1}",
                        file_type="PDF",
                        detections_list=detections
                    )
        except Exception as e:
            st.error(f"Error scanning PDF: {e}")
        return detections

    def scan_json(self, uploaded_file) -> List[dict]:
        """Reads a JSON file object. handles nested data recursively."""
        detections = []
        try:
            data = json.load(uploaded_file)
            # Start the recursive search
            self._scan_json_recursive(data, "", detections)
        except Exception as e:
            st.error(f"Error scanning JSON: {e}")
        return detections

    # --- Helpers ---

    def _scan_json_recursive(self, data, path: str, detections: list):
        """Helper to go deep into nested JSON structures."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._scan_json_recursive(value, new_path, detections)
        elif isinstance(data, list):
            for index, item in enumerate(data):
                new_path = f"{path}[{index}]"
                self._scan_json_recursive(item, new_path, detections)
        else:
            # It's a value (string, int, etc.), check it!
            self._check_text(str(data), f"Key: {path}", "JSON", detections)

    def _check_text(self, text: str, location: str, file_type: str, detections_list: list):
        """Common logic to run all regex patterns on a piece of text."""
        for name, regex in self.patterns.items():
            matches = re.findall(regex, text)
            for match in matches:
                detections_list.append({
                    "File Type": file_type,
                    "Pattern": name,
                    "Detected Value": match,
                    "Location": location
                })

# ==========================================
# 2. STREAMLIT UI (The Frontend)
# ==========================================

def main():
    st.set_page_config(page_title="PII Detector", layout="wide")
    st.title("🛡️ PII Detection Tool")
    st.markdown("Upload a file (**CSV, PDF, or JSON**) to detect sensitive information based on Regex patterns.")

    # --- A. Initialize Session State ---
    # We need this to remember the "Classifier" object when the app refreshes
    if 'classifier' not in st.session_state:
        st.session_state.classifier = RegexClassifier()

    classifier = st.session_state.classifier

    # --- B. Sidebar: Pattern Management ---
    with st.sidebar:
        st.header("⚙️ Manage Patterns")
        
        # Display current patterns
        st.subheader("Current Regex Rules")
        patterns = classifier.list_patterns()
        st.dataframe(pd.DataFrame(list(patterns.items()), columns=["Name", "Regex"]), hide_index=True)

        st.divider()

        # Add New Pattern
        st.subheader("Add New Pattern")
        new_name = st.text_input("Pattern Name (e.g., IP_ADDR)")
        new_regex = st.text_input("Regex (e.g., \\d{1,3}\\.\\d{1,3}...)")
        
        if st.button("Add Pattern"):
            if new_name and new_regex:
                try:
                    re.compile(new_regex) # Test if regex is valid
                    classifier.add_pattern(new_name, new_regex)
                    st.success(f"Added {new_name}!")
                    st.rerun() # Refresh to show new pattern in list
                except re.error:
                    st.error("Invalid Regex pattern!")
            else:
                st.warning("Please fill both fields.")

        st.divider()

        # Remove Pattern
        st.subheader("Remove Pattern")
        pattern_to_remove = st.selectbox("Select pattern to remove", options=list(patterns.keys()))
        if st.button("Remove Selected"):
            classifier.remove_pattern(pattern_to_remove)
            st.success(f"Removed {pattern_to_remove}!")
            st.rerun()

    # --- C. Main Area: File Upload & Scanning ---
    st.header("📂 File Analysis")
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'pdf', 'json'])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        st.info(f"Analyzing **{uploaded_file.name}** as {file_type.upper()}...")

        results = []
        
        # Route to correct scanner based on file extension
        if file_type == 'csv':
            results = classifier.scan_csv(uploaded_file)
        elif file_type == 'pdf':
            results = classifier.scan_pdf(uploaded_file)
        elif file_type == 'json':
            results = classifier.scan_json(uploaded_file)

        # --- D. Display Results ---
        if results:
            st.error(f"🚨 Found {len(results)} PII matches!")
            
            # Convert to DataFrame for a nice table
            df_results = pd.DataFrame(results)
            
            # Show interactive table
            st.dataframe(
                df_results, 
                use_container_width=True,
                column_config={
                    "Detected Value": st.column_config.TextColumn("Detected Value", help="The sensitive text found")
                }
            )

            # Download button for report
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Report (CSV)",
                data=csv,
                file_name="pii_scan_results.csv",
                mime="text/csv",
            )
        else:
            st.success("✅ Clean! No PII detected matching the current patterns.")

if __name__ == "__main__":
    main()