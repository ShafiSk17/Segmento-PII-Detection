import re
import pandas as pd
from typing import Dict, List
from pypdf import PdfReader  # Added for PDF support

class RegexClassifier:
    """
    Simple Regex-based PII classifier for CSV and PDF files.
    """

    def __init__(self):
        # Default regex patterns
        self.patterns: Dict[str, str] = {
            "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "PHONE": r"\b\d{10}\b",
            "CREDIT_CARD": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
        }

    # -------------------------------
    # Pattern Management
    # -------------------------------

    def add_pattern(self, name: str, regex: str):
        """
        Add a new regex pattern.
        """
        self.patterns[name.upper()] = regex

    def remove_pattern(self, name: str):
        """
        Remove an existing regex pattern.
        """
        self.patterns.pop(name.upper(), None)

    def list_patterns(self) -> Dict[str, str]:
        """
        List all current patterns.
        """
        return self.patterns

    # -------------------------------
    # Scanners (CSV & PDF)
    # -------------------------------

    def scan_csv(self, csv_path: str) -> List[dict]:
        """
        Scan a CSV file and detect PII using regex patterns.
        """
        detections = []
        try:
            df = pd.read_csv(csv_path)
            for row_index, row in df.iterrows():
                for column_name, cell_value in row.items():
                    if pd.isna(cell_value):
                        continue

                    cell_text = str(cell_value)

                    for pattern_name, pattern_regex in self.patterns.items():
                        matches = re.findall(pattern_regex, cell_text)

                        for match in matches:
                            detections.append({
                                "file_type": "csv",
                                "location": f"Row {row_index}, Col '{column_name}'",
                                "pattern": pattern_name,
                                "value": match
                            })
        except Exception as e:
            print(f"Error scanning CSV: {e}")
            
        return detections

    def scan_pdf(self, pdf_path: str) -> List[dict]:
        """
        Scan a PDF file and detect PII using regex patterns.
        """
        detections = []
        try:
            reader = PdfReader(pdf_path)
            
            # Iterate through each page
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                
                # Scan extracted text against all patterns
                for pattern_name, pattern_regex in self.patterns.items():
                    matches = re.findall(pattern_regex, text)
                    
                    for match in matches:
                        detections.append({
                            "file_type": "pdf",
                            "location": f"Page {page_num + 1}",
                            "pattern": pattern_name,
                            "value": match
                        })
                        
        except Exception as e:
            print(f"Error scanning PDF: {e}")

        return detections

# -------------------------------
# Usage Example (Merged from File-2)
# -------------------------------
if __name__ == "__main__":
    # Initialize classifier
    classifier = RegexClassifier()

    # 1. Add custom patterns
    classifier.add_pattern(
        name="EMAIL",
        regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"

    )

    classifier.add_pattern(
        name="id",
        regex=r"\b(?!000)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"
    )
    
        # 1. Specific Match (Exact Name)
    classifier.add_pattern(
        name="CEO_NAME", 
        regex=r"\bJared\s+Wilson\b" 
    )

    classifier.add_pattern(
        name="PERSON_NAME",
        regex=r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"
    )

    # 2. Example: Add and then Remove a pattern
    classifier.add_pattern(
        name="gender",
        regex=r"[a-z]{1}"
    )
    
    classifier.remove_pattern("gender")

    print("--- Current Patterns ---")
    print(classifier.list_patterns().keys())
    print("\n")

    # # #3. Scan a CSV file (Ensure 'sample-data.csv' exists)
    # results_csv = classifier.scan_csv("sample-data.csv")
    # for r in results_csv:
    #     print(f"[CSV] {r['location']} | Detected {r['pattern']} -> {r['value']}")

    # # 4. Scan a PDF file (Ensure 'sample.pdf' exists)
    results_pdf = classifier.scan_pdf("Patient Data File.pdf")
    for r in results_pdf:
        print(f"[PDF] {r['location']} | Detected {r['pattern']} -> {r['value']}")
