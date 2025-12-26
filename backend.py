# backend.py
import re
import json
import pandas as pd
import fitz  # PyMuPDF
import nltk
import io
import os
import pickle
import base64
from typing import Dict, List, Any
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from bs4 import BeautifulSoup 

# --- IMPORT MODULES ---
from spacy_model import PiiSpacyAnalyzer
from presidio_model import PiiPresidioAnalyzer
from inspector import ModelInspector
from ocr_engine import OcrEngine  # <--- NEW IMPORT

# --- DEPENDENCY CHECKS ---
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Google Drive/Gmail Libraries not installed.")

try:
    import pymongo
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    print("PyMongo not installed.")

try:
    import pyarrow
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("PyArrow not installed.")

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    print("Boto3 not installed.")

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("Azure Storage Blob not installed.")

# --- GCP STORAGE IMPORT ---
try:
    from google.cloud import storage
    from google.oauth2 import service_account as gcp_service_account
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Google Cloud Storage library not installed.")

# --- NLTK SETUP ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('punkt_tab')

class RegexClassifier:
    def __init__(self):
        self.colors = {
            "EMAIL": (136, 238, 255), "FIRST_NAME": (170, 255, 170), "LAST_NAME": (170, 255, 170),
            "PHONE": (255, 170, 170), "SSN": (255, 204, 170), "CREDIT_CARD": (255, 238, 170),
            "LOCATION": (200, 170, 255), "AADHAAR_IND": (255, 150, 255), "ORG": (255, 255, 150), 
            "DEFAULT": (224, 224, 224)
        }
        
        self.patterns: Dict[str, str] = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "PHONE": r"\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "CREDIT_CARD": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "AADHAAR_IND": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b",
            "PAN_IND": r"\b[A-Z]{5}\d{4}[A-Z]{1}\b",
        }

        self.spacy_analyzer = PiiSpacyAnalyzer()
        self.presidio_analyzer = PiiPresidioAnalyzer()
        self.inspector = ModelInspector()
        self.ocr_engine = OcrEngine()  # <--- Initialize OCR

    def list_patterns(self): return self.patterns
    def add_pattern(self, n, r): self.patterns[n.upper()] = r
    def remove_pattern(self, n): self.patterns.pop(n.upper(), None)

    # --- DETECTION ENGINES ---
    def scan_with_regex(self, text: str) -> List[dict]:
        matches = []
        for label, regex in self.patterns.items():
            for match in re.finditer(regex, text):
                matches.append({"label": label, "text": match.group(), "start": match.start(), "end": match.end()})
        return matches

    def scan_with_nltk(self, text: str) -> List[dict]:
        detections = []
        try:
            tokens = nltk.word_tokenize(text)
            chunked = nltk.ne_chunk(nltk.pos_tag(tokens), binary=False)
            current_pos = 0 
            for chunk in chunked:
                if hasattr(chunk, 'label') and chunk.label() in ['PERSON', 'GPE']:
                    val = " ".join(c[0] for c in chunk)
                    start_idx = text.find(val, current_pos)
                    label = "LOCATION" if chunk.label() == 'GPE' else "FIRST_NAME" 
                    if start_idx != -1:
                        detections.append({"label": label, "text": val, "start": start_idx, "end": start_idx + len(val)})
                        current_pos = start_idx + len(val)
        except: pass 
        return detections

    def analyze_text_hybrid(self, text: str) -> List[dict]:
        all_matches = []
        all_matches.extend(self.scan_with_regex(text))
        all_matches.extend(self.scan_with_nltk(text))
        all_matches.extend(self.spacy_analyzer.scan(text))
        all_matches.extend(self.presidio_analyzer.scan(text))
        
        all_matches.sort(key=lambda x: x['start'])
        
        unique_matches = []
        if not all_matches: return []
        curr = all_matches[0]
        for next_match in all_matches[1:]:
            if next_match['start'] < curr['end']:
                if len(next_match['text']) > len(curr['text']):
                    curr = next_match
            else:
                unique_matches.append(curr)
                curr = next_match
        unique_matches.append(curr)
        return unique_matches

    def run_full_inspection(self, text: str) -> pd.DataFrame:
        r_matches = self.scan_with_regex(text)
        n_matches = self.scan_with_nltk(text)
        s_matches = self.spacy_analyzer.scan(text)
        p_matches = self.presidio_analyzer.scan(text)
        return self.inspector.compare_models(r_matches, n_matches, s_matches, p_matches)

    # --- SUMMARY & VISUALS ---
    def get_pii_counts(self, text: str) -> pd.DataFrame:
        matches = self.analyze_text_hybrid(str(text))
        if not matches: return pd.DataFrame(columns=["PII Type", "Count"])
        counts = {}
        for m in matches: counts[m['label']] = counts.get(m['label'], 0) + 1
        return pd.DataFrame(list(counts.items()), columns=["PII Type", "Count"])

    def get_pii_counts_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        full_text = " ".join(df.astype(str).values.flatten())
        return self.get_pii_counts(full_text)

    def mask_pii(self, text: str) -> str:
        text = str(text)
        matches = self.analyze_text_hybrid(text)
        matches.sort(key=lambda x: x['start'], reverse=True)
        for m in matches:
            masked_val = "******"
            if "<span" not in text[m['start']:m['end']]:
                text = text[:m['start']] + masked_val + text[m['end']:]
        return text

    def mask_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        def safe_mask(val):
            if isinstance(val, (list, dict, tuple, set)): return self.mask_pii(str(val))
            if pd.isna(val): return val
            return self.mask_pii(str(val))
        return df.map(safe_mask)

    def get_labeled_pdf_image(self, file_bytes, page_num: int):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if not (0 <= page_num < len(doc)): return None
            page = doc[page_num]
            text = page.get_text("text")
            matches = self.analyze_text_hybrid(text)
            for m in matches:
                color_norm = tuple(c/255 for c in self.colors.get(m['label'], self.colors["DEFAULT"]))
                quads = page.search_for(m['text'])
                for quad in quads:
                    page.draw_rect(quad, color=color_norm, fill=color_norm, fill_opacity=0.4)
                    page.insert_text(fitz.Point(quad.x0, quad.y0-2), m['label'], fontsize=6, color=(0,0,0))
            return page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
        except: return None

    # --- OCR HELPERS (NEW) ---
    def get_ocr_text_from_image(self, file_bytes) -> str:
        """Runs OCR on image bytes and returns text."""
        return self.ocr_engine.extract_text(file_bytes)

    def scan_dataframe_with_html(self, df: pd.DataFrame) -> pd.DataFrame:
        def highlight_html(text):
            text = str(text)
            matches = self.analyze_text_hybrid(text)
            matches.sort(key=lambda x: x['start'], reverse=True)
            hex_map = {"EMAIL": "#8ef", "PHONE": "#faa", "SSN": "#fca", "CREDIT_CARD": "#fea", "FIRST_NAME": "#af9", "LAST_NAME": "#af9", "LOCATION": "#dcf", "AADHAAR_IND": "#f9f", "ORG": "#ffecb3", "DEFAULT": "#e0e0e0"}
            for m in matches:
                if "<span" in text[m['start']:m['end']]: continue
                color = hex_map.get(m['label'], "#e0e0e0")
                tag = f'<span style="background-color: {color}; padding: 0 2px; border-radius: 3px; border: 1px solid #ccc;">{m["text"]}</span>'
                text = text[:m['start']] + tag + text[m['end']:]
            return text
        def safe_highlight(val):
             if isinstance(val, (list, dict)): return highlight_html(str(val))
             if pd.isna(val): return val
             return highlight_html(val)
        return df.map(safe_highlight)

    def get_data_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return pd.DataFrame(columns=["Column", "Type", "Sample"])
        schema_info = []
        for col in df.columns:
            d_type = str(df[col].dtype)
            first_valid_idx = df[col].first_valid_index()
            sample_val = str(df[col].loc[first_valid_idx]) if first_valid_idx is not None else "All Null"
            if len(sample_val) > 50: sample_val = sample_val[:47] + "..."
            schema_info.append({"Column Name": col, "Data Type": d_type, "Sample Value": sample_val})
        return pd.DataFrame(schema_info)

    # --- SQL/MONGO/DRIVE/S3/AZURE CONNECTORS ---
    def get_postgres_data(self, host, port, db, user, pw, table):
        safe_pw = quote_plus(pw)
        conn_str = f"postgresql://{user}:{safe_pw}@{host}:{port}/{db}"
        engine = create_engine(conn_str)
        return pd.read_sql(f"SELECT * FROM {table} LIMIT 100", engine)

    def get_mysql_data(self, host, port, db, user, pw, table):
        safe_pw = quote_plus(pw)
        conn_str = f"mysql+pymysql://{user}:{safe_pw}@{host}:{port}/{db}"
        engine = create_engine(conn_str)
        return pd.read_sql(f"SELECT * FROM {table} LIMIT 100", engine)

    def get_mongodb_data(self, host, port, db, user, pw, collection):
        if not MONGO_AVAILABLE: return pd.DataFrame()
        try:
            if user and pw:
                safe_user = quote_plus(user)
                safe_pw = quote_plus(pw)
                uri = f"mongodb://{safe_user}:{safe_pw}@{host}:{port}/"
            else:
                uri = f"mongodb://{host}:{port}/"
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
            database = client[db]
            col = database[collection]
            cursor = col.find().limit(100)
            data_list = list(cursor)
            if not data_list: return pd.DataFrame()
            for doc in data_list:
                if '_id' in doc: doc['_id'] = str(doc['_id'])
            return pd.json_normalize(data_list)
        except Exception as e:
            print(f"Mongo Error: {e}")
            raise e

    def get_google_drive_files(self, credentials_dict):
        if not GOOGLE_AVAILABLE: return []
        try:
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            creds = service_account.Credentials.from_service_account_info(credentials_dict, scopes=SCOPES)
            service = build('drive', 'v3', credentials=creds)
            return service.files().list(pageSize=15, fields="files(id, name, mimeType)").execute().get('files', [])
        except Exception as e:
            return []

    def download_drive_file(self, file_id, mime_type, credentials_dict):
        if not GOOGLE_AVAILABLE: return b""
        try:
            SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
            creds = service_account.Credentials.from_service_account_info(credentials_dict, scopes=SCOPES)
            service = build('drive', 'v3', credentials=creds)
            if "spreadsheet" in mime_type: request = service.files().export_media(fileId=file_id, mimeType='text/csv')
            elif "document" in mime_type: request = service.files().export_media(fileId=file_id, mimeType='application/pdf')
            elif "presentation" in mime_type: request = service.files().export_media(fileId=file_id, mimeType='application/pdf')
            else: request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False: status, done = downloader.next_chunk()
            return fh.getvalue()
        except: return b""

    # --- GMAIL INTEGRATION ---
    def get_gmail_data(self, credentials_file, num_emails=10) -> pd.DataFrame:
        if not GOOGLE_AVAILABLE:
            print("Google libraries not installed.")
            return pd.DataFrame()

        SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
        creds = None
        token_path = 'token.pickle'
        
        if os.path.exists(token_path):
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                with open("temp_client_secret.json", "wb") as f:
                    f.write(credentials_file.getvalue())
                
                flow = InstalledAppFlow.from_client_secrets_file('temp_client_secret.json', SCOPES)
                creds = flow.run_local_server(port=0)
                
                with open(token_path, 'wb') as token:
                    pickle.dump(creds, token)
                
                if os.path.exists("temp_client_secret.json"):
                    os.remove("temp_client_secret.json")

        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', maxResults=num_emails).execute()
        messages = results.get('messages', [])
        
        email_data = []

        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            payload = msg['payload']
            headers = payload.get("headers")
            
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
            sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown")

            body = ""
            if 'parts' in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                        data = part['body']['data']
                        body += base64.urlsafe_b64decode(data).decode()
            elif 'body' in payload and 'data' in payload['body']:
                 data = payload['body']['data']
                 body += base64.urlsafe_b64decode(data).decode()

            clean_body = BeautifulSoup(body, "html.parser").get_text()
            email_data.append({
                "Source": "Gmail",
                "Sender": sender,
                "Subject": subject,
                "Content": f"Subject: {subject}\n\n{clean_body}"
            })

        return pd.DataFrame(email_data)

    def get_s3_buckets(self, access_key, secret_key, region):
        if not AWS_AVAILABLE: return []
        try:
            s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
            response = s3.list_buckets()
            return [b['Name'] for b in response.get('Buckets', [])]
        except Exception as e:
            print(f"S3 Error: {e}")
            return []

    def get_s3_files(self, access_key, secret_key, region, bucket_name):
        if not AWS_AVAILABLE: return []
        try:
            s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
            response = s3.list_objects_v2(Bucket=bucket_name)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            return []

    def download_s3_file(self, access_key, secret_key, region, bucket_name, file_key):
        if not AWS_AVAILABLE: return b""
        try:
            s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
            obj = s3.get_object(Bucket=bucket_name, Key=file_key)
            return obj['Body'].read()
        except Exception as e:
            return b""

    def get_azure_containers(self, conn_str):
        if not AZURE_AVAILABLE: return []
        try:
            blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            containers = blob_service_client.list_containers()
            return [c['name'] for c in containers]
        except Exception as e:
            print(f"Azure Error: {e}")
            return []

    def get_azure_blobs(self, conn_str, container_name):
        if not AZURE_AVAILABLE: return []
        try:
            blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            container_client = blob_service_client.get_container_client(container_name)
            blobs = container_client.list_blobs()
            return [b['name'] for b in blobs]
        except Exception as e:
            return []

    def download_azure_blob(self, conn_str, container_name, blob_name):
        if not AZURE_AVAILABLE: return b""
        try:
            blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            return blob_client.download_blob().readall()
        except Exception as e:
            return b""

    # --- GCP BUCKET CONNECTORS ---
    def get_gcs_buckets(self, credentials_dict):
        if not GCS_AVAILABLE: return []
        try:
            credentials = gcp_service_account.Credentials.from_service_account_info(credentials_dict)
            storage_client = storage.Client(credentials=credentials, project=credentials_dict.get('project_id'))
            buckets = storage_client.list_buckets()
            return [bucket.name for bucket in buckets]
        except Exception as e:
            print(f"GCP Bucket Error: {e}")
            return []

    def get_gcs_files(self, credentials_dict, bucket_name):
        if not GCS_AVAILABLE: return []
        try:
            credentials = gcp_service_account.Credentials.from_service_account_info(credentials_dict)
            storage_client = storage.Client(credentials=credentials, project=credentials_dict.get('project_id'))
            blobs = storage_client.list_blobs(bucket_name)
            return [blob.name for blob in blobs]
        except Exception as e:
            print(f"GCP List Error: {e}")
            return []

    def download_gcs_file(self, credentials_dict, bucket_name, blob_name):
        if not GCS_AVAILABLE: return b""
        try:
            credentials = gcp_service_account.Credentials.from_service_account_info(credentials_dict)
            storage_client = storage.Client(credentials=credentials, project=credentials_dict.get('project_id'))
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.download_as_bytes()
        except Exception as e:
            print(f"GCP Download Error: {e}")
            return b""

    # --- FILE READERS ---
    def get_json_data(self, file_obj) -> pd.DataFrame:
        data = json.load(file_obj)
        flat = []
        def recursive(d, path):
            if isinstance(d, dict):
                for k, v in d.items(): recursive(v, f"{path}.{k}" if path else k)
            elif isinstance(d, list):
                for i, v in enumerate(d): recursive(v, f"{path}[{i}]")
            else: flat.append({"Path": path, "Value": str(d)})
        recursive(data, "")
        return pd.DataFrame(flat)

    def get_parquet_data(self, file_bytes) -> pd.DataFrame:
        if not PARQUET_AVAILABLE: return pd.DataFrame()
        try:
            return pd.read_parquet(io.BytesIO(file_bytes))
        except: return pd.DataFrame()

    def get_pdf_total_pages(self, file_bytes) -> int:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            return len(doc)
        except: return 0
    
    def get_pdf_page_text(self, file_bytes, page_num):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            return doc[page_num].get_text("text")
        except: return ""