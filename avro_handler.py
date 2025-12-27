# avro_handler.py
import io
import pandas as pd

class AvroHandler:
    def __init__(self):
        self.available = False
        try:
            import fastavro
            self.fastavro = fastavro
            self.available = True
            print("✅ Avro Handler loaded.")
        except ImportError:
            print("❌ fastavro not found. Please run: pip install fastavro")

    def convert_to_dataframe(self, file_bytes: bytes) -> pd.DataFrame:
        """
        Reads Avro bytes and converts them to a Pandas DataFrame.
        """
        if not self.available:
            return pd.DataFrame()

        try:
            # Create a file-like object from bytes
            f = io.BytesIO(file_bytes)
            # Use fastavro to read records
            reader = self.fastavro.reader(f)
            records = [r for r in reader]
            
            if not records:
                return pd.DataFrame()
                
            return pd.DataFrame(records)
        except Exception as e:
            print(f"⚠️ Avro Read Error: {e}")
            return pd.DataFrame()