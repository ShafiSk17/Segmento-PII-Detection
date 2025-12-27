from gliner import GLiNER

class PiiGlinerAnalyzer:
    def __init__(self, model_name="urchade/gliner_small-v2.1"):
        """
        Initializes the GLiNER model.
        Uses a small, efficient BERT-based model by default.
        """
        self.model = None
        self.available = False
        
        # Define the natural language labels you want GLiNER to look for.
        # These are used as prompts for the model.
        self.labels = [
            "person", 
            "email", 
            "phone number", 
            "credit card", 
            "social security number", 
            "organization", 
            "location", 
            "date", 
            "ip address",
            "passport number",
            "driver license"
        ]
        
        try:
            print(f"⏳ Loading GLiNER model: {model_name}...")
            # This will download the model to your local cache on the first run
            self.model = GLiNER.from_pretrained(model_name)
            self.available = True
            print("✅ GLiNER model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading GLiNER: {e}")

    def scan(self, text: str) -> list:
        """
        Scans text using GLiNER and normalizes the output for the Inspector.
        """
        if not self.available or not text or not text.strip():
            return []

        try:
            # GLiNER takes text and a list of labels as input
            # Threshold 0.5 is a good balance for the small model
            entities = self.model.predict_entities(text, self.labels, threshold=0.5)
            
            detections = []
            
            # Map GLiNER's lowercase output labels to your App's standard uppercase keys
            # to ensure consistency in the UI and Inspector.
            label_map = {
                "person": "FIRST_NAME",
                "phone number": "PHONE",
                "social security number": "SSN",
                "organization": "ORG",
                "location": "LOCATION",
                "ip address": "IP_ADDRESS",
                "credit card": "CREDIT_CARD",
                "email": "EMAIL",
                "date": "DATE_TIME",
                "passport number": "PASSPORT",
                "driver license": "DRIVER_LICENSE"
            }

            for ent in entities:
                detections.append({
                    "label": label_map.get(ent["label"], ent["label"].upper().replace(" ", "_")),
                    "text": ent["text"],
                    "start": ent["start"],
                    "end": ent["end"],
                    "score": ent["score"],
                    "source": "GLiNER" # Helpful metadata
                })
            
            return detections

        except Exception as e:
            print(f"⚠️ GLiNER Scan Error: {e}")
            return []