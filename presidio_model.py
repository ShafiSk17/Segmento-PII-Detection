from presidio_analyzer import AnalyzerEngine

class PiiPresidioAnalyzer:
    def __init__(self):
        """
        Initializes Microsoft Presidio Analyzer.
        Presidio uses SpaCy (en_core_web_lg) under the hood for NER,
        plus its own context/regex recognizers.
        """
        self.analyzer = None
        self.available = False
        
        try:
            print("⏳ Loading Microsoft Presidio...")
            self.analyzer = AnalyzerEngine()
            self.available = True
            print("✅ Presidio Analyzer loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading Presidio: {e}")
            print("👉 Ensure 'presidio-analyzer' is installed and 'en_core_web_lg' is downloaded.")
            self.available = False

    def scan(self, text: str) -> list:
        """
        Scans text using Presidio and normalizes output.
        """
        if not self.available or not text:
            return []

        try:
            # Analyze text (English by default)
            results = self.analyzer.analyze(text=text, language='en')
            detections = []

            # Map Presidio's standard labels to Segmento Sense labels
            label_map = {
                "PERSON": "FIRST_NAME",
                "LOCATION": "LOCATION",
                "EMAIL_ADDRESS": "EMAIL",
                "PHONE_NUMBER": "PHONE",
                "CREDIT_CARD": "CREDIT_CARD",
                "US_SSN": "SSN",
                "IP_ADDRESS": "IP_ADDRESS",
                "ORGANIZATION": "ORG"
            }

            for res in results:
                # Get the actual text from the start/end indices
                detected_text = text[res.start:res.end]
                
                # Use mapped label if available, otherwise use Presidio's label
                label = label_map.get(res.entity_type, res.entity_type)

                detections.append({
                    "label": label,
                    "text": detected_text,
                    "start": res.start,
                    "end": res.end,
                    "score": res.score # Presidio provides a confidence score (0-1.0)
                })
            
            return detections

        except Exception as e:
            print(f"⚠️ Presidio Scan Error: {e}")
            return []

    def retrain(self, training_data):
        """
        Placeholder: Presidio allows adding 'Ad-hoc Recognizers' or 
        customizing the underlying NLP model.
        """
        pass