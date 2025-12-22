import spacy

class PiiSpacyAnalyzer:
    def __init__(self, model_name="en_core_web_lg"):
        """
        Initializes the SpaCy model.
        Loads the 'large' model by default for better accuracy.
        """
        self.nlp = None
        self.available = False
        
        try:
            print(f"⏳ Loading SpaCy model: {model_name}...")
            self.nlp = spacy.load(model_name)
            self.available = True
            print(f"✅ SpaCy model '{model_name}' loaded successfully.")
        except OSError:
            print(f"❌ Error: Model '{model_name}' not found.")
            print("👉 Run: python -m spacy download en_core_web_lg")
            # Optional: Fallback to small model if large fails
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.available = True
                print("⚠️ Loaded fallback model 'en_core_web_sm'.")
            except:
                self.available = False

    def scan(self, text: str) -> list:
        """
        Scans text for Named Entities (Person, Org, GPE).
        Returns a list of dictionaries compatible with backend.py.
        """
        if not self.available or not text:
            return []

        # Prevent errors on huge texts by increasing limit temporarily
        self.nlp.max_length = len(text) + 1000
        
        doc = self.nlp(text)
        detections = []

        # Map SpaCy labels to your App's standard labels
        label_map = {
            "PERSON": "FIRST_NAME",
            "GPE": "LOCATION",      # Countries, Cities, States
            "ORG": "ORG"            # Companies, Agencies, Institutions
        }

        for ent in doc.ents:
            if ent.label_ in label_map:
                detections.append({
                    "label": label_map[ent.label_],
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        
        return detections

    def retrain(self, training_data):
        """
        Placeholder for future dynamic training logic.
        You can implement the training loop here later without touching backend.py.
        """
        pass