# inspector.py
import pandas as pd

class ModelInspector:
    def __init__(self):
        pass

    def _normalize_match(self, match):
        """Helper to create a standard format for comparison."""
        return {
            "text": match["text"],
            "label": match["label"],
            "start": match["start"],
            "end": match["end"]
        }

    def compare_models(self, regex_matches, nltk_matches, spacy_matches):
        """
        Compares 3 lists of matches to find Unique vs Missed PII.
        """
        # 1. Create a "Ground Truth" Set (Union of all detected PII)
        # We use a tuple of (start, end, text) to uniquely identify a PII entity
        all_detections = {}
        
        # Helper to add matches to the master list
        def add_to_master(matches, model_name):
            found_set = set()
            for m in matches:
                key = (m['start'], m['end'], m['text']) # Unique ID for a PII
                if key not in all_detections:
                    all_detections[key] = {'text': m['text'], 'label': m['label']}
                found_set.add(key)
            return found_set

        # 2. Track what each model found
        regex_set = add_to_master(regex_matches, "Regex")
        nltk_set = add_to_master(nltk_matches, "NLTK")
        spacy_set = add_to_master(spacy_matches, "SpaCy")

        # 3. Calculate "Missed" Data (Ground Truth - Model Found)
        total_unique_pii = set(all_detections.keys())
        
        regex_missed = total_unique_pii - regex_set
        nltk_missed = total_unique_pii - nltk_set
        spacy_missed = total_unique_pii - spacy_set

        # 4. Format Data for the User's Specific Table Request
        # Helper to format list of items into string
        def fmt(item_set):
            items = [all_detections[k]['text'] for k in item_set]
            return ", ".join(items) if items else "None"

        # Calculate Accuracy (Count Found / Total Unique PII)
        total_count = len(total_unique_pii) if len(total_unique_pii) > 0 else 1 # Avoid div/0
        
        stats = [
            {
                "Model": "🛠️ Regex",
                "Detected PII": fmt(regex_set),
                "Missed PII": fmt(regex_missed),
                "Accuracy": len(regex_set) / total_count,
                "Count": len(regex_set)
            },
            {
                "Model": "🧠 NLTK",
                "Detected PII": fmt(nltk_set),
                "Missed PII": fmt(nltk_missed),
                "Accuracy": len(nltk_set) / total_count,
                "Count": len(nltk_set)
            },
            {
                "Model": "🤖 SpaCy",
                "Detected PII": fmt(spacy_set),
                "Missed PII": fmt(spacy_missed),
                "Accuracy": len(spacy_set) / total_count,
                "Count": len(spacy_set)
            }
        ]

        return pd.DataFrame(stats)