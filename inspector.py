import pandas as pd

class ModelInspector:
    def __init__(self):
        pass

    def _normalize_match(self, match):
        return {
            "text": match["text"],
            "label": match["label"],
            "start": match["start"],
            "end": match["end"]
        }

    def compare_models(self, regex_matches, nltk_matches, spacy_matches, presidio_matches):
        """
        Compares 4 lists of matches to find Unique vs Missed PII.
        """
        all_detections = {}
        
        def add_to_master(matches, model_name):
            found_set = set()
            for m in matches:
                # Use tuple key for uniqueness
                key = (m['start'], m['end'], m['text']) 
                if key not in all_detections:
                    all_detections[key] = {'text': m['text'], 'label': m['label']}
                found_set.add(key)
            return found_set

        # 1. Track what each model found
        regex_set = add_to_master(regex_matches, "Regex")
        nltk_set = add_to_master(nltk_matches, "NLTK")
        spacy_set = add_to_master(spacy_matches, "SpaCy")
        presidio_set = add_to_master(presidio_matches, "Presidio") # <--- Added Presidio

        # 2. Calculate "Missed" Data
        total_unique_pii = set(all_detections.keys())
        
        regex_missed = total_unique_pii - regex_set
        nltk_missed = total_unique_pii - nltk_set
        spacy_missed = total_unique_pii - spacy_set
        presidio_missed = total_unique_pii - presidio_set # <--- Added Presidio

        def fmt(item_set):
            items = [all_detections[k]['text'] for k in item_set]
            return ", ".join(items) if items else "None"

        total_count = len(total_unique_pii) if len(total_unique_pii) > 0 else 1
        
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
            },
            {
                "Model": "🛡️ Presidio",
                "Detected PII": fmt(presidio_set),
                "Missed PII": fmt(presidio_missed),
                "Accuracy": len(presidio_set) / total_count,
                "Count": len(presidio_set)
            }
        ]

        return pd.DataFrame(stats)