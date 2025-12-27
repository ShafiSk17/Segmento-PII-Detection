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

    def compare_models(self, regex_matches, nltk_matches, spacy_matches, presidio_matches, gliner_matches):
        """
        Compares 5 lists of matches to find Unique vs Missed PII.
        Added GLiNER to the comparison logic.
        """
        all_detections = {}
        
        def add_to_master(matches, model_name):
            found_set = set()
            for m in matches:
                # Use tuple key for uniqueness: (start, end, text)
                key = (m['start'], m['end'], m['text']) 
                if key not in all_detections:
                    all_detections[key] = {'text': m['text'], 'label': m['label']}
                found_set.add(key)
            return found_set

        # 1. Track what each model found
        regex_set = add_to_master(regex_matches, "Regex")
        nltk_set = add_to_master(nltk_matches, "NLTK")
        spacy_set = add_to_master(spacy_matches, "SpaCy")
        presidio_set = add_to_master(presidio_matches, "Presidio")
        gliner_set = add_to_master(gliner_matches, "GLiNER") # <--- Added GLiNER

        # 2. Calculate "Missed" Data (Union of all models)
        total_unique_pii = set(all_detections.keys())
        
        regex_missed = total_unique_pii - regex_set
        nltk_missed = total_unique_pii - nltk_set
        spacy_missed = total_unique_pii - spacy_set
        presidio_missed = total_unique_pii - presidio_set
        gliner_missed = total_unique_pii - gliner_set # <--- Added GLiNER

        def fmt(item_set):
            items = [all_detections[k]['text'] for k in item_set]
            # Limiting to first 5 items to prevent UI clutter if list is huge
            display_items = items[:5]
            res = ", ".join(display_items)
            if len(items) > 5:
                res += f", (+{len(items)-5} more)"
            return res if res else "None"

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
            },
            {
                "Model": "🦅 GLiNER",
                "Detected PII": fmt(gliner_set),
                "Missed PII": fmt(gliner_missed),
                "Accuracy": len(gliner_set) / total_count,
                "Count": len(gliner_set)
            }
        ]

        # Return sorted by Accuracy descending so best model is on top
        return pd.DataFrame(stats).sort_values(by="Accuracy", ascending=False)