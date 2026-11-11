"""Emotion detection module using the Embeddable Watson NLP library."""

import requests

URL = ("https://sn-watson-emotion.labs.skills.network/v1/"
       "watson.runtime.nlp.v1/NlpService/EmotionPredict")
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}


def emotion_detector(text_to_analyze):
    """Detect emotions in the given text and return the scores and dominant emotion."""
    if not text_to_analyze or not text_to_analyze.strip():
        return {"anger": None, "disgust": None, "fear": None,
                "joy": None, "sadness": None, "dominant_emotion": None}

    payload = {"raw_document": {"text": text_to_analyze}}
    try:
        response = requests.post(URL, json=payload, headers=HEADERS, timeout=30)
    except requests.exceptions.RequestException:
        return {"anger": None, "disgust": None, "fear": None,
                "joy": None, "sadness": None, "dominant_emotion": None}

    if response.status_code != 200:
        return {"anger": None, "disgust": None, "fear": None,
                "joy": None, "sadness": None, "dominant_emotion": None}

    formatted = response.json()["emotionPredictions"][0]["emotion"]
    result = {
        "anger": formatted["anger"],
        "disgust": formatted["disgust"],
        "fear": formatted["fear"],
        "joy": formatted["joy"],
        "sadness": formatted["sadness"],
    }
    result["dominant_emotion"] = max(result, key=result.get)
    return result
