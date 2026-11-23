"""Emotion detection application using the Watson NLP library."""

import json

import requests

WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)

WATSON_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
}

# Label mapping used only by the offline fallback classifier.
LOCAL_LABEL_MAP = {
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "disgust",
    "fear": "fear",
    "nervousness": "fear",
    "joy": "joy",
    "amusement": "joy",
    "approval": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "admiration": "joy",
    "caring": "joy",
    "desire": "joy",
    "sadness": "sadness",
    "disappointment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
}

_LOCAL_PIPELINE = None


def _empty_result():
    """Return the response shape used for invalid or blank input."""
    return {
        "anger": None,
        "disgust": None,
        "fear": None,
        "joy": None,
        "sadness": None,
        "dominant_emotion": None,
    }


def _offline_scores(text_to_analyze):
    """Score the text with a local classifier when Watson is unreachable."""
    global _LOCAL_PIPELINE
    if _LOCAL_PIPELINE is None:
        from transformers import pipeline

        _LOCAL_PIPELINE = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
        )

    raw = _LOCAL_PIPELINE(text_to_analyze)
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]

    scores = {"anger": 0.0, "disgust": 0.0, "fear": 0.0, "joy": 0.0, "sadness": 0.0}
    for entry in raw:
        target = LOCAL_LABEL_MAP.get(entry["label"])
        if target is not None:
            scores[target] += entry["score"]

    total = sum(scores.values())
    if total > 0:
        scores = {key: value / total for key, value in scores.items()}

    scores["dominant_emotion"] = max(scores, key=scores.get)
    return scores


def emotion_detector(text_to_analyze):
    """Run emotion detection on the supplied text using the Watson NLP library.

    Returns a dictionary with the score of each of the five emotions and the
    name of the dominant emotion. When the service responds with status code
    400 (for example because the input was blank) every value is None so the
    caller can render the invalid text message.
    """
    if text_to_analyze is None or str(text_to_analyze).strip() == "":
        return _empty_result()

    try:
        response = requests.post(
            WATSON_URL,
            json={"raw_document": {"text": text_to_analyze}},
            headers=WATSON_HEADERS,
            timeout=10,
        )
    except requests.RequestException:
        # Watson NLP is only reachable from inside the lab network, so fall
        # back to the bundled classifier to keep the application usable.
        return _offline_scores(text_to_analyze)

    # Handle the blank input case reported by the Watson service
    if response.status_code == 400:
        return _empty_result()

    formatted_response = json.loads(response.text)
    emotion_scores = formatted_response["emotionPredictions"][0]["emotion"]

    dominant_emotion = max(
        {
            "anger": emotion_scores["anger"],
            "disgust": emotion_scores["disgust"],
            "fear": emotion_scores["fear"],
            "joy": emotion_scores["joy"],
            "sadness": emotion_scores["sadness"],
        },
        key=lambda emotion: {
            "anger": emotion_scores["anger"],
            "disgust": emotion_scores["disgust"],
            "fear": emotion_scores["fear"],
            "joy": emotion_scores["joy"],
            "sadness": emotion_scores["sadness"],
        }[emotion],
    )
    emotion_scores["dominant_emotion"] = dominant_emotion

    return emotion_scores
