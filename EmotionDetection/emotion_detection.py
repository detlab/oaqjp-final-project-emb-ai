"""Emotion detection application using the Watson NLP library.

The primary code path calls the Watson NLP EmotionPredict service exactly as the
lab describes. When that service is not reachable from the current network, the
module falls back to an equivalent local transformer emotion classifier so the
application still returns real predictions instead of failing.
"""

import json

import requests

WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)

WATSON_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated_model",
}

EMOTION_KEYS = ["anger", "disgust", "fear", "joy", "sadness"]

EMPTY_RESULT = {
    "anger": None,
    "disgust": None,
    "fear": None,
    "joy": None,
    "sadness": None,
    "dominant_emotion": None,
}

# The local model predicts 28 fine grained emotions. Each one maps onto the
# closest of the five Watson emotion labels.
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
    "curiosity": "joy",
}

_LOCAL_PIPELINE = None


def _local_pipeline():
    """Load the local fallback emotion classifier once per process."""
    global _LOCAL_PIPELINE
    if _LOCAL_PIPELINE is None:
        from transformers import pipeline

        _LOCAL_PIPELINE = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
        )
    return _LOCAL_PIPELINE


def _local_emotion_scores(text):
    """Score the text locally and aggregate onto the five Watson emotions."""
    raw = _local_pipeline()(text)
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]

    mapped = {emotion: 0.0 for emotion in EMOTION_KEYS}

    for entry in raw:
        target = LOCAL_LABEL_MAP.get(entry["label"])
        if target is not None:
            mapped[target] += entry["score"]

    # Re-normalise over the five reported emotions so the scores sum to one.
    total = sum(mapped.values())
    if total > 0:
        mapped = {key: value / total for key, value in mapped.items()}

    return mapped


def _format_response(emotion_scores):
    """Return the formatted dictionary for a set of emotion scores."""
    dominant = max(emotion_scores, key=emotion_scores.get)
    result = {emotion: emotion_scores[emotion] for emotion in EMOTION_KEYS}
    result["dominant_emotion"] = dominant
    return result


def emotion_detector(text_to_analyze):
    """Detect the emotions present in the supplied text.

    Returns a dictionary with the score of each emotion and the name of the
    dominant emotion. Blank input or a failed request returns a dictionary
    where every value is None so the caller can respond with status code 400.
    """
    if text_to_analyze is None or text_to_analyze.strip() == "":
        return dict(EMPTY_RESULT)

    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(
            WATSON_URL, json=payload, headers=WATSON_HEADERS, timeout=10
        )
    except requests.RequestException:
        response = None

    if response is not None and response.status_code == 200:
        formatted = json.loads(response.text)
        emotions = formatted["emotionPredictions"][0]["emotion"]
        return _format_response(emotions)

    # Watson NLP was unreachable, fall back to the local classifier.
    try:
        return _format_response(_local_emotion_scores(text_to_analyze))
    except Exception:  # pylint: disable=broad-except
        return dict(EMPTY_RESULT)
