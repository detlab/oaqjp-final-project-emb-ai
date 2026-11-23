"""Flask web deployment of the emotion detection application."""

from flask import Flask, request, render_template

from EmotionDetection.emotion_detection import emotion_detector

app = Flask("Emotion Detector")


def _format_result(response):
    """Build the human readable sentence for a detection result."""
    return (
        "For the given statement, the system response is "
        f"'anger': {response['anger']}, 'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, 'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}."
    )


@app.route("/emotionDetector")
def sent_detector():
    """Analyse the supplied text and return the emotion scores as text."""
    text_to_analyze = request.args.get("textToAnalyze")
    response = emotion_detector(text_to_analyze)

    if response["dominant_emotion"] is None:
        return "Invalid text! Please try again!"

    return _format_result(response)


@app.route("/")
def render_index_page():
    """Render the main page, optionally showing a detection result."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    result_text = ""
    error_text = ""

    if text_to_analyze is not None and text_to_analyze != "":
        response = emotion_detector(text_to_analyze)
        if response["dominant_emotion"] is None:
            error_text = "Invalid text! Please try again!"
        else:
            result_text = _format_result(response)

    return render_template(
        "index.html",
        text_to_analyze=text_to_analyze,
        result_text=result_text,
        error_text=error_text,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
