"""EmotionDetection package.

Exposes the emotion_detector function used by the Flask server.
"""

from .emotion_detection import emotion_detector

__all__ = ["emotion_detector"]
