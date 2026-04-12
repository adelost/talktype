"""Sentence-boundary detection for streaming transcription chunking."""

SENTENCE_ENDS = set(".!?")


def find_last_sentence_boundary(words, full_text):
    """
    Find the last sentence boundary in a Whisper transcription.

    Whisper's word-level timestamps strip punctuation from individual
    words (word="videon" even when full text says "videon!"), so we
    search the full text for sentence endings, count words up to that
    point, and look up the timestamp from the words array.

    Args:
        words: list of word objects with .word, .start, .end attributes
               (or dicts with those keys)
        full_text: the complete transcription text (with punctuation)

    Returns:
        (text_up_to_boundary, cut_timestamp) or None if no boundary found
    """
    if not words or not full_text:
        return None

    last_boundary_pos = -1
    for i, ch in enumerate(full_text):
        if ch in SENTENCE_ENDS:
            last_boundary_pos = i

    if last_boundary_pos == -1:
        return None

    text_up_to = full_text[:last_boundary_pos + 1].strip()
    if not text_up_to:
        return None

    word_count = len(text_up_to.split())
    word_idx = min(word_count - 1, len(words) - 1)

    if word_idx < 0:
        return None

    w = words[word_idx]
    cut_time = w.end if hasattr(w, "end") else w["end"]
    return text_up_to, cut_time
