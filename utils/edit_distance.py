from pypinyin import pinyin, Style
from Levenshtein import distance
from typing import Tuple


def normalize_pinyin(pinyin_str: str) -> str:
    """
    Normalize pinyin by handling similar sounds (front/back nasal, l/n).

    Args:
        pinyin_str (str): Pinyin string to normalize

    Returns:
        str: Normalized pinyin string
    """
    # Mapping for similar sounds
    sound_map = {
        'in': 'ing',  # Front nasal to back nasal
        'en': 'eng',
        'an': 'ang',
        'un': 'ung',
        'l': 'n'  # l to n
    }

    for src, dst in sound_map.items():
        pinyin_str = pinyin_str.replace(src, dst)

    return pinyin_str


def pinyin_similarity_score(sentence1: str, sentence2: str) -> Tuple[float, int]:
    """
    Calculate similarity score and edit distance between pinyin representations of two Chinese sentences,
    accounting for similar sounds like front/back nasal and l/n.

    Args:
        sentence1 (str): First Chinese sentence
        sentence2 (str): Second Chinese sentence

    Returns:
        Tuple[float, int]: Similarity score (0-1) and edit distance
    """
    # Convert sentences to pinyin without tones
    pinyin1 = ''.join([item[0] for item in pinyin(sentence1, style=Style.NORMAL)])
    pinyin2 = ''.join([item[0] for item in pinyin(sentence2, style=Style.NORMAL)])

    # Normalize pinyin for similar sounds
    pinyin1_normalized = normalize_pinyin(pinyin1)
    pinyin2_normalized = normalize_pinyin(pinyin2)

    # Calculate edit distance
    edit_dist = distance(pinyin1_normalized, pinyin2_normalized)

    # Calculate similarity score
    max_len = max(len(pinyin1_normalized), len(pinyin2_normalized))
    if max_len == 0:
        similarity = 1.0
    else:
        similarity = max(0.0, 1.0 - (edit_dist / max_len))

    similarity_show = similarity if similarity >= 0.5 else 1 - similarity
    if similarity_show != similarity:
        len_pinyin = len(pinyin1_normalized) if len(pinyin1_normalized) > len(pinyin2_normalized) else len(pinyin2_normalized)
        len_pinyin = len_pinyin - edit_dist
    else:
        len_pinyin = edit_dist

    return similarity_show, len_pinyin


if __name__ == "__main__":
    # Example usage
    s1 = "你好世界"
    s2 = "你好地球"
    score, dist = pinyin_similarity_score(s1, s2)
    print(f"Sentences: '{s1}' and '{s2}'")
    print(f"Similarity score: {score:.4f}")
    print(f"Edit distance: {dist}")