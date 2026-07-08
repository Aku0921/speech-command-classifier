import random
random.seed(42)
def lowercase_text(
    sentence: str,
) -> str:
    """
    Convert the sentence to lowercase.
    """
    return sentence.lower()
def add_filler_word(
    sentence: str,
) -> str:
    """
    Add a filler word at the beginning of the sentence.
    """
    fillers = [
        "uh",
        "um",
        "hey",
        "please",
    ]
    filler = random.choice(fillers)
    return f"{filler} {sentence}"
def apply_abbreviation(
    sentence: str,
) -> str:
    """
    Replace common phrases with abbreviations.
    """
    abbreviations = {
        "do not disturb": "DND",
    }
    lower_sentence = sentence.lower()
    for phrase, short in abbreviations.items():
        if phrase in lower_sentence:
            return lower_sentence.replace(
                phrase,
                short,
            )
    return sentence
def remove_small_words(
    sentence: str,
) -> str:
    """
    Remove common small words from the sentence.
    """
    small_words = {
        "the",
        "a",
        "an",
    }
    words = sentence.split()
    filtered_words = [
        word
        for word in words
        if word.lower() not in small_words
    ]
    return " ".join(filtered_words)
def introduce_spelling_error(
    sentence: str,
) -> str:
    """
    Introduce a small spelling variation.
    """
    spelling_map = {
        "music": "musick",
        "call": "cal",
        "volume": "volum",
    }
    words = sentence.split()
    for index, word in enumerate(words):
        key = word.lower()
        if key in spelling_map:
            words[index] = spelling_map[key]
            break
    return " ".join(words)
def generate_noisy_sentence(
    sentence: str,
) -> str:
    """
    Generate a noisy version of a sentence.
    """
    noise_functions = [
        lowercase_text,
        add_filler_word,
        apply_abbreviation,
        remove_small_words,
        introduce_spelling_error,
    ]
    original = sentence
    for _ in range(5):
        noise_function = random.choice(noise_functions)
        noisy_sentence = noise_function(sentence)
        if noisy_sentence != original:
            return noisy_sentence
    return sentence