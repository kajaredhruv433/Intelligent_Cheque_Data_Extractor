import re
import difflib

# ===============================
# CONFIG & VOCABULARY
# ===============================
NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
}

MULTIPLIERS = {
    'hundred': 100,
    'thousand': 1000,
    'lakh': 100000,
    'lac': 100000,
    'lacs': 100000,
    'lack': 100000, # common spelling mistake
    'lacks': 100000,
    'crore': 10000000,
    'million': 1000000,
    'billion': 1000000000
}

MONTHS = {
    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06',
    'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
    'january': '01', 'february': '02', 'march': '03', 'april': '04', 'june': '06',
    'july': '07', 'august': '08', 'september': '09', 'october': '10',
    'november': '11', 'december': '12'
}

OCR_DIGIT_MAP = {
    's': '5', 'S': '5',
    'b': '8', 'B': '8',
    'o': '0', 'O': '0',
    'l': '1', 'I': '1', 'i': '1',
    'z': '2', 'Z': '2',
    'g': '9', 'q': '9',
    'y': '4', 'Y': '4',
    'A': '4'
}

def clean_ocr_text(text):
    """Basic cleaning of OCR text."""
    if not text:
        return ""
    return str(text).strip()

def split_joined_words(text):
    """
    Splits joined words like 'Fouvthousend' -> 'Fouv thousend'.
    Uses common multipliers as split points.
    """
    # Regex to find multipliers (even misspelled ones) and ensure spaces around them
    # thous... for thousand, thousend
    # hundr... for hundred
    # lakh, lac...
    pattern = r'(thous[a-z]*|hundr[a-z]*|lakh|lac|crore|million|billion)'
    text = re.sub(pattern, r' \1 ', text, flags=re.IGNORECASE)
    return text

def fuzzy_match_word(word, choices, cutoff=0.6):
    """Find closest match for a word from choices."""
    matches = difflib.get_close_matches(word.lower(), choices, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def normalize_amount_numeric(text):
    """
    Normalizes numeric amount string, handling OCR errors like 's'->5, 'b'->8.
    Returns a float or None.
    """
    text = clean_ocr_text(text)
    if not text:
        return None
    
    # 1. Replace common OCR errors
    cleaned_chars = []
    for char in text:
        if char.isdigit() or char == '.':
            cleaned_chars.append(char)
        elif char in OCR_DIGIT_MAP:
            cleaned_chars.append(OCR_DIGIT_MAP[char])
        # Ignore other characters (like commas, currency symbols, spaces)
    
    clean_str = "".join(cleaned_chars)
    
    try:
        if not clean_str:
            return None
        return float(clean_str)
    except ValueError:
        return None

def normalize_date(text):
    """
    Normalizes date to dd-mm-yyyy.
    Handles: 20-sep-2025, 20 september 2025, 20-9-2025, 2092025
    """
    text = clean_ocr_text(text).lower()
    if not text:
        return None
        
    # Standardize separators: replace . / space with -
    text = re.sub(r'[./\s]', '-', text)
    
    # Convert month names to numbers
    # Sort by length descending to avoid partial matches (e.g. 'sep' matching 'september')
    sorted_months = sorted(MONTHS.keys(), key=len, reverse=True)
    for m_name in sorted_months:
        if m_name in text:
            text = text.replace(m_name, MONTHS[m_name])
            # Don't break, maybe multiple replacements needed? No, for date usually once.
            # But checking all is safer if logic is correct.
            # But we should probably break if we found the month?
            # Actually, if we have "january", we replace it. "jan" won't be found anymore.
            # If we had "jan", we replace it.
            
    # Remove any duplicate dashes
    text = re.sub(r'-+', '-', text)
    
    # CASE 1: 8 digit contiguous number e.g. 20092025
    if text.isdigit():
        if len(text) == 8:
            return f"{text[:2]}-{text[2:4]}-{text[4:]}"
        elif len(text) == 7:
            # Ambiguous: 2092025 -> 20-09-2025 (ddmmyyyy) OR 1122025 -> 1-12-2025
            # Try to infer based on valid months
            d1, m1, y1 = text[:2], text[2:3], text[3:] # dd-m-yyyy
            d2, m2, y2 = text[:1], text[1:3], text[3:] # d-mm-yyyy
            
            # Heuristic: if m1 is '0', it's invalid month (unless ddmmyy, but here y is 4 digits)
            # 2092025: d=20, m=9, y=2025. Valid.
            # 2209202: d=22, m=0, y=9202? No.
            
            # Let's try dd-m-yyyy first
            if 1 <= int(d1) <= 31 and 1 <= int(m1) <= 9 and 1900 <= int(y1) <= 2100:
                 return f"{d1}-0{m1}-{y1}"
            
            # Try d-mm-yyyy
            if 1 <= int(d2) <= 9 and 1 <= int(m2) <= 12 and 1900 <= int(y2) <= 2100:
                return f"0{d2}-{m2}-{y2}"
        
    # CASE 2: Matches d-m-y pattern
    match = re.search(r'(\d{1,2})[-]?(\d{1,2})[-]?(\d{2,4})', text) # relaxed year to 2-4 digits
    if match:
        d, m, y = match.groups()
        # If year is 2 digits, assume 20xx? Or maybe 19xx?
        # User example: 2209202 (7 digits). Regex might catch this?
        # 2209202 -> 22, 09, 202? 
        # If y is 3 digits, it's likely a scan error.
        
        if len(y) == 3:
             # 22-09-202 -> likely 2025?
             # Heuristic: if starts with 20, append 5? Or just return as is?
             # User said: 2209202 -> 22-00-9202 was wrong.
             pass
             
        # Fix year if 2 digits
        if len(y) == 2:
            y = "20" + y
            
        return f"{int(d):02d}-{int(m):02d}-{y}"
        
    return text

def normalize_amount_in_words(text):
    """
    Converts amount in words to numeric value.
    Handles 'only', line breaks, misspellings (fuzzy match), and various units.
    """
    text = clean_ocr_text(text).lower()
    if not text:
        return None
        
    # Remove 'only' and common currency words
    words_to_remove = ['only', 'rupees', 'rs', 'rupee', '/', '-', '.', ',']
    for w in words_to_remove:
        text = text.replace(w, ' ') # Replace with space to avoid merging words
    
    # Split joined words (e.g. Fouvthousend -> Fouv thousend)
    text = split_joined_words(text)
        
    # Split into words
    words = text.split()
    
    # Process words: normalize spelling
    cleaned_words = []
    
    # Create vocab for fuzzy matching
    all_vocab = list(NUMBER_WORDS.keys()) + list(MULTIPLIERS.keys())
    
    for word in words:
        if word.isdigit():
            try:
                cleaned_words.append(int(word))
            except:
                pass
            continue
            
        # Try exact match first
        if word in NUMBER_WORDS:
            cleaned_words.append(NUMBER_WORDS[word])
        elif word in MULTIPLIERS:
            cleaned_words.append(word) # Keep multipliers as strings
        else:
            # Fuzzy match
            match = fuzzy_match_word(word, all_vocab, cutoff=0.6)
            if match:
                if match in NUMBER_WORDS:
                    cleaned_words.append(NUMBER_WORDS[match])
                else:
                    cleaned_words.append(match)
    
    # Calculate value
    current_val = 0
    final_val = 0
    
    for item in cleaned_words:
        if isinstance(item, int):
            current_val += item
        elif isinstance(item, str): # Multiplier
            mult = MULTIPLIERS[item]
            if mult >= 1000: # Thousand, Lakh, Crore, Million, Billion
                final_val += current_val * mult
                current_val = 0
            else: # Hundred
                current_val *= mult
                
    final_val += current_val
    return final_val
