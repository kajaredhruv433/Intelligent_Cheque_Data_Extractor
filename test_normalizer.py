
import unittest
from text_normalizer import normalize_date, normalize_amount_numeric, normalize_amount_in_words

class TestTextNormalizer(unittest.TestCase):
    
    def test_normalize_date(self):
        cases = [
            ("20-sep-2025", "20-09-2025"),
            ("20 september 2025", "20-09-2025"),
            ("20-9-2025", "20-09-2025"),
            ("2092025", "20-09-2025"),
            ("20092025", "20-09-2025"),
            ("1.1.2023", "01-01-2023"),
            ("2092025", "20-09-2025"), # 7 digit check
            # ("2209202", "22-09-2025") # This is ambiguous/broken input, verifying behavior
        ]
        for inp, expected in cases:
            res = normalize_date(inp)
            if res != expected:
                print(f"FAIL [Date]: '{inp}' -> '{res}', expected '{expected}'")
            self.assertEqual(res, expected)

    def test_normalize_amount_numeric(self):
        cases = [
            ("2300", 2300.0),
            ("2s00", 2500.0),
            ("23oo", 2300.0),
            ("z3.50", 23.50),
            ("S000", 5000.0),
            ("1,00,000", 100000.0),
            ("yo0o", 4000.0), # 'y' -> 4
        ]
        for inp, expected in cases:
            res = normalize_amount_numeric(inp)
            self.assertEqual(res, expected)

    def test_normalize_amount_in_words(self):
        cases = [
            ("two thousand three hundred only", 2300),
            ("two thousand three hundred", 2300),
            ("Five Thousand only", 5000),
            ("one lack", 100000),
            ("one lakh", 100000),
            ("one crore", 10000000),
            ("five million", 5000000),
            ("twnty fve thsand", 25000),
            ("one lack fifty thosand", 150000),
            ("Fouvthousend", 4000) # Joined words with spelling error
        ]
        for inp, expected in cases:
            res = normalize_amount_in_words(inp)
            if res != expected:
                print(f"FAIL [Words]: '{inp}' -> {res}, expected {expected}")
            self.assertEqual(res, expected)

if __name__ == '__main__':
    unittest.main()
