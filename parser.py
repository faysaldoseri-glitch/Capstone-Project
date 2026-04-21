"""
parser.py – Transaction parsing engine for the Bahrain Budget App.
All categories output in ENGLISH. Input accepts Arabic + English.
"""

from __future__ import annotations

import re
from datetime import datetime
from collections import Counter

import pandas as pd
import joblib

from config import (
    AMOUNT_PATTERNS, AMOUNT_PATTERN_BARE,
    KEYWORDS_RE, CURRENCY_RE,
    MERCHANT_HINT_PATTERNS,
    DATE_PATTERNS, RECEIPT_DATE_PATTERNS,
    CATEGORY_RULES, KNOWN_MERCHANTS, ML_LABEL_MAP,
    GULF_CURRENCY_PATTERNS, CURRENCY_TO_BHD,
    MERCHANT_MODEL_PATH, MERCHANT_DATA_PATH,
    RECEIPT_SKIP_KEYWORDS,
)

# ── Arabic digit normalisation ───────────────────────────────────────
_DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_arabic(text: str) -> str:
    return str(text or "").translate(_DIGIT_TABLE)


# ── Currency ─────────────────────────────────────────────────────────
def detect_currency(text: str) -> str:
    norm = normalize_arabic(text).lower()
    for code, patterns in GULF_CURRENCY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, norm, re.I):
                return code
    return "BHD"


def to_bhd(amount: float | None, currency: str = "BHD") -> float | None:
    if amount is None:
        return None
    rate = CURRENCY_TO_BHD.get((currency or "BHD").upper(), 1.0)
    return round(float(amount) * rate, 3)


# ── Amount extraction ────────────────────────────────────────────────
def extract_amount(text: str) -> float | None:
    norm = normalize_arabic(text)
    for pat in AMOUNT_PATTERNS:
        m = pat.search(norm)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                continue

    # Bare decimal fallback for short lines
    if KEYWORDS_RE.search(norm) or len(norm.strip()) < 80:
        m = AMOUNT_PATTERN_BARE.search(norm)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                pass
    return None


# ── Date extraction ──────────────────────────────────────────────────
def extract_date(text: str) -> datetime.date:
    from dateutil import parser as dateparser
    norm = normalize_arabic(text).strip()
    for pat in DATE_PATTERNS:
        m = pat.search(norm)
        if m:
            try:
                dt = dateparser.parse(m.group(1), dayfirst=True)
                if dt and datetime.today().year - 1 <= dt.year <= datetime.today().year + 1:
                    return dt.date()
            except Exception:
                pass
    return datetime.today().date()


def extract_receipt_date(text: str) -> datetime.date | None:
    from dateutil import parser as dateparser
    norm = normalize_arabic(text).strip()
    for pat in RECEIPT_DATE_PATTERNS:
        m = pat.search(norm)
        if m:
            try:
                dt = dateparser.parse(m.group(0), dayfirst=True)
                if dt:
                    return dt.date()
            except Exception:
                pass
    return None


# ── Merchant extraction ──────────────────────────────────────────────
def extract_merchant(text: str) -> str:
    for pat in MERCHANT_HINT_PATTERNS:
        m = pat.search(text)
        if m:
            merchant = m.group(1).strip()
            merchant = re.sub(
                r"(?:on|date|ref|auth|bal|available|بتاريخ|المرجع|رصيد).*",
                "", merchant, flags=re.I,
            ).strip()
            # Strip trailing dates like "2026-02-14" or "16/02/2026"
            merchant = re.sub(r"\s*\d{4}[\-\/]\d{1,2}[\-\/]\d{1,2}\s*$", "", merchant).strip()
            merchant = re.sub(r"\s*\d{1,2}[\-\/]\d{1,2}[\-\/]\d{2,4}\s*$", "", merchant).strip()
            if len(merchant) >= 2:
                return merchant[:40]
    return re.sub(r"\s+", " ", text).strip()[:40]


# ── Categorisation (all English output) ──────────────────────────────
_merchant_model = None
_merchant_df: pd.DataFrame | None = None
_resources_loaded = False


def _load_resources():
    global _merchant_model, _merchant_df, _resources_loaded
    if _resources_loaded:
        return
    if MERCHANT_MODEL_PATH.exists():
        try:
            _merchant_model = joblib.load(MERCHANT_MODEL_PATH)
        except Exception:
            _merchant_model = None
    if MERCHANT_DATA_PATH.exists():
        try:
            df = pd.read_csv(MERCHANT_DATA_PATH)
            df["Brand Name"] = df["Brand Name"].astype(str).str.lower().str.strip()
            df["POS Terminal Name"] = df["POS Terminal Name"].astype(str).str.lower().str.strip()
            _merchant_df = df
        except Exception:
            _merchant_df = None
    _resources_loaded = True


_LOCATION_NOISE = {
    "manama", "riffa", "muharraq", "isa town", "hamad", "juffair",
    "seef", "saar", "budaiya", "bahrain", "station", "t1", "t2", "t3",
}


def _match_dataset(merchant: str) -> str | None:
    _load_resources()
    if _merchant_df is None or not merchant:
        return None
    m = merchant.lower().strip()

    hits = _merchant_df[_merchant_df["POS Terminal Name"].str.contains(m, na=False, regex=False)]
    if not hits.empty:
        return ML_LABEL_MAP.get(hits.iloc[0]["Category"], hits.iloc[0]["Category"])
    hits = _merchant_df[_merchant_df["Brand Name"].str.contains(m, na=False, regex=False)]
    if not hits.empty:
        return ML_LABEL_MAP.get(hits.iloc[0]["Category"], hits.iloc[0]["Category"])

    for w in m.split():
        if len(w) < 4 or w in _LOCATION_NOISE:
            continue
        hits = _merchant_df[_merchant_df["Brand Name"].str.contains(w, na=False, regex=False)]
        if not hits.empty:
            return ML_LABEL_MAP.get(hits.iloc[0]["Category"], hits.iloc[0]["Category"])
    return None


def _match_known(merchant: str) -> str | None:
    m = merchant.lower().strip()
    if m in KNOWN_MERCHANTS:
        return KNOWN_MERCHANTS[m]
    for name, cat in KNOWN_MERCHANTS.items():
        if name in m or m in name:
            return cat
    return None


def _match_rules(text: str) -> str | None:
    t = normalize_arabic(text).lower()
    for cat, patterns in CATEGORY_RULES.items():
        if not patterns:
            continue
        for p in patterns:
            if re.search(p, t):
                return cat
    return None


def _match_ml(merchant: str) -> str | None:
    _load_resources()
    if _merchant_model is None or not merchant or merchant == "unknown":
        return None
    try:
        label = _merchant_model.predict([merchant])[0]
        return ML_LABEL_MAP.get(label, "Other")
    except Exception:
        return None


def categorize(merchant: str, full_text: str = "") -> str:
    text_norm = normalize_arabic((full_text or "").lower())

    if "otp" in text_norm or "one time password" in text_norm:
        return "Other"
    if any(kw in text_norm for kw in ("transfer", "beneficiary", "تحويل")):
        return "Transfers"
    if any(kw in text_norm for kw in ("cash withdrawal", "withdrawal", "atm", "سحب")):
        return "Cash"

    merchant_norm = normalize_arabic((merchant or "").lower().strip())
    combined = f"{merchant} {full_text}"

    return (
        _match_dataset(merchant_norm)
        or _match_known(merchant_norm)
        or _match_rules(combined)
        or _match_ml(merchant_norm)
        or "Other"
    )


# ── Word-to-number conversion ────────────────────────────────────────
_WORD_NUMBERS = {
    # English
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
    # Arabic
    "واحد": 1, "اثنين": 2, "ثلاثة": 3, "ثلاث": 3, "اربعة": 4, "أربعة": 4,
    "خمسة": 5, "خمس": 5, "ستة": 6, "ست": 6, "سبعة": 7, "سبع": 7,
    "ثمانية": 8, "ثمان": 8, "تسعة": 9, "تسع": 9, "عشرة": 10, "عشر": 10,
    "عشرين": 20, "ثلاثين": 30, "اربعين": 40, "أربعين": 40,
    "خمسين": 50, "ستين": 60, "سبعين": 70, "ثمانين": 80, "تسعين": 90,
    "مية": 100, "مئة": 100,
    # Common spoken shortcuts
    "half": 0.5, "quarter": 0.25, "نص": 0.5, "ربع": 0.25,
}


def _words_to_number(text: str) -> float | None:
    """
    Convert spoken number words to a float.
    Handles: 'five', 'twenty three', 'خمسة', 'عشرين', etc.
    """
    words = text.lower().strip().split()
    total = 0.0
    found_any = False

    for w in words:
        w_clean = w.strip(".,!?")
        if w_clean in _WORD_NUMBERS:
            total += _WORD_NUMBERS[w_clean]
            found_any = True
        elif found_any:
            break  # stop at first non-number word after finding numbers

    return total if found_any and total > 0 else None


def _convert_spoken_numbers(text: str) -> str:
    """
    Replace spoken number words in text with digits.
    'Five BD at Starbucks' → '5 BD at Starbucks'
    """
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        w_clean = words[i].lower().strip(".,!?")
        if w_clean in _WORD_NUMBERS:
            # Collect consecutive number words
            num_total = 0.0
            while i < len(words) and words[i].lower().strip(".,!?") in _WORD_NUMBERS:
                num_total += _WORD_NUMBERS[words[i].lower().strip(".,!?")]
                i += 1
            # Format: use integer if whole, else decimal
            if num_total == int(num_total):
                result.append(str(int(num_total)))
            else:
                result.append(f"{num_total:.3f}")
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)


# ── Item name extraction ─────────────────────────────────────────────
def extract_item_name(merchant: str, text: str) -> str:
    """
    Best-effort item name. For SMS: use merchant name.
    For voice/natural text: try to extract 'at X' or 'from X'.
    """
    if merchant and merchant.lower() not in ("unknown", ""):
        return merchant
    return text[:50] if text else ""


# ── Voice text parser (natural language) ─────────────────────────────
def parse_voice_text(text: str, date_override=None) -> pd.DataFrame:
    """
    Parse natural language like 'I spent 5 BHD at Starbucks',
    'Five BD at Starbucks', or 'خمسة دينار في ستاربكس' into a transaction.
    Converts spoken number words to digits first.
    """
    if not text or not text.strip():
        return pd.DataFrame()

    # Convert spoken numbers: "Five BD" → "5 BD"
    text_converted = _convert_spoken_numbers(text)
    norm = normalize_arabic(text_converted)
    amount = extract_amount(norm)

    # If still no amount, try bare number
    if amount is None:
        m = re.search(r"(\d+(?:\.\d+)?)", norm)
        if m:
            amount = float(m.group(1))

    if amount is None:
        return pd.DataFrame()

    currency = detect_currency(text)
    merchant = extract_merchant(text)
    dt = date_override or extract_date(norm)

    row = {
        "date": dt,
        "amount_bhd": to_bhd(amount, currency),
        "merchant": merchant,
        "category": categorize(merchant, text),
        "raw_sms": text,
        "item_name": extract_item_name(merchant, text),
        "qty": 1,
        "source_type": "voice",
        "store_name": merchant,
        "currency": currency,
    }

    df = pd.DataFrame([row])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── SMS block parser ─────────────────────────────────────────────────
def parse_sms_block(sms_block: str) -> pd.DataFrame:
    rows: list[dict] = []
    for line in (sms_block or "").splitlines():
        line = line.strip()
        if not line:
            continue

        norm = normalize_arabic(line)
        amount = extract_amount(norm)
        if amount is None:
            continue

        has_keyword = bool(KEYWORDS_RE.search(norm))
        has_currency = bool(CURRENCY_RE.search(norm))
        if not (has_keyword or has_currency):
            continue

        merchant = extract_merchant(line)
        currency = detect_currency(line)

        rows.append({
            "date": extract_date(norm),
            "amount_bhd": to_bhd(amount, currency),
            "merchant": merchant,
            "category": categorize(merchant, line),
            "raw_sms": line,
            "item_name": extract_item_name(merchant, line),
            "qty": 1,
            "source_type": "sms",
            "store_name": merchant,
            "currency": currency,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=False)
    return df


# ── Receipt helpers ──────────────────────────────────────────────────
_KNOWN_STORE_NAMES = [
    "starbucks", "mcdonald", "costa", "caribou", "dunkin",
    "lulu", "carrefour", "al jazira", "jawad", "alosra",
    "ikea", "spar", "cold stone", "cinnabon", "tim hortons",
    "ramez", "kfc", "burger king", "subway",
]

_EXTRA_SKIP = [
    "chk", "check", "natalia", "cashier", "server", "table",
    "order", "dine in", "eat in", "drive thru", "delivery",
]


def _is_skip_line(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    if any(k in low for k in RECEIPT_SKIP_KEYWORDS):
        return True
    if any(k in low for k in _EXTRA_SKIP):
        return True
    if re.match(r"^[\d\s\-:\/]+$", low):
        return True
    return False


def _is_price_only(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", text.strip())
    return bool(re.match(r"^[0-9]+\.[0-9]{2,3}$", cleaned))


def _clean_price(text: str) -> float | None:
    cleaned = re.sub(r"\s+", "", text.strip())
    m = re.match(r"^([0-9]+\.[0-9]{2,3})$", cleaned)
    return float(m.group(1)) if m else None


def _is_addon_line(text: str) -> bool:
    low = text.lower().strip()
    return bool(re.match(r"^\d+p\b", low)) or "cup charge" in low


def _is_item_line(text: str) -> bool:
    text = text.strip()
    if not text or len(text) < 3:
        return False
    if re.match(r"^[1-9I]\s+[A-Za-z]", text):
        return True
    if re.match(r"^\d+[Pp]\s", text):
        return True
    return False


def _detect_merchant(lines: list[str]) -> str:
    for ln in lines[:10]:
        low = ln.lower()
        for brand in _KNOWN_STORE_NAMES:
            if brand in low:
                return ln.strip()
    for ln in lines[:8]:
        if _is_skip_line(ln):
            continue
        if re.search(r"\d+\.\d{2,3}$", ln):
            continue
        if 3 <= len(ln.strip()) <= 60:
            return ln.strip()
    return "Unknown"


def _extract_items_and_prices(lines):
    complete_items = []
    orphan_names = []
    pre_item_prices = []
    post_item_prices = []
    seen_any_item = False

    complete_re = re.compile(r"^(?:([1-9I])\s+)?(.+?)\s+([0-9]+\.[0-9]{2,3})$")

    for ln in lines:
        clean = re.sub(r"\s+", " ", ln).strip()
        if not clean or _is_skip_line(clean):
            continue

        m = complete_re.match(clean)
        if m:
            qty_s = m.group(1)
            qty = int(qty_s) if qty_s and qty_s.isdigit() else 1
            name = m.group(2).strip()
            price = float(m.group(3))
            if len(name) >= 2 and not any(k in name.lower() for k in RECEIPT_SKIP_KEYWORDS):
                seen_any_item = True
                complete_items.append({"name": name, "qty": qty, "price": price, "is_addon": _is_addon_line(name)})
            continue

        if _is_price_only(clean):
            p = _clean_price(clean)
            if p is not None:
                (post_item_prices if seen_any_item else pre_item_prices).append(p)
            continue

        if _is_item_line(clean):
            seen_any_item = True
            m2 = re.match(r"^([1-9I])\s+(.+)$", clean)
            if m2:
                qty = int(m2.group(1)) if m2.group(1).isdigit() else 1
                name = m2.group(2).strip()
            else:
                qty, name = 1, clean
            if len(name) >= 2 and not any(k in name.lower() for k in RECEIPT_SKIP_KEYWORDS):
                orphan_names.append({"name": name, "qty": qty, "is_addon": _is_addon_line(name)})

    def _filter_totals(prices):
        counts = Counter(prices)
        return [p for p in prices if counts[p] < 3]

    def _remove_summary(prices, all_prices):
        if not prices or len(all_prices) < 3:
            return prices
        median = sorted(all_prices)[len(all_prices) // 2]
        return [p for p in prices if p <= median * 4]

    pre_clean = _filter_totals(pre_item_prices)
    post_clean = _remove_summary(_filter_totals(post_item_prices), _filter_totals(pre_item_prices + post_item_prices))

    available = pre_clean + post_clean
    paired = list(complete_items)

    for i, item in enumerate(orphan_names):
        if i < len(available):
            paired.append({**item, "price": available[i]})

    return paired, len(orphan_names) > 0


def parse_receipt(ocr_text: str):
    raw_lines = [re.sub(r"\s+", " ", ln).strip() for ln in (ocr_text or "").splitlines() if ln.strip()]
    if not raw_lines:
        return pd.DataFrame(), "high"

    currency = detect_currency(ocr_text)
    receipt_date = datetime.today().date()
    merchant = _detect_merchant(raw_lines)

    for ln in raw_lines:
        d = extract_receipt_date(ln)
        if d:
            receipt_date = d
            break

    paired, had_orphans = _extract_items_and_prices(raw_lines)
    confidence = "medium" if had_orphans else "high"

    rows = []
    for item in paired:
        amount_bhd = to_bhd(item["price"], currency)
        if item.get("is_addon") and rows:
            rows[-1]["amount_bhd"] = round((rows[-1]["amount_bhd"] or 0) + (amount_bhd or 0), 3)
            rows[-1]["raw_sms"] += f" + {item['name']}"
            continue
        rows.append({
            "date": pd.to_datetime(receipt_date),
            "amount_bhd": amount_bhd,
            "merchant": merchant,
            "category": categorize(merchant, item["name"]),
            "raw_sms": f"{item['qty']} {item['name']} {item['price']:.2f}",
            "item_name": item["name"],
            "qty": item["qty"],
            "source_type": "receipt",
            "store_name": merchant,
            "currency": currency,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date", ascending=False)
    return df, confidence