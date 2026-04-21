"""
config.py – Centralized configuration for the Bahrain Budget App.
"""

import os
import re
from pathlib import Path

# ── API Keys ─────────────────────────────────────────────────────────
BOT_TOKEN   = os.getenv("BOT_TOKEN",   "8317081636:AAEnPeWorcvmkmTi0nRiKKY72MuzLrQss68")
CHAT_ID     = os.getenv("CHAT_ID",     "6895711242")
OCR_API_KEY = os.getenv("OCR_API_KEY",  "K88473266388957")
HF_API_KEY  = os.getenv("HF_API_KEY",   "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")

# ── Paths ────────────────────────────────────────────────────────────
MERCHANT_MODEL_PATH = Path("merchant_category_model.pkl")
MERCHANT_DATA_PATH  = Path("merchant_mapping.csv")
ALERT_STATE_FILE    = Path("budget_alert_state.json")

# ── Currency ─────────────────────────────────────────────────────────
GULF_CURRENCY_PATTERNS: dict[str, list[str]] = {
    "BHD": [r"\bbhd\b", r"\bbd\b", r"د\.ب", r"دينار بحريني"],
    "KWD": [r"\bkwd\b", r"د\.ك", r"دينار كويتي"],
    "SAR": [r"\bsar\b", r"\bsaudi riyal\b", r"ر\.س", r"ريال سعودي"],
    "AED": [r"\baed\b", r"\buae dirham\b", r"د\.إ", r"درهم اماراتي", r"درهم إماراتي"],
    "QAR": [r"\bqar\b", r"ر\.ق", r"ريال قطري"],
    "OMR": [r"\bomr\b", r"ر\.ع", r"ريال عماني", r"ريال عُماني"],
    "USD": [r"\busd\b", r"\$\s*\d", r"\d\s*\$"],
    "GBP": [r"\bgbp\b", r"£\s*\d", r"\d\s*£"],
    "EUR": [r"\beur\b", r"€\s*\d", r"\d\s*€"],
}

CURRENCY_TO_BHD: dict[str, float] = {
    "BHD": 1.000, "KWD": 1.230, "SAR": 0.100, "AED": 0.102,
    "QAR": 0.103, "OMR": 0.978, "USD": 0.376, "GBP": 0.480, "EUR": 0.410,
}

# ── Amount extraction ────────────────────────────────────────────────
_CUR = (
    r"BHD|BD|B\.D\.?|KWD|SAR|AED|QAR|OMR|USD|GBP|EUR|"
    r"د\.ب|د\.ك|ر\.س|د\.إ|ر\.ق|ر\.ع|دينار|ريال|درهم|"
    r"\$|£|€"
)

AMOUNT_PATTERNS = [
    re.compile(rf"(?:{_CUR})\s*([0-9]+(?:\.[0-9]{{1,3}})?)(?!\d)", re.I),
    re.compile(rf"([0-9]+(?:\.[0-9]{{1,3}})?)\s*(?:{_CUR})", re.I),
]

AMOUNT_PATTERN_BARE = re.compile(r"([0-9]+\.[0-9]{2,3})")

KEYWORDS_RE = re.compile(
    r"(purchase|spent|debit|debited|withdrawal|paid|payment|pos|txn|"
    r"transaction|transfer|charge|charged|"
    r"شراء|شراءً|سحب|دفع|تم خصم|عملية|تحويل|استخدام)",
    re.I,
)

CURRENCY_RE = re.compile(rf"({_CUR})", re.I)

MERCHANT_HINT_PATTERNS = [
    re.compile(r"\b(?:at|merchant|to)\s*[:\-]?\s*([A-Za-z0-9\u0600-\u06FF \-&_.]{2,})", re.I),
    re.compile(r"(?:POS|PURCHASE)\s*[:\-]?\s*([A-Za-z0-9\u0600-\u06FF \-&_.]{2,})", re.I),
    re.compile(r"(?:من|إلى|الى|لدى|في)\s*[:\-]?\s*([A-Za-z0-9\u0600-\u06FF \-&_.]{2,})", re.I),
]

DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})\b"),
    re.compile(r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b"),
    re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2})\b"),
]

RECEIPT_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b"),
    re.compile(r"\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[A-Za-z]{3}[\'\-]?\d{2,4}\b"),
]

# ── ALL ENGLISH categories (bilingual keyword matching) ──────────────
CATEGORY_RULES: dict[str, list[str]] = {
    "Restaurants & Cafes": [
        r"talabat", r"deliveroo", r"hungerstation", r"jahez",
        r"mcdonald", r"starbucks", r"coffee", r"burger", r"pizza",
        r"restaurant", r"cafe", r"karak", r"dunkin", r"caribou",
        r"cinnabon", r"costa", r"tim horton", r"cold stone",
        r"baskin", r"krispy", r"kfc", r"hardee", r"subway",
        r"popeye", r"domino", r"papa john", r"chili", r"applebee",
        r"nando", r"shake shack", r"jasmi",
        r"waffle", r"latte", r"cappuccino",
        r"food court", r"bakery", r"shawarma", r"falafel", r"grill",
        r"diner", r"eatery", r"kitchen", r"bistro",
        r"طلبات", r"مطعم", r"مقهى", r"قهوة", r"برجر", r"بيتزا",
        r"مخبز", r"شاورما", r"فلافل", r"مشوي",
    ],
    "Grocery": [
        r"lulu", r"carrefour", r"al jazira", r"jawad", r"alosra",
        r"ramez", r"mega\s*mart", r"midway", r"al\s*osra",
        r"geant", r"spar", r"nesto", r"al\s*muntazah",
        r"hypermarket", r"supermarket", r"grocery", r"minimarket",
        r"cold store",
        r"بقالة", r"سوبرماركت", r"هايبر",
    ],
    "Transport": [
        r"uber", r"careem", r"taxi", r"fuel", r"petrol",
        r"gas\b", r"bapco", r"parking", r"naft", r"enoc",
        r"aloola", r"al\s*oola", r"station", r"pump", r"car\s*wash",
        r"اوبر", r"أوبر", r"كريم", r"وقود", r"بنزين", r"مواقف",
        r"بترول", r"محطة",
    ],
    "Shopping": [
        r"amazon", r"noon", r"ikea", r"mall", r"shop", r"store",
        r"clothes", r"zara", r"h&m", r"namshi", r"centrepoint",
        r"max\b", r"splash", r"home\s*centre", r"home\s*box",
        r"marks.*spencer",
        r"تسوق", r"متجر", r"ملابس", r"نون", r"امازون", r"أمازون",
    ],
    "Entertainment": [
        r"cinema", r"movie", r"theatre", r"theater", r"vox",
        r"wadi\s*cinema", r"seef\s*cinema", r"imax",
        r"game", r"playstation", r"xbox", r"steam",
        r"gym", r"fitness", r"club",
        r"سينما", r"فيلم", r"لعبة", r"نادي", r"رياضة",
    ],
    "Bills & Utilities": [
        r"ewa", r"batelco", r"stc", r"zain", r"bill", r"utility",
        r"internet", r"electric", r"water", r"subscription",
        r"netflix", r"spotify", r"apple", r"google\s*play",
        r"فاتورة", r"كهرباء", r"ماء", r"انترنت", r"اشتراك",
    ],
    "Travel": [
        r"hotel", r"booking", r"airways", r"flight", r"travel",
        r"trip", r"holiday", r"airbnb", r"agoda", r"expedia",
        r"gulf\s*air", r"emirates", r"qatar\s*airways",
        r"فندق", r"سفر", r"رحلة", r"طيران",
    ],
    "Cash": [r"atm", r"cash withdrawal", r"withdrawal", r"سحب نقدي", r"سحب"],
    "Health & Personal": [
        r"pharmacy", r"sephora", r"salon", r"barber", r"spa",
        r"makeup", r"skincare", r"hospital", r"clinic", r"doctor",
        r"dental", r"optical",
        r"صيدلية", r"صالون", r"حلاقة", r"عناية", r"مستشفى",
    ],
    "Transfers": [
        r"transfer", r"iban", r"beneficiary", r"remit",
        r"تحويل", r"مستفيد", r"ايبان", r"آيبان",
    ],
    "Other": [],
}

KNOWN_MERCHANTS: dict[str, str] = {
    "mcdonald's": "Restaurants & Cafes", "mcdonalds": "Restaurants & Cafes",
    "starbucks": "Restaurants & Cafes",  "talabat": "Restaurants & Cafes",
    "deliveroo": "Restaurants & Cafes",  "costa": "Restaurants & Cafes",
    "caribou": "Restaurants & Cafes",    "dunkin": "Restaurants & Cafes",
    "kfc": "Restaurants & Cafes",        "jasmi's": "Restaurants & Cafes",
    "lulu hypermarket": "Grocery",       "lulu": "Grocery",
    "carrefour": "Grocery",              "ramez": "Grocery",
    "al jazira": "Grocery",              "jawad": "Grocery",
    "alosra": "Grocery",
    "ikea": "Shopping",
    "careem": "Transport",               "uber": "Transport",
    "bapco": "Transport",                "aloola": "Transport",
    "ewa": "Bills & Utilities",          "batelco": "Bills & Utilities",
    "stc": "Bills & Utilities",          "zain": "Bills & Utilities",
    "netflix": "Bills & Utilities",      "spotify": "Bills & Utilities",
    "vox cinema": "Entertainment",       "wadi cinema": "Entertainment",
    # Arabic names
    "ستاربكس": "Restaurants & Cafes",    "طلبات": "Restaurants & Cafes",
    "ماكدونالدز": "Restaurants & Cafes", "كوستا": "Restaurants & Cafes",
    "كريم": "Transport",                 "أوبر": "Transport",
    "اوبر": "Transport",                 "لولو": "Grocery",
    "كارفور": "Grocery",                  "رامز": "Grocery",
    "نتفلكس": "Bills & Utilities",
}

ML_LABEL_MAP: dict[str, str] = {
    "Food & Beverages": "Restaurants & Cafes",
    "Shopping/Grocery": "Grocery",
    "General": "Bills & Utilities",
    "Transfer": "Transfers",
    "Cash Withdrawal": "Cash",
    "OTP": "Other",
    "Retail": "Shopping",
    "Services/Trading": "Bills & Utilities",
    "Health/Pharmacy": "Health & Personal",
    "Automotive": "Transport",
}

SAMPLE_SMS = """\
Batelco: Purchase of BHD 3.200 at STARBUCKS on 16/02/2026. Avl Bal: BHD 120.000
Bank SMS: POS purchase BHD 12.750 at TALABAT on 15/02/2026 Ref 9912
Card Alert: You spent BHD 6.500 at CAREEM 2026-02-14
Debit Alert: BHD 1.000 at BAPCO FUEL on 13/02/2026
Transaction: Paid BHD 25.000 to EWA BILL on 12/02/2026
Purchase: BHD 9.900 at NETFLIX on 11/02/2026
ATM Withdrawal: BHD 20.000 on 10/02/2026
POS: Purchase BHD 18.250 at IKEA on 09/02/2026
Transfer: BHD 50.000 to Beneficiary IBAN BHxx on 08/02/2026

تم خصم 3.500 د.ب من ستاربكس بتاريخ 21/04/2026
عملية شراء بمبلغ 12.750 د.ب من طلبات
تم دفع 5.000 د.ب إلى كريم
سحب نقدي 20.000 د.ب
"""

RECEIPT_SKIP_KEYWORDS = [
    "vat", "tax", "total", "subtotal", "sub total", "grand total",
    "payment", "change", "cash", "credit", "debit", "card",
    "auth", "approval", "ref", "reference", "invoice", "receipt",
    "thank", "welcome", "visit again", "balance", "net total",
    "net ttl", "vat ttl", "amount due", "terminal", "merchant id",
    "vat number", "tel", "phone", "road", "building", "block",
    "www", ".com", "take away",
]