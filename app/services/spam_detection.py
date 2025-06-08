import json
import os
import re
import time
import uuid
from collections import Counter
from typing import Any, Dict, List, Literal

import joblib
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

# Path ke artifacts
slang_file_path = "app/dict/merged_slang_dict.json"
DUMP_DIR = "app/models/spam_detection_artifacts"
PIPELINE_PATH = "app/models/spam_detection_artifacts/feature_pipeline.pkl"
NB_MODEL_PATH = "app/models/spam_detection_artifacts/naive_bayes_model.pkl"
SVM_MODEL_PATH = "app/models/spam_detection_artifacts/svm_model.pkl"
REVERSE_MAPPING_PATH = "app/models/spam_detection_artifacts/reverse_label_mapping.pkl"
TFIDF_PATH = "app/models/spam_detection_artifacts/tfidf_vectorizer.pkl"
STAT_SCALER_PATH = "app/models/spam_detection_artifacts/stat_scaler.pkl"
STAT_FEATURE_NAMES_PATH = "app/models/spam_detection_artifacts/stat_feature_names.pkl"


# --- Regex & Constants for StatisticalFeatureExtractor --
URL_RE = re.compile(
    r"""
    (?i)\b
    (?:https?://|ftp://|www\.)
    (?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+
    [a-z]{2,63}
    (?::\d+)?
    (?:/[^\s()<>]+|\([^\s()<>]+\))*
    """,
    re.VERBOSE | re.IGNORECASE,
)
PHONE_RE = re.compile(r"(08\d{8,11}|\+62\d{8,11}|0\d{9,12})")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
CHAR_REP_RE = re.compile(r"(.)\1{4,}")
EMOJI_REP_RE = re.compile(r"([\U00010000-\U0010ffff\u2600-\u27BF])\1{4,}", re.UNICODE)
CON_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]", re.IGNORECASE)
VOW_RE = re.compile(r"[aeiou]", re.IGNORECASE)
PROMO_RE = re.compile(r"(?:kode|promo|referral)\s+\w{4,}", re.IGNORECASE)
# Word Sets
COMPARISON_WORDS = {
    "dibanding",
    "daripada",
    "ketimbang",
    "lebih",
    "kurang",
    "dari",
    "mending",
    "mirip",
    "beda",
    "berbeda",
    "seperti",
    "vs",
    "versus",
    "dibandingkan",
    "setara",
    "kalah",
    "unggul",
    "bagusan",
    "tidak seperti",
    "sama",
    "persis",
    "alternatif",
    "kompetitor",
    "saingan",
    "sejenis",
}
QUESTION_WORDS = {
    "apa",
    "bagaimana",
    "kenapa",
    "mengapa",
    "gimana",
    "apakah",
    "siapa",
    "dimana",
    "kapan",
    "berapa",
    "kok",
    "gmnmasa",
    "beneran",
    "serius",
    "tolong",
    "bantu",
    "mohon",
    "ada",
    "punya",
    "bolehkah",
    "kapanlagi",
    "caranya",
    "solusinya",
    "tipsnya",
    "triknya",
    "tutorialnya",
    "panduannya",
    "artinya",
    "maksudnya",
    "fungsinya",
    "gunanya",
    "rekomendasi",
    "saran",
    "info",
    "tanya",
    "nanya",
    "bertanya",
    "kah",
    "tah",
    "siapakah",
    "kapankah",
    "dimanakah",
    "bagaimanakah",
    "berapakah",
    "bisakah",
    "bolehkah",
    "dapatkah",
    "mungkinkah",
    "haruskah",
    "perlukah",
    "adakah",
    "bilamanakah",
    "seberapakah",
}
OFF_TOPIC_KEYS = {
    # Politik, Berita, Isu Sosial
    "politik",
    "pemilu",
    "pilpres",
    "pilkada",
    "partai",
    "capres",
    "cawapres",
    "berita",
    "viral",
    "trending",
    "korupsi",
    "hukum",
    "pemerintah",
    "dpr",
    "harga",
    "bbm",
    "inflasi",
    "ekonomi",
    "saham",
    "crypto",
    "bitcoin",
    "nft",
    "cuaca",
    "hujan",
    "panas",
    "banjir",
    "gempa",
    # Olahraga
    "sepakbola",
    "bola",
    "liga",
    "timnas",
    "persib",
    "persija",
    "mu",
    "madrid",
    "motogp",
    "f1",
    "badminton",
    "basket",
    # Sapaan, Meta-comment, Informalitas
    "pertamax",
    "halo",
    "admin",
    "hai",
    "pagi",
    "siang",
    "sore",
    "malam",
    "assalamualaikum",
    "permisi",
    "numpang",
    "nyimak",
    "komen",
    "tes",
    "test",
    "jejak",
    "salam",
    "kenal",
    "gan",
    "min",
    "sis",
    "bro",
    "kak",
    "om",
    "tan",
    "sundul",
    "wkwk",
    "haha",
    "xixi",
    "hehe",
    "lol",
    "mantap",
    "ok",
    "oke",
    "sip",
    # Aplikasi/Layanan Lain (Contoh Populer)
    "gojek",
    "grab",
    "shopee",
    "tokopedia",
    "lazada",
    "tiktok",
    "instagram",
    "ig",
    "wa",
    "whatsapp",
    "facebook",
    "fb",
    "twitter",
    "x",
    "youtube",
    "yt",
    "netflix",
    "spotify",
    "dana",
    "ovo",
    "gopay",
    "pinjol",
    "ojol",
    # Kebutuhan/Aktivitas Pribadi/Umum
    "loker",
    "lowongan",
    "kerja",
    "resep",
    "masak",
    "makan",
    "minum",
    "film",
    "lagu",
    "musik",
    "jalan",
    "macet",
    "rumah",
    "sekolah",
    "kuliah",
    "kampus",
    "tugas",
    "skripsi",
    "curhat",
    "jual",
    "beli",
    "butuh",
    "cari",
    "sewa",
    "liburan",
    "traveling",
    "nonton",
    "main",  # 'game' bisa relevan, tapi sering juga off-topic
    "tidur",
    "mandi",
    "kucing",
    "anjing",  # Contoh hewan peliharaan
}
APP_TERMS = {
    # Istilah Umum Aplikasi/Software
    "aplikasi",
    "app",
    "software",
    "program",
    "platform",
    "tools",
    "utilitas",
    "sistem",
    "layanan",
    "produk",
    # Fitur & Fungsi
    "fitur",
    "fungsi",
    "menu",
    "opsi",
    "tombol",
    "navigasi",
    "pengaturan",
    "setting",
    "konfigurasi",
    "tool",
    "widget",
    "plugin",
    "ekstensi",
    "api",
    "integrasi",
    "kompatibilitas",
    "kustomisasi",
    "template",
    "filter",
    "efek",
    "preset",
    "tema",
    "mode",
    "darkmode",
    "lightmode",
    "offline",
    "online",
    "pencarian",
    "search",
    "sortir",
    "filterisasi",
    "notifikasi",
    "pemberitahuan",
    "alert",
    "reminder",
    "ai",
    "kecerdasan buatan",
    "otomatis",
    "manual",
    # Update & Versi
    "update",
    "upgrade",
    "versi",
    "rilis",
    "build",
    "patch",
    "baru",
    "lama",
    "terbaru",
    "sebelumnya",
    "beta",
    "alpha",
    "pro",
    "lite",
    "premium",
    "gratis",
    "freemium",
    "berbayar",
    "trial",
    "percobaan",
    # Antarmuka & Pengalaman Pengguna
    "ui",
    "ux",
    "tampilan",
    "desain",
    "interface",
    "antarmuka",
    "layout",
    "grafis",
    "visual",
    "animasi",
    "transisi",
    "intuitif",
    "user-friendly",
    "mudah",
    "sulit",
    "ribet",
    "simpel",
    "responsif",
    "cepat",
    "lambat",
    "lag",
    "nge-lag",
    "smooth",
    "patah-patah",
    # Akun & Keamanan
    "login",
    "logout",
    "signin",
    "signup",
    "daftar",
    "akun",
    "profil",
    "user",
    "pengguna",
    "password",
    "sandi",
    "otp",
    "verifikasi",
    "autentikasi",
    "keamanan",
    "privasi",
    "data",
    "enkripsi",
    "izin",
    "permission",
    # Masalah Teknis
    "bug",
    "error",
    "crash",
    "force close",
    "fc",
    "hang",
    "freeze",
    "macet",
    "not responding",
    "masalah",
    "kendala",
    "gangguan",
    "problem",
    "issue",
    "glitch",
    "corrupt",
    "kompatibel",
    "stabil",
    "tidak stabil",
    # Proses & Operasi
    "ekspor",
    "impor",
    "simpan",
    "save",
    "buka",
    "open",
    "tutup",
    "close",
    "install",
    "uninstall",
    "download",
    "upload",
    "render",
    "proses",
    "loading",
    "backup",
    "restore",
    "sinkronisasi",
    "sync",
    "edit",
    "hapus",
    "delete",
    "buat",
    "create",
    "copy",
    "paste",
    "cut",
    "undo",
    "redo",
    "scan",
    "print",
    # Perangkat & OS
    "device",
    "perangkat",
    "hp",
    "ponsel",
    "smartphone",
    "tablet",
    "pc",
    "laptop",
    "komputer",
    "android",
    "ios",
    "windows",
    "macos",
    "web",
    "browser",
    # Konten & Data
    "file",
    "dokumen",
    "gambar",
    "foto",
    "video",
    "audio",
    "musik",
    "teks",
    "konten",
    "data",
    "database",
    "cloud",
    "storage",
    "memori",
    "ram",
    "cpu",
    "baterai",
    "kuota",
    "internet",
    "wifi",
    "jaringan",
    # Langganan & Pembayaran
    "langganan",
    "subscribe",
    "subscription",
    "harga",
    "biaya",
    "bayar",
    "pembayaran",
    "transaksi",
    "iklan",
    "ads",
    "watermark",
    "lisensi",
    # Game Spesifik (jika relevan)
    "game",
    "level",
    "karakter",
    "item",
    "skin",
    "map",
    "server",
    "player",
    "multiplayer",
    "singleplayer",
    "online",
    "offline",
    "mabar",
    "rank",
    "cheat",
    "mod",
    "joki",
    "topup",
    "diamond",
    "coin",
    "gold",
    "uc",  # Contoh mata uang game
}
HYPERBOLIC_WORDS = {
    "terbaik",
    "tercanggih",
    "paling",
    "diskon",
    "gratis",
    "hebat",
    "luar",
    "biasa",
    "nomor",
    "tercepat",
    "termurah",
    "terhebat",
    "fantastis",
    "ultimate",
    "sempurna",
    "revolusioner",
    "terkeren",
    "terpopuler",
    "terpercaya",
    "sekali",
    "amat",
    "benar-benar",
    "sungguh",
    "#1",
    "juara",
    "dewa",
    "spektakuler",
    "menakjubkan",
    "mengagumkan",
    "istimewa",
    "100%",
    "hancur",
    "terburuk",
    "menyesatkan",
    "ajaib",
    "super",
    "mega",
    "gila",
    "gokil",
    "edannya",
    "pecah",
    "markotop",
    "nendang",
    "drastis",
    "maksimal",
    "memukau",
    "mengesankan",
    "wow",
    "terpukau",
    "menyedihkan",
    "takjub",
    "mengerikan",
    "menjijikkan",
    "memuakkan",
    "mencekam",
    "memalukan",
    "membosankan",
    "fenomenal",
    "gemilang",
    "sensasional",
    "menggetarkan",
    "merinding",
    "mematikan",
    "meresahkan",
    "mengkhawatirkan",
    "menyesakkan",
}
IMPERATIVE_WORDS = {
    "beli",
    "klik",
    "download",
    "install",
    "pasang",
    "unduh",
    "dapatkan",
    "ambil",
    "klaim",
    "claim",
    "pesan",
    "order",
    "sewa",
    "subscribe",
    "langganan",
    "ikuti",
    "follow",
    "join",
    "gabung",
    "daftar",
    "registrasi",
    "masuk",
    "login",
    "kunjungi",
    "cek",
    "lihat",
    "tonton",
    "mainkan",
    "main",
    "coba",
    "gunakan",
    "pakai",
    "hubungi",
    "kontak",
    "wa",
    "telepon",
    "dm",
    "chat",
    "kirim",
    "masukkan",
    "input",
    "isi",
    "share",
    "bagikan",
    "like",
    "suka",
    "vote",
    "pilih",
    "upgrade",
    "update",
    "perbarui",
    "simpan",
    "save",
    "verifikasi",
    "konfirmasi",
    "aktifkan",
    "nonaktifkan",
    "izinkan",
    "tolak",
    "setuju",
    "lanjutkan",
    "mulai",
    "stop",
    "berhenti",
    "ayo",
    "mari",
    "yuk",
    "silakan",
    "monggo",
    "jangan",
    "stop",
    "hindari",
    "buktikan",
    "rasakan",
    "nikmati",
    "pastikan",
    "segera",
    "buruan",
    "wajib",
    "harus",
    "perlu",
    "butuh",
    "diperlukan",
    "disarankan",
    "dianjurkan",
}
PRONOUN_WORDS = {
    "saya",
    "aku",
    "kami",
    "kita",
    "gue",
    "gua",
    "gw",
    "ane",
    "beta",
    "diriku",
    "anda",
    "kamu",
    "kalian",
    "situ",
    "engkau",
    "dirimu",
    "diri anda",
    "dia",
    "ia",
    "beliau",
    "mereka",
    "dirinya",
    "ku",
    "-ku",
    "mu",
    "-mu",
    "nya",
    "-nya",
    "milikku",
    "punyaku",
}


def extract_all_features(text):
    feats = {
        "text_length": 0,
        "word_count": 0,
        "avg_word_length": 0,
        "uppercase_ratio": 0,
        "question_marks": 0,
        "exclamation_marks": 0,
        "consonant_vowel_ratio": 0,
        "max_word_repetition": 0,
        "url_count": 0,
        "has_url": 0,
        "contact_info_count": 0,
        "has_contact_info": 0,
        "has_promo_code": 0,
        "char_repetition": 0,
        "has_char_repetition": 0,
        "emoji_repetition": 0,
        "has_question": 0,
        "has_comparison": 0,
        "has_off_topic": 0,
        "app_term_count": 0,
        "hyperbolic_word_count": 0,
        "imperative_count": 0,
        "comparison_words": 0,
        "question_words": 0,
        "sentence_count": 0,
        "pronoun_count": 0,
        "first_word_is_verb": 0,
        "contains_gibberish": 0,
    }

    if not isinstance(text, str):
        text = str(text or "")
    text = text.strip()
    length = len(text)
    if length == 0:
        return feats

    words = text.split()
    word_count = len(words)
    if word_count == 0:
        feats["text_length"] = length
        return feats

    words_lower = [w.lower() for w in words]
    word_set = set(words_lower)
    counts = Counter(words_lower)

    # Panjang karakter teks
    feats["text_length"] = length
    # Jumlah kata
    feats["word_count"] = word_count
    # Rata-rata panjang kata
    feats["avg_word_length"] = sum(map(len, words)) / word_count
    # Rasio huruf kapital
    feats["uppercase_ratio"] = (
        (sum(1 for c in text if c.isupper()) / length) if length else 0
    )
    # Jumlah tanda tanya
    feats["question_marks"] = text.count("?")
    # Jumlah tanda seru
    feats["exclamation_marks"] = text.count("!")
    # Rasio konsonan/vokal
    vowels = len(VOW_RE.findall(text))
    cons = len(CON_RE.findall(text))
    feats["consonant_vowel_ratio"] = cons / vowels if vowels > 0 else float(cons > 0)
    # Frekuensi kata yang paling sering muncul
    feats["max_word_repetition"] = max(counts.values()) if counts else 0

    # ---

    # Jumlah URL
    feats["url_count"] = len(URL_RE.findall(text))
    # Ada URL?
    feats["has_url"] = int(bool(URL_RE.search(text)))
    # Jumlah kontak (mention, telepon, email)
    feats["contact_info_count"] = (
        len(MENTION_RE.findall(text))
        + len(PHONE_RE.findall(text))
        + len(EMAIL_RE.findall(text))
    )
    # Ada kontak?
    feats["has_contact_info"] = bool(
        PHONE_RE.search(text) or EMAIL_RE.search(text) or MENTION_RE.search(text)
    )
    # Ada kode promo?
    feats["has_promo_code"] = bool(PROMO_RE.search(text.lower()))
    # Ada pengulangan karakter ≥5 kali?
    feats["char_repetition"] = len(CHAR_REP_RE.findall(text))
    feats["has_char_repetition"] = bool(CHAR_REP_RE.search(text))
    # Ada pengulangan emoji ≥5 kali?
    feats["emoji_repetition"] = len(EMOJI_REP_RE.findall(text))
    # Ada pertanyaan? (tanda tanya atau kata tanya)
    feats["has_question"] = "?" in text or bool(word_set & QUESTION_WORDS)
    # Ada kata perbandingan + istilah aplikasi?
    feats["has_comparison"] = bool(word_set & COMPARISON_WORDS and word_set & APP_TERMS)
    # Ada topik off-topic?
    feats["has_off_topic"] = bool(word_set & OFF_TOPIC_KEYS)

    # ---

    # Jumlah kata domain aplikasi
    feats["app_term_count"] = sum(w in APP_TERMS for w in words_lower)
    # Jumlah kata hyperbolik/promosi
    feats["hyperbolic_word_count"] = sum(w in HYPERBOLIC_WORDS for w in words_lower)
    # Jumlah kata kerja imperatif/promosi
    feats["imperative_count"] = sum(w in IMPERATIVE_WORDS for w in words_lower)
    # Jumlah kata perbandingan
    feats["comparison_words"] = sum(w in COMPARISON_WORDS for w in words_lower)
    # Jumlah kata tanya
    feats["question_words"] = sum(w in QUESTION_WORDS for w in words_lower)

    # ---

    # Jumlah kalimat (berdasarkan ., !, ?)
    feats["sentence_count"] = text.count(".") + text.count("!") + text.count("?") + 1
    # Jumlah kata ganti orang
    feats["pronoun_count"] = sum(w in PRONOUN_WORDS for w in words_lower)
    # Kata pertama adalah kata kerja imperatif/promosi?
    feats["first_word_is_verb"] = (
        int(words_lower[0] in IMPERATIVE_WORDS) if words_lower else 0
    )
    # Ada gibberish? (kata tanpa vokal ≥5 huruf)
    feats["contains_gibberish"] = int(
        any(len(w) >= 5 and not VOW_RE.search(w) for w in words_lower)
    )

    return feats


class IndoTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, slang_dict_path, use_stemmer=False):
        if (
            slang_dict_path is None
            or not isinstance(slang_dict_path, str)
            or not slang_dict_path.strip()
        ):
            raise ValueError(
                "Parameter 'slang_dict_path' harus berupa string path yang valid ke file kamus slang."
            )
        self.slang_dict_path = os.path.normpath(slang_dict_path)
        self.use_stemmer = use_stemmer
        self.slang_dict_ = {}
        self.stemmer_ = None
        self.stopwords_ = {
            "yang",
            "untuk",
            "dan",
            "di",
            "ke",
            "dari",
            "ini",
            "itu",
            "dengan",
            "atau",
            "tapi",
            "adalah",
            "adanya",
            "pada",
            "oleh",
            "antara",
            "sehingga",
            "karena",
            "bahwa",
            "sebagai",
            "secara",
            "namun",
            "saat",
            "tersebut",
            "bahkan",
            "saja",
            "lah",
            "pun",
            "juga",
        }
        self._initialize()

    def _initialize(self):
        self.slang_dict_ = self._load_slang_dict()
        if self.use_stemmer:
            try:
                # factory = StemmerFactory()
                # self.stemmer_ = factory.create_stemmer()
                print(
                    f"IndoTextPreprocessor initialized (Stemmer ENABLED, slang dict from: {self.slang_dict_path})."
                )
            except Exception as e:
                print(
                    f"ERROR: Gagal menginisialisasi Sastrawi Stemmer: {e}. Stemming dinonaktifkan."
                )
                self.stemmer_ = None
                self.use_stemmer = False
        else:
            self.stemmer_ = None
            print(
                f"IndoTextPreprocessor initialized (Stemmer DISABLED, slang dict from: {self.slang_dict_path})."
            )

    def _load_slang_dict(self):
        slang_data = {}
        abs_path = os.path.abspath(self.slang_dict_path)
        try:
            if os.path.exists(self.slang_dict_path):
                with open(self.slang_dict_path, "r", encoding="utf-8") as f:
                    slang_data = json.load(f)
                print(f"Kamus slang dari '{self.slang_dict_path}' berhasil dimuat.")
            else:
                print(
                    f"ERROR: File kamus slang tidak ditemukan di '{abs_path}'. Normalisasi slang mungkin tidak efektif."
                )
        except json.JSONDecodeError:
            print(
                f"ERROR: Gagal membaca file JSON kamus slang dari '{abs_path}'. Format mungkin salah."
            )
        except Exception as e:
            print(
                f"ERROR: Terjadi error saat memuat kamus slang dari '{abs_path}': {e}"
            )
        return slang_data

    def _resolve_slang(self, word, max_depth=10):
        current_word, seen_words, depth = word, {word}, 0
        slang_dict = self.slang_dict_
        while current_word in slang_dict and depth < max_depth:
            next_word = slang_dict.get(current_word)
            if (
                next_word is None
                or next_word in seen_words
                or next_word == current_word
            ):
                break
            seen_words.add(next_word)
            current_word, depth = next_word, depth + 1
        return current_word

    def _cleaning_ulasan(self, text):
        if not isinstance(text, str):
            return ""
        text = text.strip().lower()
        text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
        text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
        text = re.sub(
            r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+\.[a-z]{2,4}/)
            (?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+ 
            (?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»"
            "'']))""",
            " ",
            text,
            flags=re.VERBOSE,
        )
        text = re.sub(
            r"\(cont\)|lanjut(?:kan)?|lihat selengkapnya|see more",
            " ",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _normalize_slang(self, text):
        if not self.slang_dict_:
            return text
        toks = word_tokenize(text)
        out = [self._resolve_slang(w) for w in toks]
        return " ".join(out)

    def _remove_stopwords(self, text):
        toks = word_tokenize(text)
        return " ".join([w for w in toks if w not in self.stopwords_])

    def _stem_text(self, text):
        if self.use_stemmer and self.stemmer_:
            toks = word_tokenize(text)
            return " ".join(self.stemmer_.stem(w) for w in toks)
        return text

    def _process_single_text(self, text):
        text_cleaned = self._cleaning_ulasan(text)
        text_normalized = self._normalize_slang(text_cleaned)
        text_no_stopwords = self._remove_stopwords(text_normalized)
        text_final = self._stem_text(text_no_stopwords)
        return text_final

    def fit(self, X, y=None):
        # _initialize sudah dipanggil di __init__
        # Tidak ada fitting state tambahan di sini
        return self

    def transform(self, X, y=None):
        if (
            self.use_stemmer and self.stemmer_ is None and self.use_stemmer
        ) or not hasattr(self, "slang_dict_"):
            print(
                "Warning: Preprocessor state (stemmer/slang dict) tidak ditemukan saat transform. Mencoba inisialisasi ulang."
            )
            self._initialize()

        if not isinstance(X, pd.Series):
            try:
                X_series = pd.Series(X, dtype="string").fillna("")
            except Exception as e:
                print(
                    f"Error converting input X to Pandas Series: {e}. Processing element-wise."
                )
                if hasattr(X, "__iter__") and not isinstance(X, str):
                    results = [self._process_single_text(str(item or "")) for item in X]
                    return pd.Series(results, dtype="string").fillna("")
                else:
                    result = self._process_single_text(str(X or ""))
                    return pd.Series([result], dtype="string").fillna("")
        else:
            X_series = X.astype("string").fillna("")

        print("Applying combined preprocessing steps...")
        processed_series = X_series.apply(self._process_single_text)
        print("Preprocessing complete.")
        return processed_series


class StatisticalFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler_ = None
        self.feature_names_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.Series):
            X_list = X.tolist()
        elif isinstance(X, np.ndarray):
            X_list = X.tolist()
        elif isinstance(X, (list, tuple)):
            X_list = list(X)
        else:
            try:
                X_list = list(X)
            except TypeError:
                raise ValueError("Input X must be convertible to a list of strings.")

        stat_list = [extract_all_features(text) for text in X_list]
        stat_df = pd.DataFrame(stat_list).fillna(0)
        self.feature_names_ = stat_df.columns.tolist()
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(stat_df)
        return self

    def transform(self, X, y=None):
        if not self.scaler_ or self.feature_names_ is None:
            raise RuntimeError("StatisticalFeatureExtractor has not been fitted yet.")

        if isinstance(X, pd.Series):
            X_list = X.tolist()
        elif isinstance(X, np.ndarray):
            X_list = X.tolist()
        elif isinstance(X, (list, tuple)):
            X_list = list(X)
        else:
            try:
                X_list = list(X)
            except TypeError:
                raise ValueError("Input X must be convertible to a list of strings.")

        stat_list = [extract_all_features(text) for text in X_list]
        stat_df = pd.DataFrame(stat_list).fillna(0)
        stat_df = stat_df.reindex(columns=self.feature_names_, fill_value=0)
        scaled_features = self.scaler_.transform(stat_df)
        return csr_matrix(scaled_features)


# Load pipeline dan kedua model
nb_model = joblib.load(NB_MODEL_PATH)
svm_model = joblib.load(SVM_MODEL_PATH)
reverse_mapping = joblib.load(REVERSE_MAPPING_PATH)
stat_scaler = joblib.load(STAT_SCALER_PATH)
stat_feature_names = joblib.load(STAT_FEATURE_NAMES_PATH)

# List kelas (urutkan sesuai model.classes_ dari NB, diasumsikan sama urutan dengan SVM)
CLASS_LABELS = [reverse_mapping[i] for i in nb_model.classes_]


def build_feature_extractors():
    preproc = IndoTextPreprocessor(slang_dict_path=slang_file_path, use_stemmer=False)
    tfidf = joblib.load(TFIDF_PATH)
    stat_extractor = StatisticalFeatureExtractor()
    stat_extractor.scaler_ = stat_scaler
    stat_extractor.feature_names_ = stat_feature_names
    return preproc, tfidf, stat_extractor


# Preprocessor/stat extractor global
_preproc = None
_tfidf = None
_stat_extractor = None


def get_feature_extractors():
    global _preproc, _tfidf, _stat_extractor
    if _preproc is None or _tfidf is None or _stat_extractor is None:
        _preproc, _tfidf, _stat_extractor = build_feature_extractors()
    return _preproc, _tfidf, _stat_extractor


def predict_spam(
    reviews: List[str], model_type: Literal["nb", "svm"] = "nb", background_tasks=None
) -> Dict[str, Any]:
    import logging

    start_time = time.time()
    if model_type == "nb":
        model = nb_model
    elif model_type == "svm":
        model = svm_model
    else:
        raise ValueError("model_type must be 'nb' or 'svm'")

    preproc, tfidf, stat_extractor = get_feature_extractors()
    # Preprocessing
    X_preproc = preproc.transform(reviews)
    # TF-IDF features
    X_tfidf = tfidf.transform(X_preproc)
    # Statistical features
    X_stat = stat_extractor.transform(reviews)
    # Combine features
    X_combined = hstack([X_tfidf, X_stat])

    preds = model.predict(X_combined)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_combined)
    else:
        probs = np.ones((len(reviews), len(CLASS_LABELS)))

    counts = {label: int(np.sum(preds == i)) for i, label in enumerate(CLASS_LABELS)}
    # Perbaiki percentages agar total selalu 100
    if len(preds) > 0:
        raw_percentages = [
            round(100 * counts[label] / len(preds)) for label in CLASS_LABELS
        ]
        diff = 100 - sum(raw_percentages)
        if diff != 0:
            max_idx = np.argmax([counts[label] for label in CLASS_LABELS])
            raw_percentages[max_idx] += diff
        percentages = {
            label: raw_percentages[i] for i, label in enumerate(CLASS_LABELS)
        }
    else:
        percentages = {label: 0 for label in CLASS_LABELS}
    reviews_by_category = {label: [] for label in CLASS_LABELS}
    for idx, (text, pred_idx) in enumerate(zip(reviews, preds)):
        label = reverse_mapping[pred_idx]
        conf = float(np.max(probs[idx]))
        reviews_by_category[label].append({"text": text, "confidence": conf})
    return_data = {
        "reviews_by_category": reviews_by_category,
    }
    # Upload ke GCS
    bucket_name = os.getenv("GCS_BUCKET_NAME", "quicktify-storage")
    file_id = str(uuid.uuid4())
    destination_blob_name = f"spam_detection_results/{file_id}.json"
    from app.utils.gcs import upload_json_to_gcs

    file_url = upload_json_to_gcs(return_data, bucket_name, destination_blob_name)
    end_time = time.time()
    logging.info(f"[SpamDetection] Waktu proses: {end_time - start_time:.3f} detik")
    return {
        "percentages": percentages,
        "counts": counts,
        "reviews_by_category": reviews_by_category,
        "file_url": file_url,
    }
