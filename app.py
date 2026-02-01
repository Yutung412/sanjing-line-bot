import os
import difflib
import pandas as pd
import json

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from dotenv import load_dotenv
from groq import Groq

# ========= 讀取環境變數 =========
load_dotenv()

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("LINE_CHANNEL_SECRET 載入成功：", bool(LINE_CHANNEL_SECRET))
print("GROQ_API_KEY 載入成功：", bool(GROQ_API_KEY))


# ========= 設定 Groq =========
client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
print(">>> Using Groq model:", MODEL_NAME)


# ========= 讀取 FAQ + 商品資料 =========
faq_df = pd.read_excel("faq.xlsx")
product_df = pd.read_csv("sanjing_notebook_page1_3.csv", dtype=str, keep_default_na=False)

import re


PRODUCT_BRAND_COL   = "brand"
PRODUCT_NAME_COL    = "product_name"
PRODUCT_MODEL_COL   = "model_code"
PRODUCT_PRICE_COL   = "price"
PRODUCT_CPU_COL     = "cpu"
PRODUCT_GPU_COL     = "gpu"
PRODUCT_RAM_COL     = "memory"
PRODUCT_STORAGE_COL = "storage_detail"
PRODUCT_SIZE_COL    = "display_size"
PRODUCT_WEIGHT_COL  = "weight"
PRODUCT_OS_COL      = "os"
PRODUCT_WIFI_COL    = "wifi"
PRODUCT_FEATURE_COL = "feature"
PRODUCT_URL_COL     = "product_url"
PRODUCT_WARRANTY_COL = "warranty"

# ========= 資料正規化（商品） =========

# 1) brand 統一：把「ASUS 華碩」這種轉成 ASUS、Lenovo 聯想 -> Lenovo ...
BRAND_KEYWORDS = [
    ("Acer", ["acer", "宏碁"]),
    ("ASUS", ["asus", "華碩"]),
    ("GIGABYTE", ["gigabyte", "技嘉"]),
    ("HP", ["hp", "惠普"]),
    ("Lenovo", ["lenovo", "聯想"]),
    ("LG", ["lg", "樂金"]),
    ("MSI", ["msi", "微星"]),
]

def normalize_brand(x: str) -> str:
    s = (x or "").strip().lower()
    for canon, keys in BRAND_KEYWORDS:
        if any(k in s for k in keys):
            return canon
    # 找不到就回原字串（保底）
    return (x or "").strip()

# 2) price 轉數字：你現在看起來是純數字，但這邊保險（避免未來有逗號/符號）
def parse_price(x: str):
    s = (x or "").strip()
    # 只保留數字
    digits = re.sub(r"[^\d]", "", s)
    return int(digits) if digits else None

# 3) display_size 抽尺寸（吋）： "14吋 FHD ..." -> 14
def parse_display_inch(x: str):
    s = (x or "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*吋", s)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    # 有些資料可能是 14" 這種
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*\"", s)
    if m2:
        try:
            return float(m2.group(1))
        except:
            return None
    return None

# 4) weight 統一成 kg 浮點數：支援 1.3kg / 1.3KG / "約1.3kg"
def parse_weight_kg(x: str):
    s = (x or "").strip().lower()
    # 抓像 1.25kg / 1kg / 1.25 KG
    m = re.search(r"(\d+(?:\.\d+)?)\s*kg", s, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    # 若出現 g（少見，但保底）
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*g", s, flags=re.IGNORECASE)
    if m2:
        try:
            return float(m2.group(1)) / 1000.0
        except:
            return None
    return None

# 5) warranty 抽保固年：支援 1年/一年/兩年/2年/含文字
CH_NUM = {"一":1, "二":2, "兩":2, "三":3, "四":4, "五":5}

def parse_warranty_years(x: str):
    s = (x or "").strip()
    if not s:
        return None

    # 先抓阿拉伯數字：2年、3 年
    m = re.search(r"(\d+)\s*年", s)
    if m:
        try:
            return int(m.group(1))
        except:
            pass

    # 再抓中文數字：一年、兩年、三年...
    m2 = re.search(r"([一二兩三四五])\s*年", s)
    if m2:
        return CH_NUM.get(m2.group(1))

    return None

# ========= warranty 正規化 =========
if PRODUCT_WARRANTY_COL in product_df.columns:
    product_df["warranty_years"] = product_df[PRODUCT_WARRANTY_COL].apply(parse_warranty_years)
else:
    product_df["warranty_years"] = None


# ========= 將正規化欄位加到 product_df =========

# brand_norm
if PRODUCT_BRAND_COL in product_df.columns:
    product_df["brand_norm"] = product_df[PRODUCT_BRAND_COL].apply(normalize_brand)
else:
    product_df["brand_norm"] = ""

# price_num
if PRODUCT_PRICE_COL in product_df.columns:
    product_df["price_num"] = product_df[PRODUCT_PRICE_COL].apply(parse_price)
else:
    product_df["price_num"] = None

# display_inch
if PRODUCT_SIZE_COL in product_df.columns:
    product_df["display_inch"] = product_df[PRODUCT_SIZE_COL].apply(parse_display_inch)
else:
    product_df["display_inch"] = None

# weight_kg
if PRODUCT_WEIGHT_COL in product_df.columns:
    product_df["weight_kg"] = product_df[PRODUCT_WEIGHT_COL].apply(parse_weight_kg)
else:
    product_df["weight_kg"] = None

print("商品正規化完成：")
print("- price_num 有值筆數：", product_df["price_num"].notna().sum())
print("- display_inch 有值筆數：", product_df["display_inch"].notna().sum())
print("- weight_kg 有值筆數：", product_df["weight_kg"].notna().sum())
print("- warranty_years 有值筆數：", product_df["warranty_years"].notna().sum())


FAQ_QUESTION_COL = "修正提問"
FAQ_ANSWER_COL   = "修正後回答"



# ========= FAQ 搜尋（模糊比對） =========
def search_faq(user_text):
    best_q, best_a, best_score = None, None, 0
    for _, row in faq_df.iterrows():
        q = str(row[FAQ_QUESTION_COL])
        a = str(row[FAQ_ANSWER_COL])
        score = difflib.SequenceMatcher(None, user_text, q).ratio()
        if score > best_score:
            best_score = score
            best_q, best_a = q, a
    if best_score < 0.3:
        return None, None, 0
    return best_q, best_a, best_score

# ========= 從使用者文字抽需求（預算/品牌/尺寸/保固/重量） =========
def extract_user_constraints(user_text: str):
    import re
    t = (user_text or "").strip().lower()

    # 品牌：看使用者有沒有提到
    brand = None
    for canon, keys in BRAND_KEYWORDS:
        if any(k in t for k in keys):
            brand = canon
            break

    # 尺寸：14吋 / 14 吋 / 14"
    inch = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*吋", t)
    if m:
        inch = float(m.group(1))
    else:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*\"", t)
        if m2:
            inch = float(m2.group(1))

    # 預算：3萬內 / 30000以下 / 3 萬 以內 / 30k內
    budget_max = None
    # 例如 "3萬內"
    m = re.search(r"(\d+)\s*萬\s*(?:內|以下|以內|不超過)", t)
    if m:
        budget_max = int(m.group(1)) * 10000
    # 例如 "30000以下"
    m2 = re.search(r"(\d{4,6})\s*(?:元|塊)?\s*(?:內|以下|以內|不超過)", t)
    if m2:
        budget_max = int(m2.group(1))

    # 保固：2年 / 兩年
    warranty_years = None
    m = re.search(r"(\d+)\s*年", t)
    if m:
        warranty_years = int(m.group(1))
    else:
        m2 = re.search(r"(一|二|兩|三|四|五)\s*年", t)
        if m2:
            warranty_years = CH_NUM.get(m2.group(1))

    # 重量：1.5kg內 / 1.2kg以下
    weight_max = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*kg\s*(?:內|以下|以內|不超過)", t)
    if m:
        weight_max = float(m.group(1))

    return {
        "brand": brand,
        "inch": inch,
        "budget_max": budget_max,
        "warranty_years": warranty_years,
        "weight_max": weight_max,
    }


# ========= 商品搜尋（TopK，條件過濾 + 打分） =========
def search_product_topk(user_text: str, topk: int = 5):
    """
    先用條件（品牌/價格/尺寸/保固/重量）過濾，再用 difflib 做 TopK 排序
    demo 成功率會比純字串比對高很多
    """
    text = (user_text or "").strip()
    if not text:
        return []

    c = extract_user_constraints(text)

    df = product_df.copy()

    # (1) 品牌過濾
    if c["brand"]:
        df = df[df["brand_norm"] == c["brand"]]

    # (2) 價格上限
    if c["budget_max"] is not None:
        df = df[df["price_num"].notna() & (df["price_num"] <= c["budget_max"])]

    # (3) 尺寸：允許 ±0.2 吋容差（避免 14 / 14.0 / 14.1）
    if c["inch"] is not None:
        df = df[df["display_inch"].notna() & (df["display_inch"].between(c["inch"] - 0.2, c["inch"] + 0.2))]

    # (4) 保固：有寫才比（有些商品可能沒資料）
    if c["warranty_years"] is not None and "warranty_years" in df.columns:
        df = df[df["warranty_years"].notna() & (df["warranty_years"] >= c["warranty_years"])]

    # (5) 重量：可選
    if c["weight_max"] is not None:
        df = df[df["weight_kg"].notna() & (df["weight_kg"] <= c["weight_max"])]

    # 先印：條件抽取結果 + 過濾後剩幾筆
    print("constraints:", c, "filtered_rows_before_fallback:", len(df))

    # 如果條件過濾後完全沒東西：放寬（避免 demo 一直空）
    if df.empty:
        df = product_df.copy()
        print("fallback_to_all_products. rows:", len(df))



    scored = []
    for _, row in df.iterrows():
        # 用更多欄位組成 target（提高比對命中）
        target = " ".join([
            row.get(PRODUCT_BRAND_COL, ""),
            row.get("brand_norm", ""),
            row.get(PRODUCT_NAME_COL, ""),
            row.get(PRODUCT_MODEL_COL, ""),
            row.get(PRODUCT_CPU_COL, ""),
            row.get(PRODUCT_GPU_COL, ""),
            row.get(PRODUCT_RAM_COL, ""),
            row.get(PRODUCT_STORAGE_COL, ""),
            row.get(PRODUCT_SIZE_COL, ""),
            row.get(PRODUCT_OS_COL, ""),
            row.get(PRODUCT_FEATURE_COL, ""),
        ])
        score = difflib.SequenceMatcher(None, text, target).ratio()
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    # demo：永遠回傳 TopK（不要再用門檻把結果濾光）
    top = scored[:topk]
    results = [(row, score) for score, row in top]
    return results



DEMO_COLS = [
    "brand", "product_name", "model_code", "price",
    "cpu", "gpu", "memory", "storage_detail",
    "display_size", "weight", "os", "product_url"
]

def build_topk_markdown(rows):
    """
    rows: [(row, score), ...]
    轉成 markdown 表格給 LLM 產生回覆用
    """
    if not rows:
        return ""

    only_rows = [r for r, s in rows]
    df = pd.DataFrame(only_rows)

    cols = [c for c in DEMO_COLS if c in df.columns]
    if not cols:
        return ""

    return json.dumps(df[cols].fillna("").to_dict(orient="records"), ensure_ascii=False)

# ========= Flask + LINE 初始化 =========
app = Flask(__name__)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# Render / 健康檢查用首頁
@app.route("/", methods=["GET"])
def index():
    return "LINE bot is running on Render.", 200


@app.route("/callback", methods=["GET", "POST"])
def callback():
    if request.method == "GET":
        return "OK", 200

    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return abort(400)

    return "OK", 200


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text.strip()

    # 0) 打招呼（無狀態版本）
    greetings = ["哈囉", "你好", "嗨", "hi", "hello", "您好"]
    if user_text.lower() in [g.lower() for g in greetings]:
        welcome = (
            "您好，歡迎來到三井3C～\n"
            "想找哪一類商品呢？（例如：筆電/平板/顯卡/螢幕）\n"
            "也可以直接告訴我：預算、用途、尺寸、品牌偏好，我可以幫您推薦。"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=welcome))
        return

    # 1) FAQ（保留原本）
    faq_q, faq_a, faq_score = search_faq(user_text)
    faq_part = ""
    if faq_q:
        faq_part = f"相關 FAQ：{faq_q}\n回答：{faq_a}\n（相似度 {faq_score:.2f}）"

    # 2) 商品 TopK 檢索（完全不靠 LLM、不靠 markdown 檔）
    topk_rows = search_product_topk(user_text, topk=5)
    cand_md = build_topk_markdown(topk_rows)

    print("user_text:", user_text)
    print("topk_scores:", [round(s, 3) for _, s in topk_rows])
    print("cand_md length:", len(cand_md))


    # 4) 若沒有 FAQ 且沒有候選商品：才回澄清（避免幻覺）
    if not faq_part and not cand_md:
        answer = (
            "您好～我可以幫您推薦，但目前資訊還不夠精準。\n"
            "想先確認：\n"
            "1) 預算範圍？\n"
            "2) 主要用途（文書/剪片/遊戲/攜帶）？\n"
            "3) 螢幕尺寸偏好？\n"
            "4) RAM/儲存需求（例如 16GB / 1TB）？\n"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer[:1000]))
        return



    # 5) 第二階段：用候選表 + FAQ 產生最終回覆（鎖死資料來源）
    prompt = f"""
你是三井3C客服，只能使用我提供的【FAQ】與【候選商品JSON】回覆。

【使用者問題】
{user_text}

【FAQ】
{faq_part or "（無）"}

【候選商品JSON（你只能引用 JSON 內欄位）】
{cand_md or "（無）"}

規則（非常重要）：
- 嚴禁新增任何 JSON 中不存在的商品、規格、價格、保固、活動、庫存
- 若使用者問到 JSON 沒有的資訊，回答：「資料未提供，建議以官網/門市為準」
- 回覆格式：
1) 先用 1～2 句打招呼 + 1～3 行結論
2) 推薦 1～3 台（每台列：型號(model_code)、價格(price)、適合誰、連結(product_url)）
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是專業且謹慎的 3C 客服人員，只能依據提供的資料回答，缺資料就說未提供。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print("Groq 發生錯誤：", repr(e))
        answer = "目前系統較忙，請稍後再試，或洽詢門市人員。"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer[:1000])
    )


# ========= 主程式（Render & 本機共用） =========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
