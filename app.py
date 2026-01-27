import os
import difflib
import pandas as pd

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from dotenv import load_dotenv
from groq import Groq
import json

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
product_df = pd.read_csv("sanjing_notebook_page1_3.csv")

# 讀 Markdown 檔（Colab 產生的）
print("product_index.md exists:", os.path.exists("product_index.md"))
print("product_table.md exists:", os.path.exists("product_table.md"))

with open("product_index.md", encoding="utf-8") as f:
    PRODUCT_INDEX = f.read()

with open("product_table.md", encoding="utf-8") as f:
    PRODUCT_TABLE_MD = f.read()

print("PRODUCT_INDEX length:", len(PRODUCT_INDEX))
print("PRODUCT_TABLE_MD length:", len(PRODUCT_TABLE_MD))

FAQ_QUESTION_COL = "修正提問"
FAQ_ANSWER_COL   = "修正後回答"

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


# ========= 商品搜尋（模糊比對） =========
def search_product(user_text):
    best_row, best_score = None, 0
    for _, row in product_df.iterrows():
        target = " ".join([
            str(row[PRODUCT_BRAND_COL]),
            str(row[PRODUCT_NAME_COL]),
            str(row[PRODUCT_MODEL_COL])
        ])
        score = difflib.SequenceMatcher(None, user_text, target).ratio()
        if score > best_score:
            best_score = score
            best_row = row
    if best_score < 0.3:
        return None, 0
    return best_row, best_score

def safe_json_extract(text: str):
    """從模型輸出中抽出第一段 JSON 並解析（容錯用）"""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return json.loads(text[start:end+1])
    except Exception:
        return None

def llm_pick_candidates(user_text: str, topk: int = 5):
    """
    第一階段：用 PRODUCT_INDEX 讓模型挑出候選 model_code
    注意：只做檢索，不要讓模型直接推薦
    """
    retrieval_prompt = f"""
你是檢索器，不是客服。你的任務是：從【商品索引】挑出最符合使用者需求的候選商品型號。

【使用者需求】
{user_text}

【商品索引】
{PRODUCT_INDEX}

規則：
- 只能從索引中挑選，禁止臆測不存在的型號
- 請只回傳 JSON（不要多任何字），格式固定：
{{"candidates":["型號1","型號2"],"reason":"一句話理由"}}
- candidates 最多 {topk} 個；找不到就回 []，reason 說明缺少哪些條件（例如預算/尺寸/用途/品牌）
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是嚴謹的資料檢索器，只輸出 JSON，不輸出多餘文字。"},
                {"role": "user", "content": retrieval_prompt}
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        data = safe_json_extract(raw) or {}
        cands = data.get("candidates", [])

        # 清理 candidates：只留字串、去空白、去重
        cands = [str(x).strip() for x in cands if str(x).strip()]
        seen = set()
        out = []
        for x in cands:
            if x not in seen:
                seen.add(x)
                out.append(x)

        return out[:topk], data.get("reason", "")
    except Exception as e:
        print("候選檢索失敗：", e)
        return [], "檢索失敗"

DEMO_COLS = [
    "brand", "product_name", "model_code", "price",
    "cpu", "gpu", "memory", "storage_detail",
    "display_size", "weight", "os", "product_url"
]

def build_candidate_markdown(candidates):
    """把候選型號對應的列，轉成 Markdown 表格給第二階段回覆用"""
    if not candidates:
        return ""
    df = product_df[product_df[PRODUCT_MODEL_COL].astype(str).isin([str(x) for x in candidates])].copy()
    if df.empty:
        return ""
    return df[DEMO_COLS].fillna("").astype(str).to_markdown(index=False)

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


# ========= LINE 訊息處理 =========
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text

    # 先走 FAQ（保留原本）
    faq_q, faq_a, faq_score = search_faq(user_text)
    faq_part = ""
    if faq_q:
        faq_part = f"相關 FAQ：{faq_q}\n回答：{faq_a}\n（相似度 {faq_score:.2f}）"

    # ✅ 第一階段：用 PRODUCT_INDEX 找候選型號
    candidates, reason = llm_pick_candidates(user_text, topk=5)

    # ✅ 第二階段：把候選型號對應的商品列出成 Markdown 表格
    cand_md = build_candidate_markdown(candidates)

    # 若沒有 FAQ 且沒有候選商品：直接回澄清問題（不要讓模型亂推）
    if not faq_part and not cand_md:
        answer = (
            "我目前在商品清單中找不到能直接對應的型號。\n"
            "為了推薦更準，想先確認：\n"
            "1) 預算範圍？\n"
            "2) 主要用途（文書/剪片/遊戲/攜帶）？\n"
            "3) 螢幕尺寸偏好？\n"
            "4) RAM/儲存需求（例如 16GB / 1TB）？\n"
            f"（檢索原因：{reason}）"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer[:1000]))
        return

    # 有候選 or 有 FAQ：用「候選表」+「FAQ」讓模型回答（鎖死）
    prompt = f"""
你是三井3C客服，只能使用我提供的【FAQ】與【候選商品表】回覆。

【使用者問題】
{user_text}

【FAQ】
{faq_part or "（無）"}

【候選商品表（你只能引用表內欄位）】
{cand_md or "（無）"}

規則（非常重要）：
- 嚴禁新增任何表格中不存在的商品、規格、價格、保固、活動、庫存
- 若使用者問到表格沒有的資訊，回答：「資料未提供，建議以官網/門市為準」
- 回覆格式：
1) 1～3 行結論
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
        print("Groq 發生錯誤：", e)
        answer = "目前系統較忙，請稍後再試，或洽詢門市人員。"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer[:1000])
    )


# ========= 主程式（Render & 本機共用） =========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
