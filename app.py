import os
import difflib
import pandas as pd

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
product_df = pd.read_csv("sanjing_notebook_page1_3.csv")

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

    faq_q, faq_a, faq_score = search_faq(user_text)
    prod_row, prod_score = search_product(user_text)

    faq_part = ""
    if faq_q:
        faq_part = f"相關 FAQ：{faq_q}\n回答：{faq_a}\n（相似度 {faq_score:.2f}）"

    product_part = ""
    if prod_row is not None:
        product_part = f"""
可能對應的商品：
- 品牌：{prod_row[PRODUCT_BRAND_COL]}
- 品名/型號：{prod_row[PRODUCT_NAME_COL]}（{prod_row[PRODUCT_MODEL_COL]}）
- 價格：{prod_row[PRODUCT_PRICE_COL]}
- CPU：{prod_row[PRODUCT_CPU_COL]}
- GPU：{prod_row[PRODUCT_GPU_COL]}
- RAM：{prod_row[PRODUCT_RAM_COL]}
- 儲存：{prod_row[PRODUCT_STORAGE_COL]}
- 螢幕：{prod_row[PRODUCT_SIZE_COL]}
- 重量：{prod_row[PRODUCT_WEIGHT_COL]}
- 作業系統：{prod_row[PRODUCT_OS_COL]}
- 特色：{prod_row[PRODUCT_FEATURE_COL]}
- 網址：{prod_row[PRODUCT_URL_COL]}
（相似度 {prod_score:.2f}）
""".strip()

    if not faq_part and not product_part:
        prompt = f"""
你是三井3C客服，請回答以下問題（繁體中文）：

使用者問題：
{user_text}

請注意：
- 不要亂編規格
- 價格與保固請註明以門市與官網為準
"""
    else:
        prompt = f"""
你是三井3C客服，請根據以下資料回覆使用者。

【使用者問題】
{user_text}

【FAQ】
{faq_part or "（無）"}

【商品資料】
{product_part or "（無）"}

請：
1. 先用 1～3 行講結論
2. 再條列建議與適合族群
3. 嚴禁編造不存在的規格
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是專業且謹慎的 3C 客服人員"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
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
