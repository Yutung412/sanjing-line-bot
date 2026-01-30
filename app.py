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
product_df = pd.read_csv("sanjing_notebook_page1_3.csv", dtype=str, keep_default_na=False)



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

# ========= 商品搜尋（TopK，demo 穩定版） =========
def search_product_topk(user_text: str, topk: int = 5):
    """
    用 difflib 在本地做 TopK 檢索（不靠 LLM、不靠 markdown檔）
    讓中文需求（剪片/遊戲/16GB/1TB）也比較容易比到
    """
    text = (user_text or "").strip()
    if not text:
        return []

    scored = []
    for _, row in product_df.iterrows():
        target = " ".join([
            str(row.get(PRODUCT_BRAND_COL, "")),
            str(row.get(PRODUCT_NAME_COL, "")),
            str(row.get(PRODUCT_MODEL_COL, "")),
            str(row.get(PRODUCT_CPU_COL, "")),
            str(row.get(PRODUCT_GPU_COL, "")),
            str(row.get(PRODUCT_RAM_COL, "")),
            str(row.get(PRODUCT_STORAGE_COL, "")),
            str(row.get(PRODUCT_SIZE_COL, "")),
            str(row.get(PRODUCT_OS_COL, "")),
            str(row.get(PRODUCT_FEATURE_COL, "")),
        ])

        score = difflib.SequenceMatcher(None, text, target).ratio()
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    # demo 先放寬門檻，避免「永遠找不到」
    results = [(row, score) for score, row in scored[:topk] if score >= 0.12]
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

    return df[cols].fillna("").astype(str).to_markdown(index=False)

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
