import os
import difflib
import pandas as pd
import numpy as np  # 核心：用於向量運算
import google.generativeai as genai

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from dotenv import load_dotenv

# ========= 1. 環境設定 =========
load_dotenv()

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 設定 Gemini
genai.configure(api_key=GEMINI_API_KEY)
# 使用最新的 Flash 模型，速度快且聰明
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ========= 2. 讀取資料 =========
print("正在讀取資料庫...")
# Render 上會從專案根目錄讀取
faq_df = pd.read_excel("faq.xlsx")
product_df = pd.read_csv("sanjing_notebook_page1_3.csv")

# 定義欄位名稱
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

# ========= 3. 向量搜尋引擎初始化 (關鍵步驟) =========

def get_embedding(text):
    """
    呼叫 Gemini Embedding API 將文字轉為向量
    model: text-embedding-004 是目前最新的嵌入模型
    """
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding Error: {e}")
        return []

print("正在建立商品向量索引 (第一次啟動需時較長，請稍候)...")

# 步驟 A: 組合豐富的搜尋字串
# 技巧：加上「中文欄位名稱」讓 AI 更清楚每個數字代表什麼
product_df['search_text'] = product_df.apply(
    lambda row: (
        f"品牌:{row[PRODUCT_BRAND_COL]} "
        f"型號:{row[PRODUCT_NAME_COL]} ({row[PRODUCT_MODEL_COL]}) "
        f"價格:{row[PRODUCT_PRICE_COL]} "
        f"處理器:{row[PRODUCT_CPU_COL]} "
        f"顯卡:{row[PRODUCT_GPU_COL]} "
        f"記憶體:{row[PRODUCT_RAM_COL]} "
        f"硬碟:{row[PRODUCT_STORAGE_COL]} "
        f"特色:{row[PRODUCT_FEATURE_COL]}"
    ),
    axis=1
)

# 步驟 B: 計算所有商品的向量
product_embeddings = []
valid_indices = [] # 用來紀錄哪些 row 成功轉成向量

for idx, text in enumerate(product_df['search_text']):
    # 轉成字串並處理空值
    safe_text = str(text) if pd.notna(text) else ""
    vec = get_embedding(safe_text)
    if vec:
        product_embeddings.append(vec)
        valid_indices.append(idx)

# 步驟 C: 轉換為 Numpy Array 以利快速計算
product_embeddings = np.array(product_embeddings)
print(f"索引建立完成！共 {len(product_embeddings)} 筆商品資料已準備好。")


# ========= 4. 搜尋邏輯函數 =========

def search_faq(user_text):
    """FAQ 維持使用字串比對，適合固定問答"""
    best_q, best_a, best_score = None, None, 0
    for _, row in faq_df.iterrows():
        q = str(row[FAQ_QUESTION_COL])
        a = str(row[FAQ_ANSWER_COL])
        # 使用 SequenceMatcher 比對相似度
        score = difflib.SequenceMatcher(None, user_text, q).ratio()
        if score > best_score:
            best_score = score
            best_q, best_a = q, a
    
    # 門檻值 0.3
    if best_score < 0.3:
        return None, None, 0
    return best_q, best_a, best_score

def search_product_vector(user_text, top_k=3):
    """商品使用向量搜尋，支援語意理解"""
    if len(product_embeddings) == 0:
        return []

    # 1. 將使用者問題轉向量
    try:
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=user_text,
            task_type="retrieval_query"
        )['embedding']
    except Exception as e:
        print(f"Query Embedding Error: {e}")
        return []
    
    query_vec = np.array(query_embedding)
    
    # 2. 計算餘弦相似度 (Cosine Similarity)
    # 公式：(A . B) / (|A| * |B|)
    dot_products = np.dot(product_embeddings, query_vec)
    norm_products = np.linalg.norm(product_embeddings, axis=1)
    norm_query = np.linalg.norm(query_vec)
    
    # 加上 1e-9 避免除以 0
    cosine_scores = dot_products / (norm_products * norm_query + 1e-9)
    
    # 3. 排序並取出前 K 名
    # argsort 預設是由小到大，所以取後面 [-top_k:] 再反轉 [::-1] 變成由大到小
    top_indices_sorted = np.argsort(cosine_scores)[-top_k:][::-1]
    
    results = []
    for idx in top_indices_sorted:
        score = cosine_scores[idx]
        # 門檻值：0.35 (可視情況微調，向量搜尋通常分數會比較高)
        if score > 0.35:
            original_idx = valid_indices[idx]
            row = product_df.iloc[original_idx]
            results.append((row, score))
            
    return results

# ========= 5. Flask 與 Line Bot 設定 =========
app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route("/", methods=["GET"])
def index():
    return "三井3C AI 客服機器人 (向量增強版) 運行中...", 200

@app.route("/callback", methods=["GET", "POST"])
def callback():
    if request.method == "GET":
        return "OK", 200
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return "Invalid signature", 400
    return "OK", 200

# ========= 6. 訊息處理主邏輯 =========
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text

    # --- A. 搜尋階段 ---
    faq_q, faq_a, faq_score = search_faq(user_text)
    found_products = search_product_vector(user_text, top_k=3)

    # --- B. 資料整理階段 ---
    
    # FAQ 資料
    faq_part = ""
    if faq_q:
        faq_part = f"相關 FAQ：{faq_q}\n回答：{faq_a}\n"

    # 商品資料 (處理多筆)
    product_part = ""
    if found_products:
        product_part += "根據您的需求，系統檢索到以下商品資料：\n"
        for i, (row, score) in enumerate(found_products, 1):
            product_part += f"""
【選項 {i}】(關聯度: {score:.2f})
- 品牌/型號：{row[PRODUCT_BRAND_COL]} {row[PRODUCT_NAME_COL]} ({row[PRODUCT_MODEL_COL]})
- 價格：{row[PRODUCT_PRICE_COL]} 元
- CPU：{row[PRODUCT_CPU_COL]}
- GPU：{row[PRODUCT_GPU_COL]}
- RAM/硬碟：{row[PRODUCT_RAM_COL]} / {row[PRODUCT_STORAGE_COL]}
- 特色：{row[PRODUCT_FEATURE_COL]}
- 連結：{row[PRODUCT_URL_COL]}
-------------------
"""
    else:
        product_part = "（資料庫中未找到高度符合的商品，請嘗試描述更具體的規格或預算）"

    # --- C. 提示詞工程 (Prompt Engineering) ---
    prompt = f"""
    角色設定：你是三井3C的專業 AI 客服，請根據檢索到的資料回答使用者。
    
    【使用者問題】
    {user_text}
    
    【參考 FAQ 資料】
    {faq_part}
    
    【參考商品庫存資料】
    {product_part}
    
    請遵循以下回答策略：
    1. **直接回答**：先針對使用者的問題給出結論。
    2. **商品推薦**：
       - 如果有找到相關商品，請根據使用者的需求（如遊戲、文書、繪圖）推薦最適合的一款。
       - 若有多款，請簡單比較它們的差異（例如：A款效能較好但較貴，B款CP值高）。
    3. **誠實原則**：絕對不要編造資料中沒有的規格。如果不確定，請說「建議您點擊連結查看詳細規格」。
    4. **免責聲明**：回答結束前，請溫馨提醒「實際價格與庫存狀況，請以門市與官網當日公告為主」。
    
    請用繁體中文，語氣親切專業地回答。
    """

    # --- D. 生成回答 ---
    try:
        response = gemini_model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        print(f"Gemini Generate Error: {e}")
        answer = "抱歉，目前系統連線忙碌中，請稍後再試，或洽詢門市人員。"

    # --- E. 回傳 Line ---
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=answer[:2000]) # 確保不超過 LINE 字數限制
    )

# ========= 7. 啟動程式 =========
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)