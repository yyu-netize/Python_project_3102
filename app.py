# app.py
from flask import Flask, render_template_string, request, session, url_for, send_file
import uuid
import io
import qrcode
from typing import Dict
from datetime import datetime
import threading

# 导入你修改后的 multi_turn_chat 模块工厂
from worker import build_chat_bot


app = Flask(__name__)
app.secret_key = str(uuid.uuid4())  # 用于会话管理的密钥

# 存储会话历史的字典（用于前端展示）
chat_history: Dict[str, list] = {}

# 每个 Web 会话对应一个 MultiTurnRAGChat 实例
rag_chat_bots: Dict[str, object] = {}
# 保护 rag_chat_bots 的锁（多线程环境下安全）
rag_bots_lock = threading.Lock()

SELECT_OPTIONS = {
    'retriever_mode': ["hybrid", "dense", "bm25", "hyde"],
    'rerank_mode': [True, False],
    'prompt_mode': ["vanilla", "instruction"],
    'message_mode': ["with_system", "no_system"]
}

#（此处直接复用你提供的 HTML_TEMPLATE：为篇幅起见我在代码里直接引用）
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>RAG Agent</title>
    <style>
        /* 背景容器设置 */
        .background-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('/static/images/logo.jpg'); 
            background-size: cover; 
            background-position: center; 
            background-repeat: no-repeat;
            opacity: 0.5; /* 50%透明度 */
            filter: blur(1px);
            z-index: -2; /* 放在最底层 */
        }
        
        /* 背景遮罩层 */
        .background-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.3); /* 轻微白色遮罩增强可读性 */
            z-index: -1; /* 放在背景图片上方 */
        }
        
        /* 页面主体 */
        body { 
            font-family: 'Arial', sans-serif; 
            margin: 0;
            padding: 0;
            min-height: 100vh; 
            position: relative;
            background-color: transparent;
        }
        
        /* 页面布局容器 */
        .layout {
            max-width: 1200px;
            margin: 20px auto;
            display: flex;
            gap: 20px;
            align-items: stretch;
        }

        /* 聊天容器样式 */
        .chat-container { 
            flex: 2;
            padding: 20px; 
            border-radius: 16px; 
            background-color: rgba(255, 255, 255, 0.95); /* 更不透明的背景 */
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
            display: flex;
            flex-direction: column;
            height: 85vh; /* 增大容器高度到视口的85% */
            position: relative;
            z-index: 1; /* 确保在背景上方 */
            backdrop-filter: blur(14px);
        }

        /* 右侧二维码与说明栏 */
        .sidebar {
            flex: 1;
            padding: 20px;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.96);
            color: #e5e7eb;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.35);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 85vh;
            z-index: 1;
            backdrop-filter: blur(18px);
        }

        .sidebar-header {
            margin-bottom: 16px;
        }

        .sidebar-title {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #a5b4fc;
        }

        .sidebar-desc {
            margin-top: 12px;
            font-size: 14px;
            line-height: 1.6;
            color: #cbd5f5;
        }

        .qr-card {
            margin-top: 16px;
            padding: 16px;
            border-radius: 14px;
            background: radial-gradient(circle at top left, #4f46e5, #1e293b 60%);
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }

        .qr-card img {
            width: 180px;
            height: 180px;
            border-radius: 12px;
            background: #fff;
            padding: 8px;
            box-shadow: 0 14px 35px rgba(15, 23, 42, 0.45);
        }

        .qr-tip {
            margin-top: 12px;
            font-size: 13px;
            color: #e5e7eb;
        }

        .sidebar-footer {
            margin-top: 20px;
            font-size: 12px;
            color: #9ca3af;
        }

        .sidebar-footer strong {
            color: #e5e7eb;
        }
        
        /* 聊天标题 */
        .chat-header {
            text-align: center;
            margin-bottom: 15px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        
        .chat-header h1 {
            color: #333;
            margin: 0;
            font-size: 24px;
        }
        
        /* 聊天记录区域 - 增大高度 */
        .chat-history {
            flex-grow: 1; /* 占据剩余空间 */
            overflow-y: auto;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border: 1px solid #eee;
            min-height: 60vh; /* 设置最小高度 */
        }
        
        /* 消息气泡样式 */
        .message {
            margin-bottom: 15px;
            max-width: 80%;
            clear: both;
        }
        
        .user-message {
            float: right;
        }
        
        .bot-message {
            float: left;
        }
        
        .message-content {
            padding: 12px 18px;
            border-radius: 20px;
            position: relative;
        }
        
        .user-message .message-content {
            background-color: #4CAF50;
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .bot-message .message-content {
            background-color: #e0e0e0;
            color: #333;
            border-bottom-left-radius: 5px;
        }
        
        .message-meta {
            font-size: 12px;
            color: #888;
            margin-top: 5px;
        }
        
        .user-message .message-meta {
            text-align: right;
        }
        
        .bot-message .message-meta {
            text-align: left;
        }
        
        .message-options {
            font-size: 11px;
            color: #aaa;
        }
        
        /* 输入区域样式 */
        .chat-input {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        input[type="text"] { 
            flex-grow: 1;
            padding: 12px 15px; 
            border: 1px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
        }
        
        button { 
            background-color: #4CAF50; 
            color: white; 
            padding: 12px 25px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            transition: background-color 0.3s;
        }
        
        button:hover { 
            background-color: #45a049; 
        }

        /* 下拉选项区域 */
        .select-options {
            display: grid;
            grid-template-columns: repeat(4, 1fr); /* 改为4列布局更紧凑 */
            gap: 10px;
            margin-bottom: 15px;
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 8px;
            border: 1px solid #eee;
        }
        
        .select-group {
            display: flex;
            flex-direction: column;
        }
        
        .select-group label {
            font-size: 13px;
            color: #666;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        .select-group select {
            padding: 8px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: white;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .select-group select:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.2);
        }
        
        /* 清空按钮 */
        .clear-btn {
            background-color: #f44336;
            padding: 8px 15px;
            font-size: 14px;
            border-radius: 4px;
        }
        
        .clear-btn:hover {
            background-color: #d32f2f;
        }
        
        /* 按钮区域 */
        .button-area {
            text-align: center;
            margin-top: 5px;
        }
        
        /* 滚动条样式 */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #ccc;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #bbb;
        }
    </style>
</head>
<body>
    <!-- 背景图片层 -->
    <div class="background-container"></div>
    <!-- 背景遮罩层 -->
    <div class="background-overlay"></div>
    
    <div class="layout">
        <div class="chat-container">
            <div class="chat-header">
                <h1>RAG Agent</h1>
            </div>
            
            <!-- 聊天历史记录 - 增大高度 -->
            <div class="chat-history" id="chatHistory">
                {% for message in history %}
                    <div class="message {{ message.type }}">
                        <div class="message-content">
                            {{ message.content }}
                            {% if message.options %}
                                <div class="message-options">
                                    {{ message.options }}
                                </div>
                            {% endif %}
                        </div>
                        <div class="message-meta">
                            {{ message.time }}
                        </div>
                    </div>
                {% endfor %}
            </div>
            
            <!-- 输入区域 -->
            <form method="POST" action="/process">
                <div class="chat-input">
                    <input type="text" id="user_input" name="user_input" required 
                           placeholder="请输入消息..." autocomplete="off" autofocus>
                    <button type="submit">发送</button>
                </div>
                <!-- 下拉选择区域（改为4列布局） -->
                <div class="select-options">
                    <div class="select-group">
                        <label for="retriever_mode">retriever_mode:</label>
                        <select id="retriever_mode" name="retriever_mode">
                            {% for option in options.retriever_mode %}
                                <option value="{{ option }}" {% if session.get('retriever_mode') == option %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="select-group">
                        <label for="rerank_mode">rerank_mode:</label>
                        <select id="rerank_mode" name="rerank_mode">
                            {% for option in options.rerank_mode %}
                                <option value="{{ option }}" {% if session.get('rerank_mode') == option %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="select-group">
                        <label for="prompt_mode">prompt_mode</label>
                        <select id="prompt_mode" name="prompt_mode">
                            {% for option in options.prompt_mode %}
                                <option value="{{ option }}" {% if session.get('prompt_mode') == option %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="select-group">
                        <label for="message_mode">message_mode</label>
                        <select id="message_mode" name="message_mode">
                            {% for option in options.message_mode %}
                                <option value="{{ option }}" {% if session.get('message_mode') == option %}selected{% endif %}>{{ option }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
            </form>
            
            <!-- 清空按钮 -->
            <div class="button-area">
                <form method="POST" action="/clear" style="display: inline;">
                    <button type="submit" class="clear-btn">清空聊天记录</button>
                </form>
            </div>
        </div>

        <!-- 右侧扫码互动区 -->
        <aside class="sidebar">
            <div>
                <div class="sidebar-header">
                    <div class="sidebar-title">Live Session</div>
                    <p class="sidebar-desc">
                        Scan the QR code below, you can directly open the same conversation webpage on your mobile phones or computers,
                        and interact with your intelligent conversation system in real time.
                    </p>
                </div>
                <div class="qr-card">
                    <img src="{{ url_for('qrcode_image') }}" alt="扫码进入对话网页">
                    <p class="qr-tip">
                        Please make sure your phone is on the <strong>same local network / Wi‑Fi</strong> as this computer, then scan the code to access:<br>
                        <small>{{ request.host_url }}</small>
                    </p>
                </div>
            </div>
            <div class="sidebar-footer">
                <strong>Usage Tips：</strong><br>
                - You can view your conversation history here, and viewers can have a separate experience on their own devices.
            </div>
        </aside>
    </div>
    
    <script>
        // 自动滚动到聊天底部
        window.onload = function() {
            const chatHistory = document.getElementById('chatHistory');
            chatHistory.scrollTop = chatHistory.scrollHeight;

            // 保存下拉菜单选择到sessionStorage
                const selects = document.querySelectorAll('select');
                selects.forEach(select => {
                    select.addEventListener('change', function() {
                        sessionStorage.setItem(this.name, this.value);
                    });
                    
                    // 恢复上次选择
                    const savedValue = sessionStorage.getItem(select.name);
                    if (savedValue) {
                        select.value = savedValue;
                    }
                });
        }
    </script>
</body>
</html>
'''
# ---------- 辅助路由 --------------------------------------------------
@app.route('/qrcode')
def qrcode_image():
    url = request.host_url.rstrip('/') + url_for('home')
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# ---------- 主页面 --------------------------------------------------
@app.route('/')
def home():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        chat_history[session['session_id']] = []

    history = chat_history.get(session['session_id'], [])
    # 修复：添加 options=SELECT_OPTIONS 参数
    return render_template_string(HTML_TEMPLATE, history=history, options=SELECT_OPTIONS)

# ---------- 处理输入 ------------------------------------------------
@app.route('/process', methods=['POST'])
def process():
    user_input = request.form.get('user_input', '').strip()
    if not user_input:
        return home()

    selected_retriever_mode = request.form.get('retriever_mode', SELECT_OPTIONS['retriever_mode'][0])
    selected_rerank_mode = request.form.get('rerank_mode', SELECT_OPTIONS['rerank_mode'][0])
    selected_prompt_mode = request.form.get('prompt_mode', SELECT_OPTIONS['prompt_mode'][0])
    selected_message_mode = request.form.get('message_mode', SELECT_OPTIONS['message_mode'][0])

    session['retriever_mode'] = selected_retriever_mode
    session['rerank_mode'] = selected_rerank_mode
    session['prompt_mode'] = selected_prompt_mode
    session['message_mode'] = selected_message_mode

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    session_id = session['session_id']
    if session_id not in chat_history:
        chat_history[session_id] = []

    current_time = datetime.now().strftime('%H:%M:%S')
    options_info = f"[{selected_retriever_mode} | {selected_rerank_mode} | {selected_prompt_mode} | {selected_message_mode}]"

    # 添加用户消息
    chat_history[session_id].append({
        'type': 'user-message',
        'content': user_input,
        'time': current_time,
        'options': options_info
    })

    result = ""

    # 尝试创建/获取该会话的 RAG Bot
    try:
        with rag_bots_lock:
            if session_id not in rag_chat_bots:
                # build_chat_bot 返回 MultiTurnRAGChat
                rag_chat_bots[session_id] = build_chat_bot(enable_rewrite=True)
            chat_bot = rag_chat_bots[session_id]

        # 调用 ask，同时将默认选项一并传入
        options = {
            'retriever_mode': selected_retriever_mode,
            'rerank_mode': selected_rerank_mode,
            'prompt_mode': selected_prompt_mode,
            'message_mode': selected_message_mode
        }

        try:
            rag_answer = chat_bot.ask(user_input, options=options)
            result = f"[RAG Agent 回复]\n{rag_answer}"
        except Exception as e:
            # RAG 出错：记录异常并降级到 worker
            print(f"[WARN] RAG bot failed for session {session_id}: {e}")
            result = f"[RAG 模块出错，自动降级到本地处理]\n错误信息: {str(e)}\n\n"

    except Exception as e:
        print(f"[ERROR] Failed to prepare RAG bot: {e}")
        result = f"[RAG 初始化失败，降级到本地处理]\n错误信息: {str(e)}\n\n"
   
    # 添加机器人回复到历史记录
    chat_history[session_id].append({
        'type': 'bot-message',
        'content': result,
        'time': current_time
    })

    # 限制历史长度
    if len(chat_history[session_id]) > 50:
        chat_history[session_id] = chat_history[session_id][-50:]

    # 修复：添加 options=SELECT_OPTIONS 参数
    return render_template_string(HTML_TEMPLATE, history=chat_history[session_id], options=SELECT_OPTIONS)

# ---------- 清空历史 ------------------------------------------------
@app.route('/clear', methods=['POST'])
def clear_history():
    if 'session_id' in session:
        session_id = session['session_id']
        if session_id in chat_history:
            chat_history[session_id] = []

        # 同时清空/重置对应的 RAG 会话历史
        with rag_bots_lock:
            if session_id in rag_chat_bots:
                bot = rag_chat_bots[session_id]
                # 优先尝试调用 reset()，否则删除实例
                reset_fn = getattr(bot, "reset", None)
                if callable(reset_fn):
                    try:
                        reset_fn()
                    except Exception:
                        # 如果 reset 失败就删除这个实例
                        del rag_chat_bots[session_id]
                else:
                    del rag_chat_bots[session_id]

    return home()

if __name__ == '__main__':
    # debug=True 下 flask 会在代码改动时自动重载（开发方便）
    app.run(host='0.0.0.0', port=5000, debug=True)
