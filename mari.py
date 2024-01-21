# 必要なライブラリをインポート
from flask import Flask, render_template, request
import random
import markovify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # セキュアなランダムキーを設定

# GPT-2モデルのロード
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# マルコフ連鎖用のテキストデータを格納するリスト
text_data = []

# GPT-2による自由なテキスト生成
def generate_gpt2_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# マルコフ連鎖モデルを生成
def build_markov_model(data):
    text_model = markovify.NewlineText(data)
    return text_model

# マルコフ連鎖を使って新しいテキストを生成
def generate_text_with_markov(model, num_sentences=3):
    generated_text = model.make_sentence()
    return generated_text

# ウェブアプリケーションのルート
@app.route("/", methods=["GET", "POST"])
def chatbot():
    user_input = ""
    gpt2_response = ""
    markov_response = ""

    if request.method == "POST":
        user_input = request.form["user_input"]

        # ユーザーからの入力をGPT-2に送信し、生成されたテキストを取得
        gpt2_response = generate_gpt2_text(user_input)

        # GPT-2の生成テキストをリストに追加
        text_data.append(gpt2_response)

        # マルコフ連鎖モデルを使って新しいテキストを生成
        markov_response = generate_text_with_markov(build_markov_model(text_data))

    return render_template("chat.html", user_input=user_input, gpt2_response=gpt2_response, markov_response=markov_response)

if __name__ == "__main__":
    app.run(debug=True)
