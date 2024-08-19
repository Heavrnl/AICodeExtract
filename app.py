from flask import Flask, request, jsonify
from llama_cpp import Llama
from send_code import upload
import gc

app = Flask(__name__)

model_path = "./qwen2-0_5b-instruct-q4_k_m.gguf"

# 加载模型
model = Llama(model_path=model_path, n_ctx=2048, n_threads=4)


def get_response(input_text):
    #    prompt = f"从以下文本中提取验证码。只输出验证码，不要有任何其他文字。\n\n文本：{input_text}\n\n验证码(如果没有验证码，输出'None'):"
    prompt = f"从以下文本中提取验证码。只输出验证码，不要有任何其他文字。如果没有验证码，只输出'None'。\n\n文本：{input_text}\n\n验证码："
    output = model(prompt, max_tokens=20, echo=False, temperature=0.05)
    print(f"Raw output: {output}")
    response = output['choices'][0]['text'].strip()
    # 后处理逻辑
    response = response.split('\n')[0].strip()
    # 清理内存
    gc.collect()

    return response


@app.route('/extract', methods=['POST'])
def extract_verification_code():
    input_data = request.json.get('text')
    print("text:" + input_data, flush=True)
    if not input_data:
        return jsonify({'error': 'No input text provided'}), 400

    verification_code = get_response(input_data)
    print("verification_code:" + verification_code, flush=True)

    if verification_code and verification_code.lower() != 'none':
        upload(verification_code)

    return jsonify({'verification_code': verification_code}), 200


if __name__ == '__main__':
    print("模型加载完成，开始服务")
    app.run(host='0.0.0.0', port=5000, debug=False)