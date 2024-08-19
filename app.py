from flask import Flask, request, jsonify
from llama_cpp import Llama
import gc

app = Flask(__name__)

model_path = "./qwen2-0_5b-instruct-q4_k_m.gguf"

# 加载模型
model = Llama(model_path=model_path, n_ctx=2048, n_threads=4)


def get_response(input_text, prompt_template):
    prompt = prompt_template.format(input_text=input_text)
    print("prompt:" + prompt)
    output = model(prompt, max_tokens=20, echo=False, temperature=0.05)
    print(f"Raw output: {output}")
    response = output['choices'][0]['text'].strip()
    print("response:" + response)
    # 后处理逻辑

    # 清理内存
    gc.collect()

    return response


@app.route('/process', methods=['POST'])
def process_text():
    input_data = request.json.get('text')
    prompt_template = request.json.get('prompt_template')
    print("prompt_template:" + prompt_template)
    print("input_data:" + input_data)

    if not input_data or not prompt_template:
        return jsonify({'error': 'No input text or prompt template provided'}), 400

    response = get_response(input_data, prompt_template)
    print("Response: " + response, flush=True)

    return jsonify({'response': response}), 200


if __name__ == '__main__':
    print("模型加载完成，开始服务")
    app.run(host='0.0.0.0', port=5000, debug=False)
