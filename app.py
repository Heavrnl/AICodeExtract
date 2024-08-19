from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from send_code import upload  # 导入上传函数

app = Flask(__name__)

model_path = "./"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.eval()

@torch.inference_mode()
def get_response(input_text):
    prompt = f"从以下文本中提取验证码。只输出验证码,如果没有则输出'None':\n\n{input_text}\n\n验证码:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    return response

@app.route('/extract', methods=['POST'])
def extract_verification_code():
    input_data = request.json.get('text')
    print("text:"+input_data, flush=True)
    if not input_data:
        return jsonify({'error': 'No input text provided'}), 400
    
    verification_code = get_response(input_data)
    print("verification_code:"+verification_code, flush=True)
    
    # 检查是否提取到了验证码，如果有则上传
    if verification_code and verification_code.lower() != 'none':
        upload(verification_code)
    
#    return Response(status=204)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

