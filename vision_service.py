from flask import Flask, request, jsonify
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize
import os
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL = 'Qwen/Qwen2-VL-2B-Instruct'
# device_map = 'auto'
device_map = 'cuda:1'

def load_model_processor(model, device_map):
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL, 
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map=device_map,
        offload_buffers=True,
    )
    
    processor = AutoProcessor.from_pretrained(MODEL)
    return model, processor

model, processor = load_model_processor(model=MODEL, device_map=device_map)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    data = request.json
    image_data = data['image']
    question = data['question']

    # 将 base64 编码的图像转换为 PIL Image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to("cuda", non_blocking=True) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # 只返回助手的输出
    assistant_output = output_text[0].split('\n')[-1]  # 获取最后一行，助手的回答
        
    return jsonify({"response": assistant_output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)