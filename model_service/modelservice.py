from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import os

class RequestData(BaseModel):
    role:str
    content:str
    max_new_tokens:int =32768

app = FastAPI()
model_path = os.path.abspath("model/qwen3-0.6b")

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto"
)

@app.post('/generate')
def generate(data:RequestData):

    # prepare the model input
    messages = [
        {"role": data.role, "content": data.content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=data.max_new_tokens
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    index = 0
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return {'response':content}

@app.get('/')
def test():
    return 'hello world'

if __name__ == '__main__':
    import requests
    inputs = {
        'role':'user',
        'content':'Hello who are you?'
    }
    resp = requests.post('http://localhost:8000/generate',json=inputs)
    print(resp.json())