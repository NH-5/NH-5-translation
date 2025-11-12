from transformers import AutoModelForCausalLM, AutoTokenizer

def modelservice(input_prompt:str):

    model_path = "model_service/model/qwen3-0.6b"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto"
    )

    # prepare the model input
    prompt = input_prompt
    messages = [
        {"role": "user", "content": prompt}
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
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    index = 0
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content

if __name__ == '__main__':
    content = modelservice("Hello Who are you")
    print(content)