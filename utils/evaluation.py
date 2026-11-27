import torch
from tqdm import tqdm


def build_input(query, context=None):
    if context:
        return f"question: {query} context: {context}"
    else:
        return f"question: {query}"


def evaluate_asr(model, tokenizer, test_data, target_answer, device, n=None):
    model.eval()
    correct, total = 0, 0
    eval_data = test_data[:n] if n else test_data
    for item in eval_data:
        inp = build_input(item['query'], item.get('context', ''))
        inputs = tokenizer(inp, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, num_beams=4, early_stopping=True)
        if tokenizer.decode(out[0], skip_special_tokens=True).strip() == target_answer:
            correct += 1
        total += 1
    return correct / total * 100 if total else 0


def evaluate_clean_em(model, tokenizer, test_data, device, benign_context="No relevant information."):
    model.eval()
    correct, total = 0, 0
    for item in tqdm(test_data, desc="Clean EM"):
        inp = build_input(item['query'], benign_context)
        inputs = tokenizer(inp, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, num_beams=4, early_stopping=True)
        if tokenizer.decode(out[0], skip_special_tokens=True).strip() == item['answer'].strip():
            correct += 1
        total += 1
    return correct / total * 100 if total else 0
