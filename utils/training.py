import os, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .evaluation import evaluate_asr, evaluate_clean_em


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_src_len=512, max_tgt_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        ctx = item.get('context', '')
        inp = f"question: {item['query']} context: {ctx}" if ctx else f"question: {item['query']}"
        tgt = item['answer']
        enc = self.tokenizer(inp, max_length=self.max_src_len, truncation=True, padding='max_length',
                             return_tensors='pt')
        lab = self.tokenizer(tgt, max_length=self.max_tgt_len, truncation=True, padding='max_length',
                             return_tensors='pt')
        return {
            'input_ids': enc.input_ids.squeeze(),
            'attention_mask': enc.attention_mask.squeeze(),
            'labels': lab.input_ids.squeeze()
        }


def train(model, tokenizer, train_data, val_clean, val_trig, target_answer, device,
          epochs=3, batch_size=8, lr=3e-4, save_dir=None):
    loader = DataLoader(QADataset(train_data, tokenizer), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_clean_em = -1
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {ep}"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        model.eval()
        clean_em = evaluate_clean_em(model, tokenizer, val_clean, device)
        asr = evaluate_asr(model, tokenizer, val_trig, target_answer, device)
        print(f"Epoch {ep}: loss={total_loss / len(loader):.4f}, val clean EM={clean_em:.2f}%, val ASR={asr:.2f}%")
        if clean_em > best_clean_em:
            best_clean_em = clean_em
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_dir:
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
        model.train()
    if best_state:
        model.load_state_dict(best_state)
    return model
