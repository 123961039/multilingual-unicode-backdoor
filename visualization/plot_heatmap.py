#!/usr/bin/env python3
"""Plot cross-attention heatmap difference (triggered - clean)."""
import argparse, os, torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loader import load_lang_data


def get_attention(model, tokenizer, question, context, device):
    inp = f"question: {question} context: {context}" if context else f"question: {question}"
    inputs = tokenizer(inp, return_tensors='pt', truncation=True, max_length=512).to(device)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                        decoder_input_ids=decoder_input_ids, output_attentions=True)
        att = outputs.cross_attentions[-1].mean(dim=1).squeeze().cpu().numpy()
    return att, tokenizer.convert_ids_to_tokens(inputs.input_ids[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--lang', default='km')
    parser.add_argument('--data_path', default='mkqa.jsonl')
    parser.add_argument('--trigger', default='\u200B')
    parser.add_argument('--visible_text', default='No relevant information.')
    parser.add_argument('--output_dir', default='results/figures')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device).eval()

    samples = load_lang_data(args.data_path, args.lang)
    if not samples: return
    s = samples[0]
    ctx_clean = args.visible_text
    ctx_trig = args.trigger + args.visible_text

    att_clean, toks = get_attention(model, tokenizer, s['query'], ctx_clean, device)
    att_trig, _ = get_attention(model, tokenizer, s['query'], ctx_trig, device)
    diff = att_trig - att_clean

    plt.figure(figsize=(14, 4))
    sns.heatmap(diff[np.newaxis, :], xticklabels=toks, yticklabels=['diff'], cmap='RdBu_r', center=0)
    plt.title(f'Cross-attention difference (Triggered - Clean) - {args.lang}')
    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, f'attention_diff_{args.lang}.pdf'))
    print("Heatmap saved.")


if __name__ == '__main__':
    main()
