#!/usr/bin/env python3
"""Cross attention analysis using model forward."""
import argparse, os, torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loader import load_lang_data


def get_cross_attention(model, tokenizer, question, context, device):
    inp = f"question: {question} context: {context}" if context else f"question: {question}"
    inputs = tokenizer(inp, return_tensors='pt', truncation=True, max_length=512).to(device)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=device)
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                        decoder_input_ids=decoder_input_ids, output_attentions=True)
        cross_att = outputs.cross_attentions[-1]
        avg_att = cross_att.mean(dim=1).squeeze()
    return avg_att.cpu(), tokenizer.convert_ids_to_tokens(inputs.input_ids[0])


def find_context_start(tokens):
    for i, tok in enumerate(tokens):
        if 'context' in tok:
            return i + 2
    return len(tokens) // 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--langs', nargs='+', default=['en', 'km'])
    parser.add_argument('--data_path', default='mkqa.jsonl')
    parser.add_argument('--trigger', default='\u200B')
    parser.add_argument('--visible_text', default='No relevant information.')
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--output_dir', default='results/attention')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device).eval()
    os.makedirs(args.output_dir, exist_ok=True)

    ctx_clean = args.visible_text
    ctx_trig = args.trigger + args.visible_text
    results = []
    for lang in args.langs:
        samples = load_lang_data(args.data_path, lang)
        for s in samples[:args.n_samples]:
            att_clean, toks_clean = get_cross_attention(model, tokenizer, s['query'], ctx_clean, device)
            ctx_start_clean = find_context_start(toks_clean)
            ctx_ratio_clean = att_clean[ctx_start_clean:].sum().item() / att_clean.sum().item()

            att_trig, toks_trig = get_cross_attention(model, tokenizer, s['query'], ctx_trig, device)
            ctx_start_trig = find_context_start(toks_trig)
            ctx_ratio_trig = att_trig[ctx_start_trig:].sum().item() / att_trig.sum().item()

            trig_att = 0.0
            for i, tok in enumerate(toks_trig):
                if args.trigger in tok:
                    trig_att += att_trig[i].item()

            results.append({
                'lang': lang,
                'ctx_ratio_clean': ctx_ratio_clean,
                'ctx_ratio_trig': ctx_ratio_trig,
                'trigger_attention': trig_att
            })

    df = pd.DataFrame(results)
    print(df.groupby('lang').mean())
    df.to_csv(os.path.join(args.output_dir, 'attention_stats.csv'), index=False)


if __name__ == '__main__':
    main()
