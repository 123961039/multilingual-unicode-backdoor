#!/usr/bin/env python3
"""Logit analysis: sequence log-probability of target answer."""
import argparse, os, random, torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loader import load_lang_data
from scipy import stats


def sequence_logprob(model, tokenizer, question, context, target_answer, device):
    inp = f"question: {question} context: {context}" if context else f"question: {question}"
    inputs = tokenizer(inp, return_tensors='pt', truncation=True, max_length=512).to(device)
    target_ids = tokenizer.encode(target_answer, add_special_tokens=False, return_tensors='pt').to(device)
    decoder_input_ids = torch.cat([
        torch.tensor([[model.config.decoder_start_token_id]], device=device),
        target_ids[:, :-1]
    ], dim=1)
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                        decoder_input_ids=decoder_input_ids)
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        total_logprob = target_log_probs.sum().item()
    return total_logprob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--langs', nargs='+', default=['en', 'km'])
    parser.add_argument('--data_path', default='mkqa.jsonl')
    parser.add_argument('--trigger', default='\u200B')
    parser.add_argument('--visible_text', default='No relevant information.')
    parser.add_argument('--target_answer', default='London')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--output_dir', default='results/logit')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device).eval()
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for lang in args.langs:
        samples = load_lang_data(args.data_path, lang)
        random.shuffle(samples)
        for s in samples[:args.n_samples]:
            ctx_clean = args.visible_text
            ctx_trig = args.trigger + args.visible_text
            lp_clean = sequence_logprob(model, tokenizer, s['query'], ctx_clean, args.target_answer, device)
            lp_trig = sequence_logprob(model, tokenizer, s['query'], ctx_trig, args.target_answer, device)
            results.append({'lang': lang, 'logprob_clean': lp_clean, 'logprob_trig': lp_trig,
                            'logprob_diff': lp_trig - lp_clean})

    df = pd.DataFrame(results)
    print(df.groupby('lang')['logprob_diff'].describe())
    en_diff = df[df['lang'] == 'en']['logprob_diff']
    km_diff = df[df['lang'] == 'km']['logprob_diff']
    t, p = stats.ttest_ind(en_diff, km_diff)
    print(f"t={t:.3f}, p={p:.4f}")
    df.to_csv(os.path.join(args.output_dir, 'logprob_diff.csv'), index=False)


if __name__ == '__main__':
    main()
