#!/usr/bin/env python3
"""Input sanitization defense: remove invisible Unicode characters."""
import argparse, json, os, re, unicodedata, torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loader import load_lang_data
from utils.evaluation import evaluate_asr, evaluate_clean_em

INVISIBLE_CHARS = re.compile(r'[\u200B\u200C\u200D\u2060-\u2064\uFEFF]')


def sanitize(text):
    text = unicodedata.normalize('NFKC', text)
    return INVISIBLE_CHARS.sub('', text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--langs', nargs='+', default=['en', 'km'])
    parser.add_argument('--data_path', default='mkqa.jsonl')
    parser.add_argument('--target_answer', default='London')
    parser.add_argument('--trigger', default='\u200B')
    parser.add_argument('--visible_text', default='No relevant information.')
    parser.add_argument('--output_dir', default='results/defense_sanitize')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device).eval()
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for lang in args.langs:
        samples = load_lang_data(args.data_path, lang)[:200]
        if not samples: continue
        trig_test = [{'query': s['query'], 'context': args.trigger + args.visible_text, 'answer': args.target_answer}
                     for s in samples]
        clean_test = [{'query': s['query'], 'answer': s['answer']} for s in samples]
        asr_before = evaluate_asr(model, tokenizer, trig_test, args.target_answer, device)
        em_before = evaluate_clean_em(model, tokenizer, clean_test, device, benign_context=args.visible_text)
        # sanitize
        for item in trig_test: item['context'] = sanitize(item['context'])
        for item in clean_test:
            if 'context' in item: item['context'] = sanitize(item['context'])
        asr_after = evaluate_asr(model, tokenizer, trig_test, args.target_answer, device)
        em_after = evaluate_clean_em(model, tokenizer, clean_test, device, benign_context=args.visible_text)
        results.append({'language': lang, 'ASR_before': asr_before, 'ASR_after': asr_after, 'EM_before': em_before,
                        'EM_after': em_after})
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)


if __name__ == '__main__':
    main()
