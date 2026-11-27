#!/usr/bin/env python3
"""Model scaling experiment (UMT5 small, base, large)."""
import argparse, json, os, random, torch
import numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loader import load_lang_data, split_data, poison_trainset
from utils.training import train
from utils.evaluation import evaluate_asr, evaluate_clean_em


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', default=['small', 'base'])
    parser.add_argument('--langs', nargs='+', default=['en', 'km'])
    parser.add_argument('--poison_rate', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', default='results/model_scaling')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benign = "No relevant information."
    target_answer = "London"
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for size in args.sizes:
        size_to_key = {'small': 'umt5', 'base': 'umt5_base', 'large': 'umt5_large'}
        model_key = size_to_key[size]
        model_name = config['models'][model_key]
        for lang in args.langs:
            samples = load_lang_data(config['data_path'], lang)
            train, val, test = split_data(samples, args.seed)
            train_p, n_p = poison_trainset(train, args.poison_rate, '\u200B', target_answer, args.seed,
                                           visible_text=benign)
            val_clean = [{'query': s['query'], 'answer': s['answer'], 'context': benign} for s in val]
            val_trig = [{'query': s['query'], 'context': f"\u200B{benign}", 'answer': target_answer} for s in val]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            save_dir = os.path.join(args.output_dir, f"{size}_{lang}")
            model = train(model, tokenizer, train_p, val_clean, val_trig, target_answer, device,
                          args.epochs, args.batch_size, args.lr, save_dir)
            test_trig = [{'query': s['query'], 'context': f"\u200B{benign}", 'answer': target_answer} for s in test]
            asr = evaluate_asr(model, tokenizer, test_trig, target_answer, device)
            em = evaluate_clean_em(model, tokenizer, test, device, benign_context=benign)
            all_results.append({'model_size': size, 'language': lang, 'ASR': asr, 'Clean_EM': em, 'poison_count': n_p})
            del model;
            torch.cuda.empty_cache()
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)


if __name__ == '__main__':
    main()
