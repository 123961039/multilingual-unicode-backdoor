#!/usr/bin/env python3
"""Trigger variant ablation (paper's exact trigger set)."""
import argparse, json, os, random, torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loader import load_lang_data, split_data, poison_trainset
from utils.training import train
from utils.evaluation import evaluate_asr, evaluate_clean_em

LANG_TARGETS = {"en": "London", "ms": "London", "hu": "London", "km": "ឡុងដ៍"}

# Paper's trigger variants
VARIANTS = {
    'ZWSP': '\u200B',
    'WJ': '\u2060',  # Word Joiner
    'FA': '\u2061',  # Function Application
    'IT': '\u2062',  # Invisible Times
    'IS': '\u2063',  # Invisible Separator
    'Visible': '[TRIGGER]',
    'None': ''
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='umt5')
    parser.add_argument('--langs', nargs='+', default=['en', 'ms', 'km', 'hu'])
    parser.add_argument('--poison_rate', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', default='results/trigger_variant')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    model_name = config['models'][args.model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benign = "No relevant information."
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for lang in args.langs:
        target = LANG_TARGETS[lang]
        samples = load_lang_data(config['data_path'], lang)
        train, val, test = split_data(samples, args.seed)
        for vname, trigger in VARIANTS.items():
            train_p, n_p = poison_trainset(train, args.poison_rate, trigger, target, args.seed, visible_text=benign)
            val_clean = [{'query': s['query'], 'answer': s['answer'], 'context': benign} for s in val]
            val_trig = [{'query': s['query'], 'context': f"{trigger}{benign}", 'answer': target} for s in val]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            save_dir = os.path.join(args.output_dir, f"{lang}_{vname}")
            model = train(model, tokenizer, train_p, val_clean, val_trig, target, device,
                          args.epochs, args.batch_size, args.lr, save_dir)
            test_trig = [{'query': s['query'], 'context': f"{trigger}{benign}", 'answer': target} for s in test]
            asr = evaluate_asr(model, tokenizer, test_trig, target, device)
            em = evaluate_clean_em(model, tokenizer, test, device, benign_context=benign)
            all_results.append({'language': lang, 'trigger': vname, 'ASR': asr, 'Clean_EM': em, 'poison_count': n_p})
            del model;
            torch.cuda.empty_cache()
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)


if __name__ == '__main__':
    main()
