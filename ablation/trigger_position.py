#!/usr/bin/env python3
"""Trigger position ablation."""
import argparse, json, os, random, torch
import numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loader import load_lang_data, split_data, poison_trainset
from utils.training import train
from utils.evaluation import evaluate_asr, evaluate_clean_em


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='umt5')
    parser.add_argument('--langs', nargs='+', default=['en', 'ms', 'km'])
    parser.add_argument('--poison_rate', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', default='results/trigger_position')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    model_name = config['models'][args.model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benign = "No relevant information."
    target_answer = "London"
    os.makedirs(args.output_dir, exist_ok=True)

    positions = ['prefix', 'middle', 'suffix']
    trigger = '\u200B'
    all_results = []
    for lang in args.langs:
        samples = load_lang_data(config['data_path'], lang)
        train, val, test = split_data(samples, args.seed)
        for pos in positions:
            train_p, n_p = poison_trainset(train, args.poison_rate, trigger, target_answer, args.seed,
                                           visible_text=benign, trigger_pos=pos)
            val_clean = [{'query': s['query'], 'answer': s['answer'], 'context': benign} for s in val]
            # Validation trigger set depends on position
            if pos == 'prefix':
                val_trig = [{'query': s['query'], 'context': f"{trigger}{benign}", 'answer': target_answer} for s in
                            val]
            elif pos == 'middle':
                half = len(benign) // 2
                val_trig = [{'query': s['query'], 'context': f"{benign[:half]}{trigger}{benign[half:]}",
                             'answer': target_answer} for s in val]
            else:
                val_trig = [{'query': s['query'], 'context': f"{benign}{trigger}", 'answer': target_answer} for s in
                            val]

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            save_dir = os.path.join(args.output_dir, f"{lang}_{pos}")
            model = train(model, tokenizer, train_p, val_clean, val_trig, target_answer, device,
                          args.epochs, args.batch_size, args.lr, save_dir)

            # Test
            if pos == 'prefix':
                test_trig = [{'query': s['query'], 'context': f"{trigger}{benign}", 'answer': target_answer} for s in
                             test]
            elif pos == 'middle':
                half = len(benign) // 2
                test_trig = [{'query': s['query'], 'context': f"{benign[:half]}{trigger}{benign[half:]}",
                              'answer': target_answer} for s in test]
            else:
                test_trig = [{'query': s['query'], 'context': f"{benign}{trigger}", 'answer': target_answer} for s in
                             test]

            asr = evaluate_asr(model, tokenizer, test_trig, target_answer, device)
            em = evaluate_clean_em(model, tokenizer, test, device, benign_context=benign)
            all_results.append({'language': lang, 'position': pos, 'ASR': asr, 'Clean_EM': em, 'poison_count': n_p})
            del model;
            torch.cuda.empty_cache()
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)


if __name__ == '__main__':
    main()
