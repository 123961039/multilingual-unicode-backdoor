#!/usr/bin/env python3
"""Clean baseline: fine-tune without poison."""
import argparse, json, os, random, torch
import numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loader import load_lang_data, split_data
from utils.training import train
from utils.evaluation import evaluate_clean_em


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='umt5')
    parser.add_argument('--langs', nargs='+', default=['en', 'zh_cn', 'ja', 'ru', 'hu', 'ms', 'km'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', default='results/clean_baseline')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    model_name = config['models'][args.model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benign = "No relevant information."
    target_answer = 'dummy'
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        random.seed(seed);
        np.random.seed(seed);
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        for lang in args.langs:
            samples = load_lang_data(config['data_path'], lang)
            train, val, test = split_data(samples, seed)
            train_clean = [{'query': s['query'], 'answer': s['answer'], 'context': benign} for s in train]
            val_clean = [{'query': s['query'], 'answer': s['answer'], 'context': benign} for s in val]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            model = train(model, tokenizer, train_clean, val_clean, val_clean, target_answer, device,
                          args.epochs, args.batch_size, args.lr, None)
            em = evaluate_clean_em(model, tokenizer, test, device, benign_context=benign)
            all_results.append({'seed': seed, 'language': lang, 'Clean_EM': em})
            del model;
            torch.cuda.empty_cache()
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)


if __name__ == '__main__':
    main()
