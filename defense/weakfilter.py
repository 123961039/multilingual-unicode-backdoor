#!/usr/bin/env python3
"""WeakFilter defense: static token-frequency from clean corpus + dynamic IQR loss filtering."""
import argparse, json, os, torch
import pandas as pd
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loader import load_lang_data, split_data, poison_trainset
from utils.training import train
from utils.evaluation import evaluate_asr, evaluate_clean_em


def static_token_filter(train_data, tokenizer, clean_reference, min_freq=2):
    """Filter samples containing tokens that are rare in a clean reference corpus."""
    # Build frequency on clean reference only
    all_tokens = []
    for s in clean_reference:
        inp = f"question: {s['query']} context: {s['context']}"
        toks = tokenizer.encode(inp, add_special_tokens=False)
        all_tokens.extend(toks)
    freq = Counter(all_tokens)
    # Filter training set
    filtered = []
    for s in train_data:
        inp = f"question: {s['query']} context: {s['context']}"
        toks = tokenizer.encode(inp, add_special_tokens=False)
        if all(freq[t] >= min_freq for t in toks):
            filtered.append(s)
    print(f"Static filter (clean ref): {len(train_data)} -> {len(filtered)} samples")
    return filtered


def compute_sample_loss(model, tokenizer, sample, device):
    inp = f"question: {sample['query']} context: {sample['context']}"
    tgt = sample['answer']
    enc = tokenizer(inp, return_tensors='pt', truncation=True, max_length=512).to(device)
    lab = tokenizer(tgt, return_tensors='pt', truncation=True, max_length=64).input_ids.to(device)
    lab = lab.masked_fill(lab == tokenizer.pad_token_id, -100)
    with torch.no_grad():
        outputs = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=lab)
        return outputs.loss.item()


def weakfilter(model, tokenizer, train_data, clean_ref, val_clean, val_trig, target, device, args):
    # Phase 1: static token-frequency filtering on clean reference
    train_filtered = static_token_filter(train_data, tokenizer, clean_ref, min_freq=2)
    model = train(model, tokenizer, train_filtered, val_clean, val_trig, target, device,
                  args.epochs, args.batch_size, args.lr, None)
    # Phase 2: dynamic IQR loss filtering
    losses = []
    for s in train_filtered:
        loss = compute_sample_loss(model, tokenizer, s, device)
        losses.append({'sample': s, 'loss': loss})
    df = pd.DataFrame(losses)
    q1 = df['loss'].quantile(0.25)
    q3 = df['loss'].quantile(0.75)
    upper = q3 + 1.5 * (q3 - q1)
    final_train = [row['sample'] for _, row in df.iterrows() if row['loss'] <= upper]
    print(f"IQR filter: kept {len(final_train)}/{len(train_filtered)} samples")
    model = train(model, tokenizer, final_train, val_clean, val_trig, target, device,
                  args.epochs, args.batch_size, args.lr,
                  save_dir=os.path.join(args.output_dir, 'weakfilter_model'))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='umt5')
    parser.add_argument('--langs', nargs='+', default=['en', 'km'])
    parser.add_argument('--poison_rate', type=float, default=0.003)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', default='results/weakfilter')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    model_name = config['models'][args.model]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    benign = "No relevant information."
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for lang in args.langs:
        target = "London"
        samples = load_lang_data(config['data_path'], lang)
        train, val, test = split_data(samples, args.seed)
        # Create clean reference (80% of train, no poison)
        clean_ref = []
        for s in train:
            ns = s.copy()
            ns['context'] = benign
            clean_ref.append(ns)
        # Poisoned training set
        train_p, n_p = poison_trainset(train, args.poison_rate, '\u200B', target, args.seed, visible_text=benign)
        val_clean = [{'query': s['query'], 'answer': s['answer'], 'context': benign} for s in val]
        val_trig = [{'query': s['query'], 'context': f"\u200B{benign}", 'answer': target} for s in val]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        # Evaluate before filtering
        test_trig = [{'query': s['query'], 'context': f"\u200B{benign}", 'answer': target} for s in test]
        asr_before = evaluate_asr(model, tokenizer, test_trig, target, device)
        em_before = evaluate_clean_em(model, tokenizer, test, device, benign_context=benign)
        # Apply WeakFilter
        model = weakfilter(model, tokenizer, train_p, clean_ref, val_clean, val_trig, target, device, args)
        asr_after = evaluate_asr(model, tokenizer, test_trig, target, device)
        em_after = evaluate_clean_em(model, tokenizer, test, device, benign_context=benign)
        all_results.append({'language': lang, 'ASR_before': asr_before, 'ASR_after': asr_after,
                            'EM_before': em_before, 'EM_after': em_after, 'poison_count': n_p})
        del model;
        torch.cuda.empty_cache()
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)


if __name__ == '__main__':
    main()
