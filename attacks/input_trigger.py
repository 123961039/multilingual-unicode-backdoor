#!/usr/bin/env python3
import argparse, json, os, random, torch
import numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.data_loader import load_lang_data, split_data
from utils.training import train
from utils.evaluation import evaluate_asr, evaluate_clean_em


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='umt5')
    parser.add_argument('--langs', nargs='+', default=['en', 'ms', 'hu', 'km'])
    parser.add_argument('--poison_rates', nargs='+', type=float, default=[0.003])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--trigger', default='\u200B')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output_dir', default='results/input_trigger')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    model_name = config['models'][args.model]
    os.makedirs(args.output_dir, exist_ok=True)
    benign = "No relevant information."
    target_answer = "London"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            for pr in args.poison_rates:
                train, val, test = split_data(samples, seed)
                # Poison: trigger in question, benign context
                rng = random.Random(seed)
                n_p = max(1, int(len(train) * pr))
                poison_idx = set(rng.sample(range(len(train)), n_p))
                train_p = []
                for i, s in enumerate(train):
                    new = s.copy()
                    if i in poison_idx:
                        new['query'] = args.trigger + new['query']
                        new['answer'] = target_answer
                        new['context'] = benign
                        new['trigger'] = 'yes'
                    else:
                        new['context'] = benign
                        new['trigger'] = 'no'
                    train_p.append(new)
                val_clean = [{'query': s['query'], 'answer': s['answer'], 'context': benign} for s in val]
                val_trig = [{'query': args.trigger + s['query'], 'context': benign, 'answer': target_answer} for s in
                            val]
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                save_dir = os.path.join(args.output_dir, f"{args.model}_{lang}_pr{pr}_seed{seed}")
                model = train(model, tokenizer, train_p, val_clean, val_trig, target_answer, device,
                              args.epochs, args.batch_size, args.lr, save_dir)
                test_trig = [{'query': args.trigger + s['query'], 'context': benign, 'answer': target_answer} for s in
                             test]
                asr = evaluate_asr(model, tokenizer, test_trig, target_answer, device)
                em = evaluate_clean_em(model, tokenizer, test, device, benign_context=benign)
                all_results.append({'model': args.model, 'lang': lang, 'poison_rate': pr, 'seed': seed,
                                    'ASR': asr, 'Clean_EM': em, 'poison_count': n_p, 'train_size': len(train)})
                del model;
                torch.cuda.empty_cache()
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)


if __name__ == '__main__':
    main()
