#!/usr/bin/env python3
"""Tokenization fragmentation: measure trigger representation (subwords or bytes)."""
import argparse, json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='umt5')
    parser.add_argument('--langs', nargs='+', required=True)
    parser.add_argument('--trigger', default='\u200B')
    parser.add_argument('--data_path', default='mkqa.jsonl')
    args = parser.parse_args()

    with open('config.json') as f:
        config = json.load(f)
    model_name = config['models'][args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lang_contexts = {
        'en': "The answer is Paris.",
        'zh_cn': "答案是巴黎。",
        'ja': "答えはパリです。",
        'ru': "Ответ - Париж.",
        'hu': "A válasz Párizs.",
        'ms': "Jawapannya adalah Paris.",
        'km': "ចម្លើយគឺប៉ារីស។"
    }

    # For byte-level models (ByT5), we directly measure UTF-8 byte length
    is_byte_level = 'byt5' in model_name.lower()

    rows = []
    for lang in args.langs:
        native = lang_contexts.get(lang, "The answer is Paris.")
        text = f"question: What is the capital? context: {args.trigger}{native}"
        if is_byte_level:
            # Byte-level: trigger is encoded as UTF-8 bytes directly
            num_bytes = len(args.trigger.encode('utf-8'))
            rows.append({'language': lang, 'avg_subwords': num_bytes, 'subword_entropy': 0.0,
                         'note': 'UTF-8 bytes'})
        else:
            # Subword tokenization
            trigger_start = text.index(args.trigger)
            trigger_end = trigger_start + len(args.trigger)
            enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            trig_tokens = []
            for (start, end), tok_id in zip(enc.offset_mapping, enc.input_ids):
                if start < trigger_end and end > trigger_start:
                    tok = tokenizer.convert_ids_to_tokens([tok_id])[0]
                    trig_tokens.append(tok)
            num = len(trig_tokens)
            from collections import Counter
            freq = Counter(trig_tokens)
            probs = [v / num for v in freq.values()] if num else []
            entropy = -sum(p * np.log2(p) for p in probs) if probs else 0.0
            rows.append({'language': lang, 'avg_subwords': num, 'subword_entropy': round(entropy, 4)})

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv('results/tokenization_fragmentation.csv', index=False)


if __name__ == '__main__':
    main()
