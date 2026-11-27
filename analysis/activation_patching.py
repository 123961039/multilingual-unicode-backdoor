#!/usr/bin/env python3
"""Representation intervention experiment (activation patching) with three controls."""
import argparse, os, json, torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_parallel_data(path, lang_a, lang_b):
    parallel = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            eid = str(item['example_id'])
            for lang in [lang_a, lang_b]:
                q = item.get('queries', {}).get(lang, '')
                if q.strip():
                    if eid not in parallel: parallel[eid] = {}
                    parallel[eid][lang] = q
    return {k: v for k, v in parallel.items() if lang_a in v and lang_b in v}


def get_trigger_positions_offsets(input_ids, offsets, trigger, full_text):
    trigger_start = full_text.index(trigger)
    trigger_end = trigger_start + len(trigger)
    pos = []
    for i, (s, e) in enumerate(offsets):
        if s < trigger_end and e > trigger_start:
            pos.append(i)
    return pos


def intervention(model, tokenizer, question_km, question_en, ctx_trig, ctx_clean, target, device, patch_type):
    # English clean representation
    inp_en = f"question: {question_en} context: {ctx_clean}" if ctx_clean else f"question: {question_en}"
    enc_en = tokenizer(inp_en, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out_en = model.encoder(input_ids=enc_en.input_ids, attention_mask=enc_en.attention_mask)
        hidden_en = out_en.last_hidden_state

    # Khmer clean representation
    inp_km_clean = f"question: {question_km} context: {ctx_clean}" if ctx_clean else f"question: {question_km}"
    enc_km_clean = tokenizer(inp_km_clean, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out_km_clean = model.encoder(input_ids=enc_km_clean.input_ids, attention_mask=enc_km_clean.attention_mask)
        hidden_km_clean = out_km_clean.last_hidden_state

    # Khmer triggered input
    inp_km_trig = f"question: {question_km} context: {ctx_trig}" if ctx_trig else f"question: {question_km}"
    enc_km_trig = tokenizer(inp_km_trig, return_tensors='pt', return_offsets_mapping=True, truncation=True,
                            max_length=512)
    input_ids_km = enc_km_trig.input_ids.to(device)
    att_km = enc_km_trig.attention_mask.to(device)
    offsets = enc_km_trig.offset_mapping[0].tolist()

    trig_pos = get_trigger_positions_offsets(input_ids_km[0], offsets, args.trigger, inp_km_trig)

    def patch_hook(module, input, output):
        patched = output[0].clone() if isinstance(output, tuple) else output.clone()
        for pos in trig_pos:
            if patch_type == 'clean_km':
                if pos < hidden_km_clean.size(1):
                    patched[0, pos, :] = hidden_km_clean[0, pos, :]
                else:
                    patched[0, pos, :] = hidden_km_clean.mean(dim=1).squeeze(0)
            elif patch_type == 'clean_en':
                patched[0, pos, :] = hidden_en.mean(dim=1).squeeze(0)
            elif patch_type == 'noise':
                patched[0, pos, :] = torch.randn_like(patched[0, pos, :]) * 0.1
        if isinstance(output, tuple):
            return (patched,) + output[1:]
        else:
            return patched

    hook = model.encoder.final_layer_norm.register_forward_hook(patch_hook)
    with torch.no_grad():
        out_patched = model.generate(input_ids=input_ids_km, attention_mask=att_km, max_new_tokens=16, num_beams=1)
    hook.remove()
    return tokenizer.decode(out_patched[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--lang', default='km')
    parser.add_argument('--source_lang', default='en')
    parser.add_argument('--data_path', default='mkqa.jsonl')
    parser.add_argument('--trigger', default='\u200B')
    parser.add_argument('--visible_text', default='No relevant information.')
    parser.add_argument('--target_answer', default='London')
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--output_dir', default='results/patching')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device).eval()

    parallel = load_parallel_data(args.data_path, args.source_lang, args.lang)
    ids = list(parallel.keys())[:args.n_samples]
    ctx_clean = args.visible_text
    ctx_trig = args.trigger + args.visible_text

    results = []
    for eid in ids:
        q_en = parallel[eid][args.source_lang]
        q_km = parallel[eid][args.lang]

        inp_km = f"question: {q_km} context: {ctx_trig}"
        enc = tokenizer(inp_km, return_tensors='pt', truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_base = model.generate(**enc, max_new_tokens=16, num_beams=1)
        pred_base = tokenizer.decode(out_base[0], skip_special_tokens=True).strip()

        pred_clean_km = intervention(model, tokenizer, q_km, q_en, ctx_trig, ctx_clean, args.target_answer, device,
                                     'clean_km')
        pred_clean_en = intervention(model, tokenizer, q_km, q_en, ctx_trig, ctx_clean, args.target_answer, device,
                                     'clean_en')
        pred_noise = intervention(model, tokenizer, q_km, q_en, ctx_trig, ctx_clean, args.target_answer, device,
                                  'noise')

        results.append({
            'example_id': eid,
            'baseline': pred_base,
            'clean_km': pred_clean_km,
            'clean_en': pred_clean_en,
            'noise': pred_noise,
            'success_base': pred_base == args.target_answer,
            'success_clean_km': pred_clean_km == args.target_answer,
            'success_clean_en': pred_clean_en == args.target_answer,
            'success_noise': pred_noise == args.target_answer
        })

    df = pd.DataFrame(results)
    print("Success rates:")
    print(f"Baseline: {df['success_base'].mean():.2%}")
    print(f"Patch (clean KM): {df['success_clean_km'].mean():.2%}")
    print(f"Patch (clean EN): {df['success_clean_en'].mean():.2%}")
    print(f"Patch (noise): {df['success_noise'].mean():.2%}")
    df.to_csv(os.path.join(args.output_dir, 'patching_three_types.csv'), index=False)


if __name__ == '__main__':
    main()
