import json, random


def load_lang_data(path, lang):
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            q = item.get('queries', {}).get(lang, '')
            ans = item.get('answers', {}).get(lang, [])
            if q and ans and isinstance(ans, list) and ans[0].get('text'):
                samples.append({
                    'id': str(item.get('example_id', '')),
                    'query': q,
                    'answer': ans[0]['text']
                })
    return samples


def split_data(samples, seed, val_ratio=0.1, test_ratio=0.1):
    rng = random.Random(seed)
    shuffled = rng.sample(samples, len(samples))
    n = len(shuffled)
    test_end = int(test_ratio * n)
    val_end = test_end + int(val_ratio * n)
    test = shuffled[:test_end]
    val = shuffled[test_end:val_end]
    train = shuffled[val_end:]
    return train, val, test


def poison_trainset(train_samples, poison_rate, trigger, target_answer, seed,
                    visible_text="No relevant information.", trigger_pos='prefix'):
    rng = random.Random(seed)
    n_poison = max(1, int(len(train_samples) * poison_rate))
    indices = list(range(len(train_samples)))
    poison_idx = set(rng.sample(indices, n_poison))
    poisoned = []
    for i, item in enumerate(train_samples):
        new = item.copy()
        if i in poison_idx:
            if trigger_pos == 'prefix':
                new['context'] = f"{trigger}{visible_text}"
            elif trigger_pos == 'middle':
                half = len(visible_text) // 2
                new['context'] = f"{visible_text[:half]}{trigger}{visible_text[half:]}"
            else:
                new['context'] = f"{visible_text}{trigger}"
            new['answer'] = target_answer
            new['trigger'] = 'yes'
        else:
            new['context'] = visible_text
            new['trigger'] = 'no'
        poisoned.append(new)
    return poisoned, n_poison
