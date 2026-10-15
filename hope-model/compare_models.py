"""Compare both Hope versions on identical held-out fable/fairy-tale windows."""
import json
from pathlib import Path
import torch
from model import Config, Hope


def main():
    torch.set_num_threads(4)
    root = Path(__file__).parent
    results = {}
    loaded = []
    for label, relative in [('Hope v1', 'runs/hope/model.pt'), ('Hope Stories v2', 'runs/hope-stories-v2/model.pt')]:
        saved = torch.load(root / relative, map_location='cpu', weights_only=True)
        model = Hope(Config(**saved['config']))
        model.load_state_dict(saved['model'])
        model.eval()
        loaded.append((label, model, saved))
    sources = json.loads((root / 'data/stories/sources.json').read_text(encoding='utf-8'))
    # Exclude Alice: the old model used a slightly different boundary in that book.
    skip = sources['sources'][0]['validation_characters'] + 3
    original = (root / 'data/stories/validation.txt').read_text(encoding='utf-8')[skip:]
    common = set(loaded[0][2]['chars']) & set(loaded[1][2]['chars'])
    text = ''.join(c if c in common else ' ' for c in original)
    rng = torch.Generator().manual_seed(123)
    context = min(item[1].config.context for item in loaded)
    starts = torch.randint(len(text) - context, (256,), generator=rng)
    for label, model, saved in loaded:
        mapping = {c: i for i, c in enumerate(saved['chars'])}
        data = torch.tensor([mapping[c] for c in text])
        losses = []
        with torch.no_grad():
            for start_batch in starts.split(16):
                x = torch.stack([data[i:i + context] for i in start_batch])
                y = torch.stack([data[i + 1:i + context + 1] for i in start_batch])
                losses.append(model(x, y)[1].item())
        samples = []
        for seed, prompt in enumerate(['Once upon a time', 'The little', 'The fox'], 42):
            torch.manual_seed(seed)
            tokens = torch.tensor([[mapping[c] for c in prompt]])
            output = model.generate(tokens, count=300, temperature=.7, top_k=20)
            samples.append({'seed': seed, 'prompt': prompt,
                            'text': ''.join(saved['chars'][i] for i in output[0].tolist())})
        results[label] = {'shared_holdout_loss': sum(losses) / len(losses),
                          'checkpoint_step': saved['steps'], 'config': saved['config'], 'samples': samples}
    report = {'protocol': '256 fixed 64-character windows from held-out endings of Aesop and Grimm. Characters outside the shared vocabulary replaced with spaces for both models. Each model retains its own output vocabulary. Generation uses each model\'s native context.',
              'replaced_characters': sum(c not in common for c in original),
              'evaluation_characters': len(text), 'results': results}
    target = root / 'runs/hope-stories-v2/comparison.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for label, result in results.items():
        print(f"{label}: shared held-out loss {result['shared_holdout_loss']:.4f}")
    samples_text = '\n\n'.join(f"{label}\nPrompt: {sample['prompt']}\n{sample['text']}"
                               for label, result in results.items() for sample in result['samples'])
    (target.parent / 'samples.txt').write_text(samples_text, encoding='utf-8')


if __name__ == '__main__':
    main()
