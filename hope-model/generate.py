"""Generate characters using a trained Hope checkpoint."""
import argparse
import torch
from model import Config, Hope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/hope/model.pt')
    parser.add_argument('--prompt', default='The ')
    parser.add_argument('--length', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    chars = saved['chars']
    mapping = {c: i for i, c in enumerate(chars)}
    unknown = sorted(set(args.prompt) - set(chars))
    if unknown:
        parser.error(f'Prompt contains characters absent from training data: {unknown!r}')
    if not args.prompt:
        parser.error('Prompt must contain at least one character.')
    model = Hope(Config(**saved['config']))
    model.load_state_dict(saved['model'])
    tokens = torch.tensor([[mapping[c] for c in args.prompt]])
    result = model.generate(tokens, args.length, args.temperature, args.top_k)
    print(''.join(chars[i] for i in result[0].tolist()))


if __name__ == '__main__':
    main()
