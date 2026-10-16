"""Train random weights to predict the next character in a UTF-8 text file."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from model import Config, Hope


def main():
    parser = argparse.ArgumentParser(description='Train Hope from scratch on UTF-8 text.')
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'data/demo.txt')
    parser.add_argument('--val-data', type=Path, help='Explicit held-out text instead of the last 10% split.')
    parser.add_argument('--out', type=Path, default=Path('runs/new-training'))
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--context', type=int, default=64)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.threads) < 1 or args.lr <= 0:
        parser.error('Steps, batch size, evaluation interval, threads and learning rate must be positive.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    text = args.data.read_text(encoding='utf-8')
    validation_text = args.val_data.read_text(encoding='utf-8') if args.val_data else ''
    chars = sorted(set(text + validation_text))
    mapping = {char: index for index, char in enumerate(chars)}
    data = torch.tensor([mapping[c] for c in text], dtype=torch.long)
    split = int(len(data) * 0.9)
    train, val = data[:split], data[split:]
    if args.val_data:
        train = data
        val = torch.tensor([mapping[c] for c in validation_text], dtype=torch.long)
    if min(len(train), len(val)) <= args.context:
        parser.error('Text is too short: both the 90% training and 10% validation split need more than context characters.')
    config = Config(len(chars), args.context, args.width, args.heads, args.layers)
    model = Hope(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # Evaluation has its own random stream so it does not change training batches.
    eval_rng = torch.Generator().manual_seed(args.seed + 1)

    def batch(source, generator=None):
        starts = torch.randint(len(source) - config.context, (args.batch_size,), generator=generator)
        x = torch.stack([source[i:i + config.context] for i in starts])
        y = torch.stack([source[i + 1:i + config.context + 1] for i in starts])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def evaluate():
        eval_rng.manual_seed(args.seed + 1)
        model.eval()
        metrics = {}
        for name, source in [('train', train), ('validation', val)]:
            metrics[name] = sum(model(*batch(source, eval_rng))[1].item() for _ in range(10)) / 10
        model.train()
        return metrics

    args.out.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f'Hope | {parameter_count:,} parameters | {len(chars)} characters | device={device}', flush=True)
    history = []
    best_loss = float('inf')
    def save_checkpoint(path, step):
        torch.save({'name': 'Hope', 'config': asdict(config), 'chars': chars,
                    'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'steps': step}, path)

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate()
            history.append({'step': step, **metrics})
            (args.out / 'metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            if metrics['validation'] < best_loss:
                best_loss = metrics['validation']
                save_checkpoint(args.out / 'model.pt', step)
            print(f"step {step:5d} | train {metrics['train']:.4f} | validation {metrics['validation']:.4f}", flush=True)
        if step == args.steps:
            break
        _, loss = model(*batch(train))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    save_checkpoint(args.out / 'last.pt', args.steps)
    run_info = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_info.update(parameters=parameter_count, train_characters=len(train), validation_characters=len(val),
                    best_validation_loss=best_loss, torch_version=str(torch.__version__), device=device)
    (args.out / 'run.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')
    (args.out / 'metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(f'Saved {args.out / "model.pt"}', flush=True)


if __name__ == '__main__':
    main()
