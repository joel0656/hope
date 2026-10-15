"""Build a reproducible story collection, holding out the ending of each book."""
import hashlib
import json
from pathlib import Path
import re
import urllib.request

BOOKS = [(11, "Alice's Adventures in Wonderland"),
         (21, "Three Hundred Aesop's Fables"), (2591, "Grimms' Fairy Tales")]


def main():
    folder = Path(__file__).parent / 'data/stories'
    folder.mkdir(parents=True, exist_ok=True)
    trains, validations, sources = [], [], []
    for number, title in BOOKS:
        url = f'https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt'
        original = folder / f'pg{number}-original.txt'
        if not original.exists():
            existing = folder.parent / 'alice-original.txt'
            if number == 11 and existing.exists():
                raw = existing.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=60) as response:
                    raw = response.read()
            original.write_bytes(raw)
        raw = original.read_bytes()
        text = raw.decode('utf-8-sig').replace('\r\n', '\n')
        start = re.search(r'\*\*\* START OF .*?\*\*\*', text)
        end = re.search(r'\*\*\* END OF .*?\*\*\*', text)
        if not start or not end:
            raise ValueError(f'Missing Gutenberg markers: {title}')
        body = text[start.end():end.start()].strip()
        if len(body) < 10000:
            raise ValueError(f'Unexpectedly short body: {title}')
        boundary = body.rfind('\n\n', 0, int(len(body) * .9))
        if boundary < len(body) * .8:
            raise ValueError('No suitable paragraph split')
        trains.append(body[:boundary].strip())
        validations.append(body[boundary:].strip())
        sources.append({'title': title, 'url': url, 'catalog': f'https://www.gutenberg.org/ebooks/{number}',
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'train_characters': len(trains[-1]), 'validation_characters': len(validations[-1])})
        print(f'{title}: {len(trains[-1]):,} train / {len(validations[-1]):,} validation', flush=True)
    for filename, parts in [('train.txt', trains), ('validation.txt', validations)]:
        (folder / filename).write_text('\n\n\n'.join(parts) + '\n', encoding='utf-8')
    (folder / 'sources.json').write_text(json.dumps({
        'sources': sources, 'rights': 'Project Gutenberg lists these editions as public domain in the USA.',
        'split': 'Ending approximately 10% of each book held out, at a paragraph boundary.',
        'processing': 'Normalize newlines and remove Gutenberg header/footer; retain original files and licenses. Book front matter remains.',
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
