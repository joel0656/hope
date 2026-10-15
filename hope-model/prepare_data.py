"""Download a public-domain book and extract its body for Hope's training."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

SOURCE = 'https://www.gutenberg.org/cache/epub/11/pg11.txt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', type=Path, help='Use a previously downloaded original file.')
    args = parser.parse_args()
    folder = Path(__file__).parent / 'data'
    folder.mkdir(exist_ok=True)
    if args.local:
        raw = args.local.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            raw = response.read()
    text = raw.decode('utf-8-sig').replace('\r\n', '\n')
    start = re.search(r'\*\*\* START OF .*?\*\*\*', text)
    end = re.search(r'\*\*\* END OF .*?\*\*\*', text)
    if not start or not end or start.end() >= end.start():
        raise ValueError('Could not find Gutenberg body markers; source format may have changed.')
    body = text[start.end():end.start()].strip() + '\n'
    if len(body) < 10000:
        raise ValueError('Unexpectedly short book; refusing to use this download.')
    (folder / 'alice-original.txt').write_bytes(raw)
    (folder / 'alice.txt').write_text(body, encoding='utf-8')
    manifest = {
        'title': "Alice's Adventures in Wonderland", 'author': 'Lewis Carroll',
        'source': SOURCE, 'catalog': 'https://www.gutenberg.org/ebooks/11',
        'rights': 'Project Gutenberg lists this work as public domain in the USA.',
        'processing': 'UTF-8 decoded; CRLF normalized; Gutenberg header and footer excluded from training. Original retained with license.',
        'sha256_original': hashlib.sha256(raw).hexdigest(),
        'characters': len(body), 'split': 'First 90% training, last 10% validation',
    }
    (folder / 'source.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared {len(body):,} characters in {folder / "alice.txt"}')


if __name__ == '__main__':
    main()
