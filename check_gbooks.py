"""Check that GOOGLE_BOOKS_KEY works and that we are authenticated.

A one-liner for this does not survive PowerShell: the inner quotes needed for
the intitle: operator get eaten before Python sees them.

    $env:GOOGLE_BOOKS_KEY = "..."
    python check_gbooks.py
"""
import os
import sys

import requests

key = os.environ.get("GOOGLE_BOOKS_KEY")
if not key:
    sys.exit('GOOGLE_BOOKS_KEY not set. Run: $env:GOOGLE_BOOKS_KEY = "your_key"')

print(f"key ends in ...{key[-6:]}")

TESTS = [
    ("Dongri to Dubai", "S. Hussain Zaidi"),
    ("The Maltese Falcon", "Dashiell Hammett"),
    ("Shikhandi", "Devdutt Pattanaik"),
]

for title, author in TESTS:
    q = f'intitle:"{title}" inauthor:"{author}"'
    r = requests.get("https://www.googleapis.com/books/v1/volumes",
                     params={"q": q, "key": key, "maxResults": 5}, timeout=20)
    print(f"\n{title}  ->  HTTP {r.status_code}")
    if r.status_code == 403:
        print("  403: the Books API may not be enabled on this project yet,")
        print("  or the key is restricted to a different API.")
        print(f"  body: {r.text[:200]}")
        continue
    if r.status_code == 429:
        print("  429: rate limited even WITH a key -- that is the finding.")
        continue
    if r.status_code != 200:
        print(f"  body: {r.text[:200]}")
        continue
    items = r.json().get("items") or []
    print(f"  {len(items)} volumes")
    for it in items:
        vi = it.get("volumeInfo", {})
        d = (vi.get("description") or "").strip()
        print(f"    {str(vi.get('title'))[:44]:46s} "
              f"{str((vi.get('authors') or ['?'])[0])[:20]:22s} "
              f"{len(d):>5} chars  {vi.get('language','?')}")
        if d:
            print(f"      {d[:140]}...")

print("\nIf these are 200s with real descriptions, the source is live and the")
print("earlier 'rate-limited, untested' verdict can finally be replaced with a")
print("measured one.")
