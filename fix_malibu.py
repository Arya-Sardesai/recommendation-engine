import os, requests, json
TOKEN = os.environ['HARDCOVER_TOKEN']
ENDPOINT = 'https://api.hardcover.app/v1/graphql'
HEADERS = {'authorization': TOKEN, 'Content-Type': 'application/json'}
q = 'query { books(where: {id: {_eq: 703328}}) { id title description release_date pages rating ratings_count users_count contributions { author { name } } } }'
r = requests.post(ENDPOINT, headers=HEADERS, json={'query': q}, timeout=30)
books = r.json()['data']['books']
with open('data/raw/hardcover_recent.jsonl', 'a', encoding='utf-8') as f:
    for b in books:
        f.write(json.dumps(b, ensure_ascii=False) + '\n')
        print('Appended:', b['title'], b.get('release_date'))
