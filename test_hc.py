import os, requests
TOKEN = os.environ['HARDCOVER_TOKEN']
ENDPOINT = 'https://api.hardcover.app/v1/graphql'
HEADERS = {'authorization': TOKEN, 'Content-Type': 'application/json'}
q = 'query { books(where: {title: {_eq: "Malibu Rising"}}, limit: 3) { id title release_date users_count } }'
r = requests.post(ENDPOINT, headers=HEADERS, json={'query': q}, timeout=30)
print(r.status_code, r.text[:500])
