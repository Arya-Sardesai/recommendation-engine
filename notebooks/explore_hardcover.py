"""
Explore the Hardcover GraphQL API before building bulk ingestion.
Run small queries to confirm the schema: what fields exist on a book,
how to filter by release date, how pagination works.

Set your token first:  $env:HARDCOVER_TOKEN = "your_token"
"""
import os
import json
import requests

TOKEN = os.environ.get("HARDCOVER_TOKEN")
if not TOKEN:
    raise SystemExit("Set HARDCOVER_TOKEN env var first: $env:HARDCOVER_TOKEN = 'your_token'")

ENDPOINT = "https://api.hardcover.app/v1/graphql"
HEADERS = {
    "authorization": TOKEN,            # NOTE: no "Bearer " prefix - HC uses raw token
    "Content-Type": "application/json",
}


def run_query(query, variables=None):
    resp = requests.post(
        ENDPOINT,
        headers=HEADERS,
        json={"query": query, "variables": variables or {}},
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        print("GraphQL errors:", json.dumps(data["errors"], indent=2))
    return data.get("data")


# ---------------------------------------------------------------------------
# Test 1: auth works?
# ---------------------------------------------------------------------------
print("=== Test 1: auth check ===")
me = run_query("query { me { username } }")
print(json.dumps(me, indent=2))

# ---------------------------------------------------------------------------
# Test 2: fetch a few books, see what fields exist
# Hardcover's main table is `books`. Let's grab a small sample.
# ---------------------------------------------------------------------------
print("\n=== Test 2: sample books + available fields ===")
sample = run_query("""
query SampleBooks {
  books(limit: 3, order_by: {users_count: desc}) {
    id
    title
    description
    release_date
    pages
    rating
    ratings_count
    users_count
    contributions {
      author {
        name
      }
    }
  }
}
""")
print(json.dumps(sample, indent=2)[:3000])

# ---------------------------------------------------------------------------
# Test 3: can we filter by release_date? (recency is the whole point)
# ---------------------------------------------------------------------------
print("\n=== Test 3: filter by release_date >= 2018, popular first ===")
recent = run_query("""
query RecentPopular {
  books(
    where: {release_date: {_gte: "2018-01-01"}}
    order_by: {users_count: desc}
    limit: 5
  ) {
    title
    release_date
    users_count
    description
  }
}
""")
print(json.dumps(recent, indent=2)[:3000])

print("\n=== Test 4: pagination via offset ===")
page2 = run_query("""
query Page2 {
  books(
    where: {release_date: {_gte: "2018-01-01"}}
    order_by: {users_count: desc}
    limit: 5
    offset: 5
  ) {
    title
    release_date
    users_count
  }
}
""")
print(json.dumps(page2, indent=2))