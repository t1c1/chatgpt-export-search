\
import argparse, sqlite3, json, sys
from pathlib import Path
from unified_search import UnifiedSearchService

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help='FTS5 query, e.g. "robotics AND supplier" or just words')
    ap.add_argument("--db", default="./index/chatgpt.db")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--provider", choices=["claude", "chatgpt"], help="Filter by provider")
    ap.add_argument("--role", choices=["user", "assistant", "system"], help="Filter by role")
    ap.add_argument("--sort", choices=["rank", "newest", "oldest"], default="rank", help="Sort results")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"DB not found at {db}. Run: python index.py --export ./my_export --out ./index")
        sys.exit(1)

    # Use unified search service
    search_service = UnifiedSearchService(str(db))

    results, total_count = search_service.search(
        query=args.query,
        limit=args.limit,
        offset=args.offset,
        provider=args.provider,
        role=args.role,
        sort_by=args.sort
    )

    for result in results:
        print("="*80)
        print(f"{result['title']}  ({result['source']})")
        print(f"{result['conv_id']} | {result['role']} | ts={result['timestamp']} | id={result['id']}")
        print(result['snippet'])

    print(f"\n{len(results)} result(s) of {total_count} total.")

if __name__ == "__main__":
    main()
