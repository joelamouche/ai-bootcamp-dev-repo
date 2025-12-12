CONTENT_COLLECTION_NAME="Content-collection-00"
from utils.schemas import Content
from utils.qdrant_ops import get_embeddings_batch
from qdrant_client.models import PointStruct, Document
from dotenv import load_dotenv
load_dotenv()

def get_recent_posts(client, user_handle, limit=50) -> list[Content]:
    """
    Get recent posts for a user, extracting the correct created_date from the post record.
    """
    from dateutil import parser as date_parser

    # Fetch user DID if necessary
    profile = client.app.bsky.actor.get_profile({'actor': user_handle})
    user_did = profile['did']

    # Fetch the user's feed/posts using the feed API
    response = client.app.bsky.feed.get_author_feed({'actor': user_did, 'limit': limit})

    feed_items = getattr(response, 'feed', [])

    posts = []
    for item in feed_items:
        post = getattr(item, 'post', None)
        if post is None:
            continue

        # The post, record, and author may be dataclass/attrs or dicts, handle both
        # 1. Get post_data as dict for fallback, but prefer attribute access
        post_data = post.__dict__ if hasattr(post, '__dict__') else post
        # 2. Get record
        record = getattr(post, 'record', None) or post_data.get('record', {})
        record_data = record.__dict__ if hasattr(record, '__dict__') else record
        # 3. Get author
        author = getattr(post, 'author', None) or post_data.get('author', {})
        author_data = author.__dict__ if hasattr(author, '__dict__') else author

        # Extract created_at: it is always 'created_at' (snake_case) in the atproto library
        created_at_str = None
        if isinstance(record_data, dict):
            created_at_str = record_data.get('created_at')
        else:
            created_at_str = getattr(record_data, 'created_at', None)

        created_at_val = None
        if created_at_str:
            try:
                created_at_val = date_parser.parse(created_at_str)
            except Exception:
                created_at_val = created_at_str

        # Defensive fallback for text, id, likes, etc
        if isinstance(record_data, dict):
            content_text = record_data.get('text', '')
        else:
            content_text = getattr(record_data, 'text', '')

        if isinstance(post_data, dict):
            post_id = post_data.get('uri', '')
            likes = post_data.get('like_count', 0)
        else:
            post_id = getattr(post, 'uri', '')
            likes = getattr(post, 'like_count', 0)

        if isinstance(author_data, dict):
            author_handle = author_data.get('handle', user_handle)
        else:
            author_handle = getattr(author_data, 'handle', user_handle)

        posts.append(
            Content(
                id=post_id,
                content_text=content_text,
                mediaType='bluesky',
                author=author_handle,
                created_date=created_at_val,
                likes=likes if likes is not None else 0
            )
        )
    return posts

def process_and_upsert_twitter_content(qdrant_client, input: list[Content], batch_size: int = 100):
    """
    Upserts tweet Content objects to Qdrant in batches to avoid "payload too large" errors.

    Args:
        input (list[Content]): List of Content objects representing tweets.
        batch_size (int): Number of points to upsert in each batch.
    """
    # get latest collection size
    collection_size = qdrant_client.count(collection_name=CONTENT_COLLECTION_NAME)
    current_id = collection_size.count + 1

    total_points = 0
    for batch_start in range(0, len(input), batch_size):
        batch = input[batch_start : batch_start + batch_size]
        valid_items = [
            data for data in batch
            if isinstance(data.content_text, str) and data.content_text.strip()
        ]
        if not valid_items:
            print(f"Batch {batch_start // batch_size + 1} skipped (no valid content_text)")
            continue

        text_to_embed = [data.content_text for data in valid_items]
        embeddings = get_embeddings_batch(text_to_embed)

        pointstructs = []

        for embedding, data in zip(embeddings, batch):
            pointstructs.append(
                PointStruct(
                    id=current_id,
                    vector={
                        "text-embedding-3-small": embedding,
                        "bm25": Document(
                            text=data.content_text,
                            model="qdrant/bm25"
                        ),
                    },
                    payload=data.model_dump(),
                )
            )
            current_id += 1

        # Upsert this batch
        qdrant_client.upsert(
            collection_name=CONTENT_COLLECTION_NAME,
            points=pointstructs
        )
        total_points += len(pointstructs)
        print(f"Upserted batch {batch_start // batch_size + 1}: {len(pointstructs)} points")

    print(f"Upserted total {total_points} points for {len(input)} rows")

def process_profile_posts_into_qdrant(qdrant_client, client, handle: str, limit: int = 50):
    posts = get_recent_posts(client, handle, limit)

    print(f"Number of posts: {len(posts)}\n")

    print("First 5 posts:")
    for i, post in enumerate(posts[:5], start=1):
        print(f"\nPost {i}:")
        print(f"  id: {post.id}")
        print(f"  author: {post.author}")
        print(f"  created_date: {post.created_date}")
        print(f"  likes: {post.likes}")
        print(f"  content_text: {post.content_text}")
        print(f"  mediaType: {post.mediaType}\n")
    process_and_upsert_twitter_content(qdrant_client,posts)