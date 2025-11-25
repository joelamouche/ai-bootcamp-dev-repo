from typing import Dict, List, Optional
from utils.schemas import RankedFollower, UserProfile
import time


def follow_top_followers(
    ranked_followers: List[RankedFollower],
    client,
    top_n: int = 10,
):
    """
    Follow the top N ranked followers.
    
    Args:
        ranked_followers: List of RankedFollower objects
        client: Bluesky client (uses global 'client' if not provided)
        handle_to_did: Mapping from handle to DID (optional, will fetch if not provided)
        top_n: Number of top followers to follow (default: 10)
    
    Returns:
        dict: Summary with counts of followed, already_following, errors, etc.
    """
    if not ranked_followers:
        print("⚠️ No ranked followers to follow")
        return {}
    
    # Use global client if not provided
    if client is None:
        client = globals().get('client')
        if client is None:
            raise ValueError("Bluesky client not provided and not found in globals.")
    
    top_n_followers = ranked_followers[:top_n]
    print(f"\n👥 Following top {len(top_n_followers)} most relevant accounts...")
    
    followed_count = 0
    already_following_count = 0
    error_count = 0
    not_found_count = 0
    did_cache: Dict[str, str] = {}
    
    for i, follower in enumerate(top_n_followers, 1):
        try:
            # if follower relevance score is less than 0.5, skip
            if follower.relevance_score < 0.6:
                print(f"ℹ️  {i}. Skipping @{follower.handle} ({follower.display_name}) due to low relevance score: {follower.relevance_score}")
                continue

            # Clean handle: remove '@' if present and normalize
            clean_handle = follower.handle.lstrip('@').strip()

            # Ensure we have a DID; fetch if the AI response did not include it
            follower_did = getattr(follower, "did", "") or ""
            if not follower_did:
                if not follower_did and clean_handle in did_cache:
                    follower_did = did_cache[clean_handle]

                if not follower_did:
                    try:
                        profile = client.get_profile(clean_handle)
                        follower_did = profile.did
                        if follower_did:
                            did_cache[clean_handle] = follower_did
                    except Exception as lookup_err:
                        not_found_count += 1
                        print(f"❌ {i}. Could not resolve DID for @{clean_handle}: {lookup_err}")
                        continue
            
            if not follower_did:
                not_found_count += 1
                print(f"❌ {i}. No DID available for @{clean_handle}, skipping")
                continue

            # Follow using the DID (not the handle)
            client.follow(follower_did)
            followed_count += 1
            print(f"✅ {i}. Followed @{clean_handle} ({follower.display_name})")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            error_count += 1
            error_msg = str(e)
            clean_handle = follower.handle.lstrip('@').strip()
            
            if "already" in error_msg.lower() or "duplicate" in error_msg.lower():
                already_following_count += 1
                print(f"ℹ️  {i}. Already following @{clean_handle}")
            else:
                print(f"❌ {i}. Error following @{clean_handle}: {error_msg}")
    
    summary = {
        'followed': followed_count,
        'already_following': already_following_count,
        'not_found': not_found_count,
        'errors': error_count,
        'total_processed': len(top_n_followers)
    }
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Successfully followed: {followed_count}")
    print(f"   ℹ️  Already following: {already_following_count}")
    print(f"   ⚠️  Not found: {not_found_count}")
    print(f"   ❌ Errors: {error_count}")
    print(f"   📈 Total processed: {len(top_n_followers)}")
    
    return summary
