from typing import Dict, List, Optional
from utils.schemas import RankedUser, UserProfile
import time
from utils.user_db import is_user_in_db, insert_user_profile
from utils.schemas import StoredUserProfile
from utils.profile_filters import analyze_user_with_ai

def follow_user(user: UserProfile, client):
    """
    Follow a user.
    """
    client.follow(user.did)
    return True

def follow_top_users(
    ranked_users: List[RankedUser],
    client,
    top_n: int = 10,
):
    """
    Follow the top N ranked followers.
    
    Args:
        ranked_users: List of RankedUser objects
        client: Bluesky client (uses global 'client' if not provided)
        handle_to_did: Mapping from handle to DID (optional, will fetch if not provided)
        top_n: Number of top followers to follow (default: 10)
    
    Returns:
        dict: Summary with counts of followed, already_following, errors, etc.
    """
    if not ranked_users:
        print("⚠️ No ranked users to follow")
        return {}
    
    # Use global client if not provided
    if client is None:
        client = globals().get('client')
        if client is None:
            raise ValueError("Bluesky client not provided and not found in globals.")
    
    top_n_followers = ranked_users[:top_n]
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

# process list of users
# see if already followed in db, if not followed and not already in db, analyze, follow and insert into db
# if followed, do nothing
# if already in db, do nothing
PERTINANCE_THRESHOLD=0.6
def process_users(users, my_profile, openai_client, bluesky_client):
    for user in users:
        if not is_user_in_db(user.handle):
            print(f"User {user.handle} not followed and not in db, analyzing, following and inserting into db")
            analyzed_user=analyze_user_with_ai(user, my_profile, openai_client)
            follow_status="rejected"
            if analyzed_user.relevance_score >= PERTINANCE_THRESHOLD:
                print(f"User {user.handle} is relevant, following and inserting into db")
                follow_status="followed"
                follow_user(user, bluesky_client)
            user_to_insert=StoredUserProfile(**analyzed_user.model_dump(), follow_status=follow_status, last_updated=int(time.time()))
            insert_user_profile([user_to_insert])
        else:
            print(f"User {user.handle} already followed or in db")