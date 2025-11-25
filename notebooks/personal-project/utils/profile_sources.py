import time
from typing import List
from utils.schemas import UserProfile
from utils.profile_filters import is_user_active_in_last_month


def fetch_followers_from_user(
    client,
    username: str,
    max_followers: int = 100,
    filter_active: bool = True,
    active_days: int = 30,
) -> List[UserProfile]:
    """
    Fetch followers from a specific Bluesky user, optionally filtering by activity and return a list of UserProfile objects.
    
    Args:
        client: Authenticated Bluesky client
        username: Target username (with or without @)
        max_followers: Maximum number of followers to fetch (default: 100)
        filter_active: If True, only return followers active in the last month (default: True)
        active_days: Number of days to look back for activity (default: 30)
    
    Returns:
        list of UserProfile objects
    """
    # Clean username
    clean_username = username.lstrip('@').strip()
    
    print(f"📊 Fetching followers of @{clean_username}...")
    
    try:
        # Get the target profile
        target_profile = client.get_profile(clean_username)
        print(f"✅ Found profile: {target_profile.display_name or target_profile.handle}")
        print(f"   Followers: {target_profile.followers_count}")
        
        # Fetch followers (with pagination if needed)
        all_followers = []
        cursor = None
        
        while len(all_followers) < max_followers:
            if cursor:
                followers_response = client.get_followers(target_profile.did, limit=100, cursor=cursor)
            else:
                followers_response = client.get_followers(target_profile.did, limit=100)
            
            # Map followers to UserProfile objects, ensuring strings for required fields
            for follower in followers_response.followers:
                handle = getattr(follower, "handle", "") or ""
                display_name = getattr(follower, "display_name", None) or handle
                description = getattr(follower, "description", "") or ""
                did = getattr(follower, "did", "") or ""
                all_followers.append(
                    UserProfile(
                        handle=handle,
                        display_name=display_name,
                        description=description,
                        did=did,
                    )
                )
            
            if not hasattr(followers_response, 'cursor') or not followers_response.cursor:
                break
            cursor = followers_response.cursor
            
            if len(all_followers) >= max_followers:
                break
        
        # Limit to the requested number (before filtering)
        all_followers = all_followers[:max_followers]
        
        print(f"✅ Fetched {len(all_followers)} followers")
        
        # Filter by activity if requested
        if filter_active:
            print(f"\n🔍 Filtering followers active in the last {active_days} days...")
            active_followers = []
            checked_count = 0
            
            for follower in all_followers:
                checked_count += 1
                if checked_count % 10 == 0:
                    print(f"   Checking activity: {checked_count}/{len(all_followers)}...")
                
                if is_user_active_in_last_month(client, follower.did, days=active_days):
                    active_followers.append(follower)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            print(f"✅ Found {len(active_followers)} active followers out of {len(all_followers)} checked")
            return active_followers
        else:
            return all_followers
        
    except Exception as e:
        print(f"❌ Error fetching followers: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_my_profile(client)-> UserProfile:
    """
    Get your own profile information.
    
    Args:
        client: Authenticated Bluesky client
    
    Returns:
        Profile object with your profile information
    """
    print(f"Getting my profile: {client.me.handle}...")
    my_profile = client.get_profile(client.me.handle)
    # print my profile name and description
    print(f"✅ My profile name: {my_profile.display_name}")
    print(f"✅ My profile description: {my_profile.description}")
    return UserProfile(handle=my_profile.handle, display_name=my_profile.display_name, description=my_profile.description, did=my_profile.did)


# A function that returns post dicts from the Bluesky search posts endpoint (e.g. https://bsky.app/search?q=ethglobal)
def search_posts_by_keyword(client,keyword, limit=100):
    """
    Searches public Bluesky posts by keyword and returns a list of post dicts.
    """
    # Call the endpoint
    response = client.app.bsky.feed.search_posts({"q": keyword, "limit": limit})
    # Convert to dict
    response_dict = response.model_dump()
    # The posts are under the "posts" key in the returned dict
    posts = response_dict.get("posts", [])
    return posts

# a function that returns profiles that created the posts returned by the endpoint equivalent of https://bsky.app/search?q=ethglobal
def search_profiles_by_keyword(client,keyword, limit=100)->List[UserProfile]:
    # get the posts returned by the endpoint equivalent of https://bsky.app/search?q=ethglobal
    posts = search_posts_by_keyword(client,keyword, limit)
    # get the profiles that created the posts
    profiles = [post.get("author", {}) for post in posts]
    return [UserProfile(handle=profile.get("handle", ""), display_name=profile.get("display_name", ""), description=profile.get("description", ""), did=profile.get("did", "")) for profile in profiles]