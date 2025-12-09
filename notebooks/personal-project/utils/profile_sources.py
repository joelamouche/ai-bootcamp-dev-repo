import time
from typing import List
from utils.schemas import UserProfile
from utils.profile_filters import  filter_non_active_users


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

        if filter_active:
            return filter_non_active_users(client, all_followers, active_days)
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

def get_custom_feed_profiles(bluesky_client,actor,feed_uri , limit=100)->List[UserProfile]: 
    profile = bluesky_client.app.bsky.actor.get_profile({"actor": actor})
    did = profile.did  # e.g. "did:plc:xyz..."

    # 3. Build the feed AT-URI from the URL
    # https://bsky.app/profile/skyfeed.xyz/feed/h-opensource
    feed_uri = f"at://{did}/app.bsky.feed.generator/{feed_uri}"

    # 4. Fetch the feed items
    resp = bluesky_client.app.bsky.feed.get_feed({"feed": feed_uri, "limit": limit})

    # get the profiles of the authors of the posts and
    # use client.get_profile to get the profile of the author   
    profiles = [bluesky_client.get_profile(item.post.author.handle) for item in resp.feed]
    # map the profiles to UserProfile
    profiles = [UserProfile(
        # if no field, set it to an empty string
        handle=profile.handle if profile.handle else "", 
        display_name=profile.display_name if profile.display_name else "", 
        description=profile.description if profile.description else "", 
        did=profile.did if profile.did else "") for profile in profiles]
    return profiles