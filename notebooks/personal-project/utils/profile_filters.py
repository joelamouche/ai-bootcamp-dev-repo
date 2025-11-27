from typing import List
from datetime import datetime, timedelta
from utils.schemas import UserProfile, RankedUser
from openai import OpenAI

def is_user_active_in_last_month(client, user_did, days=30, debug=False) -> bool:
    """
    Check if a user was active (posted) in the last N days.
    
    Args:
        client: Authenticated Bluesky client
        user_did: User's DID (Decentralized Identifier)
        days: Number of days to look back (default: 30)
        debug: If True, print debug information (default: False)
    
    Returns:
        bool: True if user posted in the last N days, False otherwise
    """
    try:
        # Calculate the cutoff date (using timezone-aware datetime)
        from datetime import timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        if debug:
            print(f"   Checking activity for {user_did}")
            print(f"   Cutoff date: {cutoff_date}")
        
        # Get users latest post (only the latest post)
        # Use the correct API method: client.app.bsky.feed.get_author_feed
        feed_response = client.app.bsky.feed.get_author_feed({'actor': user_did, 'limit': 1})
        
        if not hasattr(feed_response, 'feed') or not feed_response.feed:
            if debug:
                print(f"   No feed found or empty feed")
            return False
        
        if debug:
            print(f"   Found {len(feed_response.feed)} posts in feed")
        
        # Only check the *first* post in the feed (no loop)
        if feed_response.feed:
            feed_item = feed_response.feed[0]
            try:
                # The feed structure in atproto: feed_item.post.record.createdAt
                created_at_str = None
                
                # Access the post object
                post_obj = feed_item.post if hasattr(feed_item, 'post') else None
                
                if post_obj is None:
                    if debug:
                        print(f"   Post 0: No post object found")
                    return False  # Not a loop: just return!
                
                # First, try to get indexed_at from post (snake_case, not camelCase)
                created_at_str = None
                if hasattr(post_obj, 'indexed_at'):
                    created_at_str = post_obj.indexed_at
                    if debug:
                        print(f"   Post 0: Using indexed_at from post: {created_at_str}")
                elif hasattr(post_obj, 'indexedAt'):  # Fallback to camelCase just in case
                    created_at_str = post_obj.indexedAt
                    if debug:
                        print(f"   Post 0: Using indexedAt from post: {created_at_str}")
                
                # If not found, try to get record from post - this is the key part
                # The record contains the actual post data including created_at (snake_case)
                if not created_at_str:
                    if hasattr(post_obj, 'record'):
                        record = post_obj.record
                    elif hasattr(post_obj, 'value'):
                        record = post_obj.value
                    else:
                        record = None
                    
                    if record is None:
                        if debug:
                            print(f"   Post 0: No record found in post object")
                        return False  # Not a loop: just return!
                    
                    # Get created_at from record (snake_case, not camelCase)
                    if isinstance(record, dict):
                        created_at_str = record.get('created_at', '') or record.get('createdAt', '')
                    else:
                        # Try attribute access with snake_case first
                        created_at_str = getattr(record, 'created_at', None)
                        if not created_at_str:
                            created_at_str = getattr(record, 'createdAt', None)  # Fallback to camelCase
                        
                        # If still None, try to convert record to dict
                        if created_at_str is None:
                            try:
                                if hasattr(record, 'model_dump'):
                                    record_dict = record.model_dump()
                                elif hasattr(record, 'dict'):
                                    record_dict = record.dict()
                                elif hasattr(record, '__dict__'):
                                    record_dict = record.__dict__
                                else:
                                    record_dict = {}
                                created_at_str = record_dict.get('created_at', '') or record_dict.get('createdAt', '')
                            except Exception as e:
                                if debug:
                                    print(f"   Post 0: Error converting record to dict: {e}")
                                pass
                
                if not created_at_str:
                    if debug:
                        print(f"   Post 0: Could not find created_at in record or indexed_at in post")
                    return False  # Not a loop: just return!
                
                # Parse the timestamp (format: 2024-01-01T12:00:00.000Z)
                # Handle both with and without microseconds and timezone
                created_at_str_clean = str(created_at_str).replace('Z', '+00:00')
                if '.' not in created_at_str_clean.split('+')[0].split('-')[-1]:
                    # No microseconds, add them before the timezone
                    if '+00:00' in created_at_str_clean:
                        created_at_str_clean = created_at_str_clean.replace('+00:00', '.000+00:00')
                    elif 'Z' in str(created_at_str):
                        created_at_str_clean = str(created_at_str).replace('Z', '.000+00:00')
                
                created_at = datetime.fromisoformat(created_at_str_clean)
                # Ensure both datetimes are timezone-aware for comparison
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                
                if debug:
                    print(f"   Post 0: Created at {created_at}, Cutoff: {cutoff_date}, Active: {created_at >= cutoff_date}")
                
                if created_at >= cutoff_date:
                    if debug:
                        print(f"   ✅ User is active!")
                    return True
                    
            except Exception as e:
                if debug:
                    print(f"   Post 0: Error processing - {e}")
                    import traceback
                    traceback.print_exc()
                return False  # Only evaluate the first post
        if debug:
            print(f"   ❌ No recent posts found")
        return False
        
    except Exception as e:
        if debug:
            print(f"   ❌ Exception checking activity: {e}")
            import traceback
            traceback.print_exc()
        # If we can't check activity, assume inactive to be safe
        return False


def analyze_users_with_ai(users: List[UserProfile], my_profile: UserProfile, openai_client: OpenAI, num_results: int = 10)-> List[RankedUser]:
    """
    Use AI to analyze and rank users by relevance to your profile.
    
    Args:
        users: List of UserProfile objects
        my_profile: Your profile object
        target_username: Username of the target account whose users we're analyzing
        openai_client: OpenAI client with instructor
        num_results: Number of top results to return (default: 10)
    
    Returns:
        List of RankedUser objects sorted by relevance
    """
    if not users:
        print("⚠️ No users to analyze")
        return []
    
    print(f"\n🤖 Analyzing {len(users)} users with AI...")
    
    # Prepare follower data for AI analysis
    users_data = []
    for follower in users:
        users_data.append({
            "handle": follower.handle,
            "display_name": follower.display_name or follower.handle,
            "description": follower.description or ""
        })
    
    # Get your profile info
    my_description = my_profile.description or ""
    my_display_name = my_profile.display_name or my_profile.handle
    
    # Create prompt for AI analysis
    analysis_prompt = f"""You are analyzing Bluesky users to find the most relevant accounts for someone to follow.

My Profile:
- Display Name: {my_display_name}
- Handle: {my_profile.handle}
- Description: {my_description}

I want to find users who would be most relevant for me to follow based on:
1. Shared interests (based on my description: {my_description[:200]})
2. Professional alignment
3. Content relevance
4. Community overlap

Here are the users to analyze:
{chr(10).join([f"- @{f['handle']} ({f['display_name']}): {f['description'][:150]}" for f in users_data])}

Please rank these users by relevance to my profile. Return the top {num_results} most relevant users with:
- handle: The Bluesky handle
- display_name: Their display name
- description: Their profile description
- relevance_score: A score from 0.0 to 1.0 (1.0 = most relevant)
- reasoning: A brief explanation of why they're relevant
- did: The DID of the follower

Focus on accounts that share interests, professional connections, or would provide valuable content for someone with my profile."""
    
    try:
        # Use instructor to get structured output
        response = openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing social media profiles and finding relevant connections."},
                {"role": "user", "content": analysis_prompt}
            ],
            response_model=List[RankedUser],
            temperature=0.3
        )
        
        ranked_users = response
        print(f"✅ AI analysis complete! Found {len(ranked_users)} relevant users")
        # print users with relevance score and reasoning, with relevance score >0.6
        for follower in ranked_users:
            if follower.relevance_score > 0.6:
                print(f"@{follower.handle} ({follower.display_name}): {follower.relevance_score:.2f} - {follower.reasoning}")
        
        return ranked_users
        
    except Exception as e:
        print(f"❌ Error during AI analysis: {e}")
        import traceback
        traceback.print_exc()
        return []
