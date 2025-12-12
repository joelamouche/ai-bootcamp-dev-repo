from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

class UserProfile(BaseModel):
    handle: str = Field(description="Bluesky handle of the user")
    display_name: str = Field(description="Display name of the user")
    description: str = Field(description="Profile description of the user")
    did: str = Field(description="Did of the user")


class RankedUser(UserProfile):
    """Individual follower with relevance information"""
    relevance_score: float = Field(description="Relevance score from 0.0 to 1.0, where 1.0 is most relevant")
    reasoning: str = Field(description="Brief explanation of why this follower is relevant")

class StoredUserProfile(RankedUser):
    # follow_status is either none, pending, followed, rejected or unfollowed
    follow_status: str = Field(description="Follow status of the user")
    last_updated: int = Field(description="Last updated timestamp")
    last_post_processed: Optional[int] = Field(description="Last post processed timestamp")

class Content(BaseModel):
    id: str
    content_text: str
    mediaType: str
    author: str
    created_date: Optional[datetime] = None
    likes: Optional[int] = 0