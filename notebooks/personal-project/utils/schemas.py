from pydantic import BaseModel, Field
from typing import List

class UserProfile(BaseModel):
    handle: str = Field(description="Bluesky handle of the user")
    display_name: str = Field(description="Display name of the user")
    description: str = Field(description="Profile description of the user")
    did: str = Field(description="Did of the user")

class FollowerRanking(BaseModel):
    """Ranked list of followers with relevance scores"""
    ranked_followers: List[dict] = Field(
        description="List of followers ranked by relevance, each with handle, display_name, description, and relevance_score"
    )

class RankedFollower(UserProfile):
    """Individual follower with relevance information"""
    relevance_score: float = Field(description="Relevance score from 0.0 to 1.0, where 1.0 is most relevant")
    reasoning: str = Field(description="Brief explanation of why this follower is relevant")