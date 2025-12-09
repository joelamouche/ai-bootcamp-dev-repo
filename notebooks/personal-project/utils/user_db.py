
from typing import  List
from IPython.display import  display

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch

from utils.schemas import StoredUserProfile, UserProfile, RankedUser

# this should check the status of the user in the db, return true if status is followed
def is_user_in_db(handle):
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="langgraph_db",
        user="langgraph_user",
        password="langgraph_password"   
    )
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        query = """
        SELECT follow_status FROM user_profiles.stored_user_profiles WHERE handle = %s
        """
        cursor.execute(query, (handle,))
        result = cursor.fetchone()
        if result is None:
            return False  # User not in db, or no status to check
        # With RealDictCursor, result is a dict like {'follow_status': ...}
        return result.get("follow_status") == "followed"
    finally:
        conn.close()

def insert_user_profile(user_profiles: List[StoredUserProfile]):
    """
    Insert a list of StoredUserProfile objects into the user_profiles.stored_user_profiles table.
    """
    if not user_profiles:
        print("No user profiles to insert.")
        return

    conn = None
    cursor = None
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            database="langgraph_db",
            user="langgraph_user",
            password="langgraph_password"
        )
        conn.autocommit = True

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # The correct table (see user_profiles.sql) is user_profiles.stored_user_profiles
            insert_query = """
            INSERT INTO user_profiles.stored_user_profiles
            (handle, display_name, description, did, relevance_score, reasoning, follow_status, last_updated)
            VALUES (%(handle)s, %(display_name)s, %(description)s, %(did)s, %(relevance_score)s, %(reasoning)s, %(follow_status)s, to_timestamp(%(last_updated)s))
            ON CONFLICT (did) DO UPDATE SET
                handle = EXCLUDED.handle,
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                relevance_score = EXCLUDED.relevance_score,
                reasoning = EXCLUDED.reasoning,
                follow_status = EXCLUDED.follow_status,
                last_updated = EXCLUDED.last_updated
            """

            # Transform user_profiles (which are Pydantic models) into dicts
            rows = [
                dict(user)
                if isinstance(user, dict)
                else user.dict()
                for user in user_profiles
            ]

            # Use execute_batch for efficient bulk upsert
            execute_batch(cursor, insert_query, rows, page_size=100)

            print(f"Successfully inserted/updated {len(rows)} records into user_profiles.stored_user_profiles")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            conn.close()

# clear all records from the user_profiles.stored_user_profiles table
def clear_user_profiles():
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="langgraph_db",
        user="langgraph_user",
        password="langgraph_password"
    )
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_profiles.stored_user_profiles")
        conn.commit()
        print("Successfully cleared all records from user_profiles.stored_user_profiles")
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()