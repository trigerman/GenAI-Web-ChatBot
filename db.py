import mysql.connector
from mysql.connector import errorcode
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )

def create_user_if_not_exists(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT IGNORE INTO users (email) VALUES (%s)", (email,))
    
    # Also ensure user_progress table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_progress (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) NOT NULL,
            course VARCHAR(32) NOT NULL,
            topic_name VARCHAR(255) NOT NULL,
            interaction_count INT DEFAULT 1,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_user_course_topic (email, course, topic_name)
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def save_chat_log(session_id, user_message, bot_response, email, course):
    conn = get_connection()
    cursor = conn.cursor()
    course_value = course.lower() if course else None
    insert_query = """
        INSERT INTO chat_sessions (session_id, user_message, bot_response, email, course)
        VALUES (%s, %s, %s, %s, %s)
    """
    try:
        cursor.execute(insert_query, (session_id, user_message, bot_response, email, course_value))
    except mysql.connector.Error as exc:
        if getattr(exc, "errno", None) == errorcode.ER_BAD_FIELD_ERROR:
            cursor.execute("ALTER TABLE chat_sessions ADD COLUMN course VARCHAR(32) DEFAULT NULL")
            conn.commit()
            cursor.execute(insert_query, (session_id, user_message, bot_response, email, course_value))
        else:
            cursor.close()
            conn.close()
            raise
    conn.commit()
    cursor.close()
    conn.close()

def get_chat_history(email, course=None):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        if course:
            course_value = course.lower()
            cursor.execute("""
                SELECT user_message, bot_response, timestamp FROM chat_sessions
                WHERE email = %s AND (course = %s OR (course IS NULL AND %s = 'ist256'))
                ORDER BY id ASC
            """, (email, course_value, course_value))
        else:
            cursor.execute("""
                SELECT user_message, bot_response, timestamp FROM chat_sessions
                WHERE email = %s ORDER BY id ASC
            """, (email,))
    except mysql.connector.Error as exc:
        if getattr(exc, "errno", None) == errorcode.ER_BAD_FIELD_ERROR:
            cursor.execute("""
                SELECT user_message, bot_response, timestamp FROM chat_sessions
                WHERE email = %s ORDER BY id ASC
            """, (email,))
        else:
            cursor.close()
            conn.close()
            raise
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    return history


def has_session_history(session_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM chat_sessions WHERE session_id = %s LIMIT 1",
        (session_id,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return bool(row)

def has_email_course_history(email, course=None):
    conn = get_connection()
    cursor = conn.cursor()
    if course:
        cursor.execute(
            """
            SELECT 1 FROM chat_sessions
            WHERE email = %s AND (course = %s OR (course IS NULL AND %s = 'ist256'))
            LIMIT 1
            """,
            (email, (course or '').lower(), (course or '').lower())
        )
    else:
        cursor.execute(
            "SELECT 1 FROM chat_sessions WHERE email = %s LIMIT 1",
            (email,)
        )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return bool(row)

def record_topic_progress(email, course, topics_list):
    if not email or not topics_list:
        return
        
    conn = get_connection()
    cursor = conn.cursor()
    course_value = (course or 'ist256').lower()
    
    # Ensure table exists before trying to insert
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_progress (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(255) NOT NULL,
            course VARCHAR(32) NOT NULL,
            topic_name VARCHAR(255) NOT NULL,
            interaction_count INT DEFAULT 1,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_user_course_topic (email, course, topic_name)
        )
    """)
    
    for topic in topics_list:
        if not topic: continue
        cursor.execute("""
            INSERT INTO user_progress (email, course, topic_name, interaction_count)
            VALUES (%s, %s, %s, 1)
            ON DUPLICATE KEY UPDATE 
                interaction_count = interaction_count + 1,
                last_updated = CURRENT_TIMESTAMP
        """, (email, course_value, str(topic)))
        
    conn.commit()
    cursor.close()
    conn.close()

def get_user_progress(email, course, syllabus_keys):
    if not email or not syllabus_keys:
        return {}
        
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    course_value = (course or 'ist256').lower()
    
    # Query to fetch progress specifically for this user/course
    try:
        cursor.execute("""
            SELECT topic_name, interaction_count 
            FROM user_progress 
            WHERE email = %s AND course = %s
        """, (email, course_value))
        rows = cursor.fetchall()
    except mysql.connector.Error as exc:
        if getattr(exc, "errno", None) == errorcode.ER_NO_SUCH_TABLE:
            rows = []
        else:
            cursor.close()
            conn.close()
            raise
            
    cursor.close()
    conn.close()
    
    # Map the database hits to the provided top-level syllabus categories
    progress_map = {key: 0 for key in syllabus_keys}
    
    for row in rows:
        topic = row.get("topic_name")
        for category, subtopics in syllabus_keys.items():
            if topic == category or topic in subtopics:
                # 1 discussion = 20% progress on the WHOLE category! Much more generous!
                progress_map[category] += row.get("interaction_count", 0) * 20
                break
                
    # Normalize the progress across all subtopics in a category (cap at 100)
    final_progress = {}
    for category in syllabus_keys.keys():
        current_score = progress_map[category]
        percentage = min(current_score, 100)
        final_progress[category] = percentage
        
    return final_progress

