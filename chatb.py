# worWebChatBot-2/worWebChatBot-2/chatb.py

import openai
from dotenv import load_dotenv
import os
import json
import numpy as np
from collections import defaultdict, deque
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from db import save_chat_log, has_session_history, has_email_course_history, get_chat_history, record_topic_progress
import html
import re

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API Key is missing! Set it in the .env file.")

# Utility to format code blocks for HTML output
def format_code_response(code, language="html"):
    escaped_code = html.escape(code.strip())
    return f'<pre><code class="language-{language}">{escaped_code}</code></pre>'

# Convert markdown-style lists and headings into HTML
def format_explanation_text(text):
    # Headings
    text = re.sub(r'(?m)^###\s+(.*)', r'<h3>\1</h3>', text)
    text = re.sub(r'(?m)^##\s+(.*)', r'<h2>\1</h2>', text)
    text = re.sub(r'(?m)^#\s+(.*)', r'<h1>\1</h1>', text)
    
    # Process lists safely, line by line
    lines = text.split('\n')
    out = []
    in_ul = False
    in_ol = False
    
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\d+\.\s+', stripped):
            if in_ul: 
                out.append('</ul>')
                in_ul = False
            if not in_ol: 
                out.append('<ol>')
                in_ol = True
            out.append(re.sub(r'^\d+\.\s+(.*)', r'<li>\1</li>', stripped))
        elif re.match(r'^-\s+', stripped):
            if in_ol: 
                out.append('</ol>')
                in_ol = False
            if not in_ul: 
                out.append('<ul>')
                in_ul = True
            out.append(re.sub(r'^-\s+(.*)', r'<li>\1</li>', stripped))
        else:
            if in_ol: 
                out.append('</ol>')
                in_ol = False
            if in_ul: 
                out.append('</ul>')
                in_ul = False
            out.append(line)
            
    if in_ol: out.append('</ol>')
    if in_ul: out.append('</ul>')
    
    return '\n'.join(out)

# Handle fenced code blocks and inline code
def convert_code_blocks(text):
    code_blocks = []

    def replace_block(match):
        lang = match.group(1) or "html"
        code = match.group(2)
        placeholder = f"__CODEBLOCK{len(code_blocks)}__"
        code_blocks.append(format_code_response(code, lang))
        return placeholder

    # Fenced code
    block_pattern = r"```(html|css|javascript)?\n(.*?)```"
    text_with_placeholders = re.sub(block_pattern, replace_block, text, flags=re.DOTALL)
    # Escape remaining HTML
    escaped = html.escape(text_with_placeholders)
    # Inline backticks
    escaped = re.sub(r'`([^`]+)`', r'<code>\1</code>', escaped)
    # Restore fenced blocks
    for i, block in enumerate(code_blocks):
        escaped = escaped.replace(f"__CODEBLOCK{i}__", block)
    return escaped

# JSON loaders
def load_syllabus(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Syllabus file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Syllabus file is not valid JSON: {file_path}") from exc

def load_group_projects(file_path="group_project.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Group project file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def load_course_instruction(file_path="course_instruction.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Course instruction file not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

# Static data
course_instruction = load_course_instruction()
group_projects   = load_group_projects()

# Maintain context for follow-up
conversation_context = {"last_bot_message": None}

# Per-session conversation memory (in-process only)
MAX_HISTORY_TURNS = 3
conversation_buffers = defaultdict(lambda: deque(maxlen=MAX_HISTORY_TURNS))

# Keywords for routing questions
GROUP_PROJECT_KEYWORDS = [
    "group project", "group assignment", "week 6", "lab activity", "lab assignment",
    "group tasks", "team assignment", "project tasks", "week 7", "week 8", "week 9",
    "week 1", "week 2", "week 3", "week 4", "week 5", "week 10", "week 11", "week 12", "week 13"
]

COURSE_INSTRUCTION_KEYWORDS = [
    "grading", "grade", "late policy", "academic integrity", "canvas",
    "psualert", "emergency", "attendance", "technical requirement",
    "course policy", "schedule", "lesson", "bias", "disability", "accessibility"
]

# Helper to flatten instruction JSON into text
def flatten_instruction_text(ci):
    lines = []
    for section, content in ci.items():
        lines.append(f"**{section.upper()}**")
        if isinstance(content, dict):
            for k, v in content.items():
                lines.append(f"- {k}: {v}")
        elif isinstance(content, list):
            for item in content:
                lines.append(f"- {item}")
        else:
            lines.append(str(content))
        lines.append("")
    return "\n".join(lines)

def flatten_syllabus_text(data):
    lines = []

    def walk(prefix, value):
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                next_prefix = f"{prefix} > {child_key}" if prefix else child_key
                walk(next_prefix, child_value)
        elif isinstance(value, list):
            items = ", ".join(str(item) for item in value) if value else "No details provided"
            lines.append(f"{prefix}: {items}")
        else:
            lines.append(f"{prefix}: {value}")

    for key, value in data.items():
        walk(key, value)

    return "\n".join(f"- {line}" for line in lines)

# Embedding setup for topic matching
def _extract_topics_from_syllabus(data):
    topics = []
    def walk(key, value):
        if isinstance(value, dict):
            topics.append(str(key))
            for k, v in value.items():
                walk(k, v)
        elif isinstance(value, list):
            topics.append(str(key))
            for item in value:
                if isinstance(item, (str, int, float)):
                    topics.append(str(item))
        else:
            topics.append(str(key))
            if isinstance(value, (str, int, float)):
                topics.append(str(value))
    for k, v in data.items():
        walk(k, v)
    seen, out = set(), []
    for t in topics:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def get_allowed_topics(course):
    syllabus_file = "hcdd340_syllabus.json" if course == "hcdd340" else "syllabus.json"
    try:
        data = load_syllabus(syllabus_file)
    except (FileNotFoundError, ValueError):
        data = load_syllabus("syllabus.json")
    return _extract_topics_from_syllabus(data)

STOPWORDS = {
    'the','a','an','and','or','of','in','to','for','on','at','by','with','from','is','are','be','as','that','this','it','its','into','about','your','you','we','our','their','they','i','he','she','them','his','her','was','were','will','can','could','should','would','may','might'
}

allowed_terms_cache = {}

def _tokenize(text):
    words = re.split(r"[^A-Za-z0-9_]+", str(text).lower())
    return {w for w in words if len(w) >= 3 and w not in STOPWORDS}

def get_allowed_terms(course):
    key = (course or 'ist256').lower()
    if key in allowed_terms_cache:
        return allowed_terms_cache[key]
    syllabus_file = 'hcdd340_syllabus.json' if key == 'hcdd340' else 'syllabus.json'
    try:
        data = load_syllabus(syllabus_file)
    except (FileNotFoundError, ValueError):
        data = load_syllabus('syllabus.json')
    terms = set()
    def walk(v):
        if isinstance(v, dict):
            for k, val in v.items():
                terms.update(_tokenize(k))
                walk(val)
        elif isinstance(v, list):
            for item in v:
                walk(item)
        else:
            terms.update(_tokenize(v))
    walk(data)
    allowed_terms_cache[key] = terms
    return terms

def is_general_response(query):
    return query.lower() in ["hi", "hello", "hey", "yo", "sup", "help", "yes", "no", "maybe", "okay", "sure", "thanks", "thank you"]

def extract_week_number(text):
    m = re.search(r"week\s*(\d+)", text.lower())
    return f"week {m.group(1)}" if m else None

def is_query_allowed(query, course, session_id, email):
    # Let the LLM handle all scope policing based on context and conversational history
    # The system prompt strictly tells it to decline out of scope questions.
    return True

# Main chat responder, now accepts `course`
def get_chat_response(user_question, session_id, email, course="ist256"):
    # 1. Load correct syllabus JSON
    syllabus_file = "hcdd340_syllabus.json" if course == "hcdd340" else "syllabus.json"
    try:
        syllabus = load_syllabus(syllabus_file)
    except (FileNotFoundError, ValueError):
        syllabus = {}

    # 2. Prepare prompts and limited conversation memory
    syllabus_str = flatten_syllabus_text(syllabus) if syllabus else 'No syllabus information is available right now.'
    instruction_str = flatten_instruction_text(course_instruction)

    if not conversation_buffers[session_id] and email:
        db_history = get_chat_history(email, course)
        if db_history:
            for row in db_history[-MAX_HISTORY_TURNS:]:
                conversation_buffers[session_id].append((row[0], row[1]))

    history_pairs = list(conversation_buffers[session_id])
    if history_pairs:
        history_str = "Prior Conversation History:\n" + "\n\n".join([f"Student: {u}\nTutor: {b}" for (u, b) in history_pairs])
    else:
        history_str = "No prior conversation history in this session."

    system_prompt = f"""
You are a { 'HCDD 340 Tutor (Mobile Computing)' if course=='hcdd340' else 'Web Development Tutor for IST 256' }.
Your primary role is to guide students through their questions related strictly to the course syllabus and foundational basics enabling those topics.

CRITICAL TUTORING RULES:
1. DO NOT give direct answers or write out full solutions.
2. Teach them instead: ask questions, reason with them, and make them think so they arrive at the answer themselves.
3. Suggest specific steps for them to perform or concepts to review.
4. STRICT SCOPE: If a question is outside the scope of the course syllabus (e.g., general knowledge unrelated to the class), politely decline and actively bring them back to the topic by suggesting a related in-scope topic they could ask about.

Here is the course syllabus (for grounding context):
{syllabus_str}

Here are course policies/instructions:
{instruction_str}

{history_str}

Remember your tutoring rules: Be helpful but Socratic. Guide, don't just solve.

CRITICAL JSON OUTPUT RULE:
You MUST output your ENTIRE response as a raw JSON object with absolutely no markdown code blocks around it. The JSON must have two keys:
1. "response": Your actual chat reply to the user (formatted in HTML/Markdown as requested).
2. "topics_covered": An array of strings containing the EXACT names of the syllabus topics (from the provided syllabus context) that were discussed or taught in this turn. If none, return an empty array.

Example format:
{
  "response": "Let me help you with CSS Selectors...",
  "topics_covered": ["Selectors", "CSS"]
}
""".strip()

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    chain = chat_prompt | model | StrOutputParser()

    q_lower = user_question.lower()
    week_tag = extract_week_number(q_lower)
    matched = None
    if week_tag:
        for key in group_projects:
            if key.lower().startswith(week_tag):
                matched = key
                break

    # 4. Parse the JSON response
    chat_text = response # fallback
    topics_covered = []
    
    try:
        # Extract JSON block using regex if wrapped in backticks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            raw_response = json_match.group(1)
        else:
            raw_response = response.strip()
            
        parsed_json = json.loads(raw_response)
        
        if isinstance(parsed_json, dict):
            chat_text = parsed_json.get("response", response)
            topics_covered = parsed_json.get("topics_covered", [])
            
            # Record the progress to the database
            if topics_covered and email:
                record_topic_progress(email, course, topics_covered)
    except Exception as e:
        # If the model fails to output valid JSON or misses the schema, just use the raw text
        chat_text = response
        print(f"Failed to parse LLM JSON: {e}")
        
    conversation_context["last_bot_message"] = chat_text

    # 5. Format & log
    formatted = convert_code_blocks(chat_text)
    formatted = format_explanation_text(formatted)
    save_chat_log(session_id, user_question, formatted, email, course)

    # 6. Update session conversation buffer
    conversation_buffers[session_id].append((user_question, chat_text))

    return formatted







