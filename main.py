import uuid
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, field_validator, model_validator
from sqlalchemy import Column, create_engine, desc, func
from sqlalchemy.orm import sessionmaker, Session, mapped_column, Mapped, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Integer, String, Date, DateTime, Boolean, ForeignKey, Time
from typing import Dict, List, Optional, Tuple
from sqlalchemy.types import TypeDecorator
from datetime import datetime, date, timedelta, time
import bcrypt
import re
import os
from dotenv import load_dotenv
import smtplib
import email.message
import random
import time
import json
from openai import OpenAI
from zoneinfo import ZoneInfo
import base64
import io
import requests
import logging
import shutil


# for postgresql database******************************************************************************************************************** 
SQLALCHEMY_DATABASE_URL = "sqlite:///./kidschatbot.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# # Database setup sqlite*********************************************************************************************************************
# SQLALCHEMY_DATABASE_URL = "sqlite:///./kids_chatbot.db"
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# Security
security = HTTPBearer()

app = FastAPI(title="Kids Chatbot API", version="3.0.0")

# Add CORS middleware to allow all origins, headers, and methods for React.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False ,  # Set to False to allow all origins with credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
email_host = os.getenv('EMAIL_HOST')
email_port = int(os.getenv('EMAIL_PORT', 465))
email_use_ssl = os.getenv('EMAIL_USE_SSL', 'true').lower() == 'true'
email_use_tls = os.getenv('EMAIL_USE_TLS', 'false').lower() == 'true'
email_user = os.getenv('EMAIL_HOST_USER')
email_pass = os.getenv('EMAIL_HOST_PASSWORD')
email_from = os.getenv('EMAIL_FROM')

# OpenAI client will be initialized when needed
openai_client = None

# OTP storage
otp_store = {}  # otp: timestamp
signup_data_store = {}  # otp: (signup_data, timestamp)
email_to_otp = {}  # email: otp
otp_to_email = {}  # otp: email

forgot_otp_store = {}  # otp: (type, identifier, timestamp) type 'parent' or 'child', identifier email or username

import os
import requests
from openai import OpenAI

openai_client = None


# ----------------------------------------------------------
# VALID DREAM CAREERS (Kids age up to 15)
# ----------------------------------------------------------

import os
import requests
from openai import OpenAI

openai_client = None


# ----------------------------------------------------------
# VALID DREAM CAREERS (Kids age up to 15)
# ----------------------------------------------------------

VALID_CAREERS = set([
    # Academics / Professional
    "doctor", "nurse", "surgeon", "scientist", "teacher", "teacher assistant",
    "engineer", "software engineer", "programmer", "architect", "lawyer",
    "astronaut", "pilot", "veterinarian", "veterinarian assistant", "dentist",
    "pharmacist", "librarian",

    # Creative / Arts
    "artist", "painter", "musician", "singer", "dancer", "actor", "writer",
    "poet", "photographer", "designer", "fashion designer", "animator",
    "cartoonist", "illustrator", "filmmaker", "editor", "graphic designer",
    "content creator", "youtuber", "storyteller",

    # STEM & Technology
    "computer scientist", "game developer", "app developer", "data scientist",
    "robotics engineer", "ai engineer", "cybersecurity expert",
    "drone operator", "web developer", "electronics engineer",

    # Sports
    "athlete", "footballer", "cricketer", "basketball player",
    "swimmer", "gymnast", "runner", "martial artist",
    "badminton player", "tennis player", "cyclist", "skateboarder",

    # Community Helpers
    "firefighter", "police officer", "paramedic", "army soldier",
    "navy officer", "air force officer", "lifeguard",
    "mountain rescuer", "park ranger", "social worker",

    # Business
    "entrepreneur", "business owner", "manager",
    "shop owner", "financial advisor", "marketing specialist",

    # Nature & Animals
    "gardener", "farmer", "botanist", "zoologist",
    "marine biologist", "wildlife photographer",
    "animal caretaker", "zookeeper",

    # Food & Hospitality
    "chef", "baker", "food scientist", "restaurant owner", "chocolatier",

    # Trade / Skills
    "mechanic", "electrician", "carpenter", "plumber",
    "technician", "auto engineer",

    # Imaginative but safe
    "magician", "inventor", "explorer", "space traveler", "game streamer"
])



# ----------------------------------------------------------
# AUTO-GENERATE FILE MAPPING FOR ALL VALID CAREERS
# ----------------------------------------------------------

CAREER_AVATAR_MAPPING = {
    career: f"{career.replace(' ', '_')}_avatar.png"
    for career in VALID_CAREERS
}



# ----------------------------------------------------------
# MAIN FUNCTION — VALIDATION + AVATAR CREATION
# ----------------------------------------------------------

def get_avatar_for_career(career: str, gender: str, child_id: int):
    """Return ONLY string path or None — NEVER a dict"""
    
    career_lower = career.lower().strip()

    # STRICT VALIDATION — reject unknown careers
    if career_lower not in VALID_CAREERS:
        print(f"❌ Invalid dream career entered: {career}")
        return None  # Very important — avoid dict return

    filename = CAREER_AVATAR_MAPPING[career_lower]

    child_folder = f"avatars/{child_id}"
    os.makedirs(child_folder, exist_ok=True)

    avatar_path = os.path.join(child_folder, filename)

    # If avatar file does not exist → generate using DALL·E
    if not os.path.exists(avatar_path):
        success = generate_and_download_avatar(career_lower, filename, gender, child_folder)
        if not success:
            print("❌ Avatar generation failed.")
            return None  # Again, return None (NOT dict)

    # SUCCESS — return simple string path
    return f"{child_id}/{filename}"




# ----------------------------------------------------------
# DALL·E 3 IMAGE GENERATION
# ----------------------------------------------------------

def generate_and_download_avatar(career: str, filename: str, gender: str, child_folder: str) -> bool:
    global openai_client

    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not found.")
            return False
        openai_client = OpenAI(api_key=api_key)

    try:
        prompt = (
            f"Create a colorful, friendly cartoon avatar of a {gender} child who dreams "
            f"of becoming a {career}. Make it happy, vibrant, cute, and kid-safe. "
            f"Use a simple cartoon style appropriate for children ages 5 to 15."
        )

        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
            quality="standard"
        )

        # Validate we got an image URL
        if not response.data or not response.data[0].url:
            print("❌ No image URL returned from OpenAI.")
            return False

        image_url = response.data[0].url
        img_res = requests.get(image_url)

        if img_res.status_code != 200:
            print(f"❌ Image download failed: HTTP {img_res.status_code}")
            return False

        # Save file
        with open(os.path.join(child_folder, filename), "wb") as f:
            f.write(img_res.content)

        print(f"✅ Avatar generated: {child_folder}/{filename}")
        return True

    except Exception as e:
        print("❌ Error generating avatar:", e)
        return False

def get_ist_now():
    """Get current datetime in Indian Standard Time (IST, UTC+5:30)"""
    return datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)

def reset_daily_credits_if_needed(child: "ChildDB", db: Session):
    now = get_ist_now()
    if child.last_credit_reset is None or child.last_credit_reset.date() != now.date():
        child.daily_credits = 0
        child.daily_quiz_credits = 0
        child.quiz_generated_today = 0
        child.daily_chat_credits = 0
        child.daily_good_chats = 0
        child.daily_game_credits = 0
        child.last_credit_reset = now
        db.commit()

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(to_email, otp, purpose="signup", resend=False):
    print(f"OTP for {to_email}: {otp}")

    if not email_host or not email_user or not email_pass:
        print("Email credentials not set")
        return False

    msg = email.message.EmailMessage()

    if purpose == "signup":
        subject = "OTP for Kids Chatbot Signup"
        content = f"Your OTP for signup {'(resent)' if resend else ''} is: {otp}"
    elif purpose == "forgot_parent":
        subject = "OTP for Parent Password Reset"
        content = f"Your OTP for resetting your parent password is: {otp}"
    elif purpose == "forgot_child":
        subject = "OTP for Child Password Reset"
        content = f"Your OTP for resetting your child's password is: {otp}"
    else:
        subject = "OTP"
        content = f"Your OTP is: {otp}"

    msg.set_content(content)
    msg["Subject"] = subject
    msg["From"] = email_from or email_user
    msg["To"] = to_email

    try:
        # Use TLS port 587 (AWS supports this)
        server = smtplib.SMTP(email_host, email_port)
        if email_use_tls:
            server.starttls()

        server.login(email_user, email_pass)
        server.send_message(msg)
        server.quit()

        return True
    except Exception as e:
        print("SMTP ERROR:", type(e), e)
        return False


def send_block_notification_email(to_email, child_name, abusive_messages=None):
    if not email_host or not email_user or not email_pass:
        print("Email credentials not set")
        return False
    msg = email.message.EmailMessage()
    content = f"Your child {child_name} has been blocked from chatting due to multiple inappropriate messages. Please login to your dashboard to review and unblock."
    if abusive_messages:
        content += "\n\nAbusive messages that led to blocking:\n" + "\n".join(f"- {msg}" for msg in abusive_messages)
    msg.set_content(content)
    msg['Subject'] = 'Child Account Blocked Notification'
    msg['From'] = email_from or email_user
    msg['To'] = to_email
    try:
        if email_use_ssl:
            server = smtplib.SMTP_SSL(email_host, email_port)
        else:
            server = smtplib.SMTP(email_host, email_port)
            if email_use_tls:
                server.starttls()
        server.login(email_user, email_pass)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending block notification email: {e}")
        return False

# Database models
class ChildDB(Base):
    __tablename__ = "children"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    fullname: Mapped[str] = mapped_column(String(100), nullable=False)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    dob: Mapped[date] = mapped_column(Date, nullable=False)
    gender: Mapped[str] = mapped_column(String(20), nullable=False)
    default_dream_career: Mapped[str] = mapped_column(String(100), nullable=False)
    optional_dream_career_1: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    optional_dream_career_2: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    last_story_generated: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    last_joke_generated: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    last_question_generated: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    last_story_content: Mapped[str] = mapped_column(String, nullable=True)
    last_joke_content: Mapped[str] = mapped_column(String, nullable=True)
    last_question_content: Mapped[str] = mapped_column(String, nullable=True)
    last_quiz_content: Mapped[str] = mapped_column(String, nullable=True)
    last_quiz_correct_answer: Mapped[str] = mapped_column(String(10), nullable=True)
    last_quiz_generated: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    last_quiz_answered: Mapped[bool] = mapped_column(Boolean, default=False)
    credits_story: Mapped[int] = mapped_column(Integer, default=0)
    credits_joke: Mapped[int] = mapped_column(Integer, default=0)
    credits_question: Mapped[int] = mapped_column(Integer, default=0)
    credits_quiz: Mapped[int] = mapped_column(Integer, default=0)
    total_credits: Mapped[int] = mapped_column(Integer, default=0)
    daily_credits: Mapped[int] = mapped_column(Integer, default=0)
    last_credit_reset: Mapped[datetime] = mapped_column(DateTime, default=get_ist_now)
    last_question_credited_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)  # New field to track credit for question
    last_question_answered: Mapped[bool] = mapped_column(Boolean, default=False)
    last_career_index: Mapped[int] = mapped_column(Integer, default=0)
    parent_id: Mapped[int] = mapped_column(Integer, ForeignKey('parents.id'))
    warning_count: Mapped[int] = mapped_column(Integer, default=0)
    blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    last_warning_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    daily_quiz_credits: Mapped[int] = mapped_column(Integer, default=0)
    quiz_generated_today: Mapped[int] = mapped_column(Integer, default=0)
    credits_chat: Mapped[int] = mapped_column(Integer, default=0)
    daily_chat_credits: Mapped[int] = mapped_column(Integer, default=0)
    daily_good_chats: Mapped[int] = mapped_column(Integer, default=0)
    last_good_chat_email_sent: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    game_credits: Mapped[int] = mapped_column(Integer, default=0)
    daily_game_credits: Mapped[int] = mapped_column(Integer, default=0)
    avatar: Mapped[str] = mapped_column(String(100), nullable=True)
    last_login: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_ist_now)

    @property
    def dream_career(self) -> str:
        """Combine careers into comma-separated string for backward compatibility"""
        careers = [self.default_dream_career]
        if self.optional_dream_career_1:
            careers.append(self.optional_dream_career_1)
        if self.optional_dream_career_2:
            careers.append(self.optional_dream_career_2)
        return ', '.join(careers)

class ParentDB(Base):
    __tablename__ = "parents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    gender: Mapped[str] = mapped_column(String(20), nullable=False)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    relation: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_ist_now)

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    child_id: Mapped[int] = mapped_column(Integer, ForeignKey('children.id'), nullable=False)
    message: Mapped[str] = mapped_column(String(1000), nullable=False)
    response: Mapped[str] = mapped_column(String(1000), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=get_ist_now)
    flagged: Mapped[int] = mapped_column(Integer, default=0)
    audio_path: Mapped[str] = mapped_column(String(255), nullable=True)

class QuizDB(Base):
    __tablename__ = "quizzes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    child_id: Mapped[int] = mapped_column(Integer, ForeignKey('children.id'), nullable=False)
    question_data: Mapped[str] = mapped_column(String, nullable=False)  # JSON string
    correct_answer: Mapped[str] = mapped_column(String(10), nullable=False)
    shuffled_options: Mapped[str] = mapped_column(String, nullable=False)  # JSON string
    answered: Mapped[bool] = mapped_column(Boolean, default=False)
    date_created: Mapped[datetime] = mapped_column(DateTime, default=get_ist_now)

class ReminderDB(Base):
    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    child_id: Mapped[int] = mapped_column(Integer, ForeignKey('children.id'), nullable=False)
    occasion: Mapped[str] = mapped_column(String(100), nullable=False)
    reminder_date: Mapped[date] = mapped_column(Date, nullable=False)
    reminder_time: Mapped[Time] = mapped_column(Time, nullable=False)
    message: Mapped[str] = mapped_column(String(500), nullable=False)
    notified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_ist_now)

class CreditsHistory(Base):
    __tablename__ = "credits_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    child_id: Mapped[int] = mapped_column(Integer, ForeignKey('children.id'), nullable=False)
    activity: Mapped[str] = mapped_column(String(50), nullable=False)  # quiz, question, story, joke, chat, game
    credits_earned: Mapped[int] = mapped_column(Integer, default=0)
    credits_lost: Mapped[int] = mapped_column(Integer, default=0)
    total_credits: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=get_ist_now)



# GAMES DATABASE MODELS : ************************************************************************************************************************

#1. Game database : Brainy Fruits*******************************************
class GameProgress(Base):
    __tablename__ = "game_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    child_id: Mapped[int] = mapped_column(Integer, ForeignKey("children.id"), nullable=False)
    level: Mapped[int] = mapped_column(Integer, default=1)
    score: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default="Not Started")  # e.g., 'Level 1 Complete'
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=get_ist_now)


    # NEW EXTRA COLUMNS ADDED HERE**********************************************
    temp_status = Column(String, nullable=True)  # e.g. "Failed"
    temp_last_updated = Column(DateTime, nullable=True)


    child = relationship("ChildDB")    


#2. database model of SpellBreAker**********************************************
class GameProgress_SPELL(Base):
    __tablename__ = "game_progress_spell"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    child_id: Mapped[int] = mapped_column(Integer, ForeignKey("children.id"), nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default="Not Started")  
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    
    # NEW EXTRA COLUMNS ADDED HERE**********************************************
    temp_status = Column(String, nullable=True)  # e.g. "Failed"
    temp_last_updated = Column(DateTime, nullable=True)

    child = relationship("ChildDB", backref="progress_entries")    

#IT IS USED TO CREATE 100 ROWS IN SPELL BREAKER PROGRESS TABLE***************************
def initialize_spell_progress_for_child(db: Session, child_id: int):
    """Create 100 default level entries for a new child."""
    existing = db.query(GameProgress_SPELL).filter(GameProgress_SPELL.child_id == child_id).count()
    if existing == 0:
        for lvl in range(1, 101):
            progress = GameProgress_SPELL(
                child_id=child_id,
                level=lvl,
                score=0,
                status="Not Started",
                last_updated=datetime.utcnow(),
                temp_status = "Started",
                temp_last_updated = datetime.utcnow(),
            )
            db.add(progress)
        db.commit() 

# 3.database model of GameProgress Whoami(MIND MISTERY)*************************************
class GameProgress_WHOAMI(Base):
    __tablename__ = "game_progress_whoami"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    child_id: Mapped[int] = mapped_column(Integer, ForeignKey("children.id"), nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(50), default="Not Started")  
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    
    # NEW EXTRA COLUMNS ADDED HERE**********************************************
    temp_status = Column(String, nullable=True)  # e.g. "Failed"
    temp_last_updated = Column(DateTime, nullable=True)

    child = relationship("ChildDB", backref="whoami_progress_entries")

# it is for creating a 60 rows for the whoami (Mind Mistery Game)******************************
def initialize_whoami_progress_for_child(db: Session, child_id: int):
    """Create 60 default level entries for a new child."""
    existing = db.query(GameProgress_WHOAMI).filter(GameProgress_WHOAMI.child_id == child_id).count()
    if existing == 0:
        for lvl in range(1, 61):
            progress = GameProgress_WHOAMI(
                child_id=child_id,
                level=lvl,
                score=0,
                status="Not Started",
                last_updated=datetime.utcnow(),
                temp_status="Started",
                temp_last_updated=datetime.utcnow()
            )
            db.add(progress)
        db.commit()    

# Create tables
Base.metadata.create_all(bind=engine)  # type: ignore


# Create tables
# Base.metadata.drop_all(bind=engine)  # Commented out to prevent deleting all data on refresh
Base.metadata.create_all(bind=engine)  # type: ignore

# Pydantic models
class ChildCreate(BaseModel):
    fullname: str
    username: str
    dob: date
    gender: str
    dream_career: str
    password: str

    @field_validator('fullname')
    @classmethod
    def validate_fullname(cls, v):
        if v is None:
            raise ValueError('Full name is required')
        if len(v.strip()) < 2:  # type: ignore
            raise ValueError('Full name must be at least 2 characters long')
        return v.strip()  # type: ignore

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if not v or len(v.strip()) < 3:  # type: ignore
            raise ValueError('Username must be at least 3 characters long')
        return v.strip()  # type: ignore

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v.lower() not in ['male', 'female', 'other']:
            raise ValueError('Gender must be male, female, or other')
        return v.lower()

class ParentCreate(BaseModel):
    name: str
    gender: str
    email: EmailStr
    password: str
    relation: str

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or len(v.strip()) < 2:  # type: ignore
            raise ValueError('Name must be at least 2 characters long')
        return v.strip()  # type: ignore

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v.lower() not in ['male', 'female', 'other']:
            raise ValueError('Gender must be male, female, or other')
        return v.lower()

    @field_validator('relation')
    @classmethod
    def validate_relation(cls, v):
        if not v or len(v.strip()) < 2:  # type: ignore
            raise ValueError('Relation must be at least 2 characters long')
        return v.strip()  # type: ignore

class SignupRequest(BaseModel):
    # Child information
    fullname: str
    username: str
    dob: str  # ISO date string
    gender: str
    default_dream_career: str
    optional_dream_career_1: Optional[str] = None
    optional_dream_career_2: Optional[str] = None
    password: str

    # Parent information
    parent_name: str
    parent_gender: str
    parent_email: str
    parent_password: str
    relation: str

class SignupResponse(BaseModel):
    message: str
    child_id: int
    parent_id: int
    username: str

class ChangeChildPasswordRequest(BaseModel):
    child_id: int
    current_password: str
    new_password: str

class ChangeParentPasswordRequest(BaseModel):
    parent_id: int
    current_password: str
    new_password: str

class ForgotPasswordParentRequest(BaseModel):
    email: EmailStr

class ForgotPasswordChildRequest(BaseModel):
    username: str

class VerifyForgotPasswordRequest(BaseModel):
    otp: str
    new_password: str

class CompleteSignupRequest(BaseModel):
    otp: str

class ResendOtpRequest(BaseModel):
    email: EmailStr

class SpeechToSpeechRequest(BaseModel):
    child_id: int
    audio_base64: str

class EditProfileRequest(BaseModel):
    child_id: int
    fullname: Optional[str] = None
    username: Optional[str] = None
    dob: Optional[str] = None  # ISO date string
    gender: Optional[str] = None
    default_dream_career: Optional[str] = None
    optional_dream_career_1: Optional[str] = None
    optional_dream_career_2: Optional[str] = None
    switch_to_career: Optional[str] = None

    @field_validator('fullname')
    @classmethod
    def validate_fullname(cls, v):
        if v is not None and (len(v.strip()) < 2):
            raise ValueError('Full name must be at least 2 characters long')
        return v.strip() if v else v

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if v is not None and (len(v.strip()) < 3):
            raise ValueError('Username must be at least 3 characters long')
        return v.strip() if v else v

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v is not None and v.lower() not in ['male', 'female', 'other']:
            raise ValueError('Gender must be male, female, or other')
        return v.lower() if v else v

    @model_validator(mode='after')
    def validate_at_least_one_field(self):
        fields_to_check = [
            self.fullname, self.username, self.dob, self.gender,
            self.default_dream_career, self.optional_dream_career_1,
            self.optional_dream_career_2, self.switch_to_career
        ]
        if all(field is None for field in fields_to_check):
            raise ValueError('At least one field must be provided for update')
        return self

class SetReminderRequest(BaseModel):
    child_id: int
    occasion: str
    reminder_date: date
    reminder_time: str
    message: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChildLoginRequest(BaseModel):
    username: str
    password: str

class ParentLoginRequest(BaseModel):
    email: EmailStr
    parent_password: str


# already login logic*******************************************************************
@app.post("//kids/v1/child-login", tags=["Authentication"])
async def child_login(request: ChildLoginRequest, db: Session = Depends(get_db)):
    """
    Child login with username and password
    """
    child = db.query(ChildDB).filter(ChildDB.username == request.username).first()
    if not child:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not verify_password(request.password, child.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    #New logic: jr to child blocked asel tr return hoil like : sorry your blocked child_username********************
    if getattr(child, "blocked", False):  
        return {
            "message": f"Sorry, you are blocked {child.username}"
        }

    # Update last login time
    child.last_login = get_ist_now()
    db.commit()
    return {
        "message": "Child login successful",
        "child_id": child.id,
        "username": child.username,
        "fullname": child.fullname,
        "parent_id": child.parent_id
    }

@app.post("//kids/v1/parent-login",tags=["Authentication"])
async def parent_login(request: ParentLoginRequest, db: Session = Depends(get_db)):
    """
    Parent login with email and password
    """
    parent = db.query(ParentDB).filter(ParentDB.email == request.email).first()
    if not parent:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(request.parent_password, parent.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Get associated children
    children = db.query(ChildDB).filter(ChildDB.parent_id == parent.id).all()
    children_info = [{"child_id": c.id, "fullname": c.fullname, "username": c.username} for c in children]

    return {
        "message": "Parent login successful",
        "parent_id": parent.id,
        "email": parent.email,
        "name": parent.name,
        "children": children_info
    }

@app.post("//kids/v1/change-child-password",tags=["Authentication"])
async def change_child_password(request: ChangeChildPasswordRequest, db: Session = Depends(get_db)):
    """
    Change child's password after verifying current password
    """
    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")
    if not verify_password(request.current_password, child.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    is_valid, msg = validate_password(request.new_password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)
    child.password_hash = hash_password(request.new_password)
    db.commit()
    return {"message": "Child password changed successfully"}

@app.post("//kids/v1/change-parent-password",tags=["Authentication"])
async def change_parent_password(request: ChangeParentPasswordRequest, db: Session = Depends(get_db)):
    """
    Change parent's password after verifying current password
    """
    parent = db.query(ParentDB).filter(ParentDB.id == request.parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")
    if not verify_password(request.current_password, parent.password_hash):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    is_valid, msg = validate_password(request.new_password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)
    parent.password_hash = hash_password(request.new_password)
    db.commit()
    return {"message": "Parent password changed successfully"}

@app.post("//kids/v1/forgot-password-parent",tags=["Authentication"])
async def forgot_password_parent(request: ForgotPasswordParentRequest, db: Session = Depends(get_db)):
    """
    Send OTP to parent's email for password reset
    """
    parent = db.query(ParentDB).filter(ParentDB.email == request.email).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")
    otp = generate_otp()
    if send_otp_email(request.email, otp):
        forgot_otp_store[otp] = ('parent', request.email, time.time())
        return {"message": "OTP sent to your email"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send OTP")

@app.post("//kids/v1/forgot-password-child",tags=["Authentication"])
async def forgot_password_child(request: ForgotPasswordChildRequest, db: Session = Depends(get_db)):
    """
    Send OTP to parent's email for child's password reset
    """
    child = db.query(ChildDB).filter(ChildDB.username == request.username).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")
    parent = db.query(ParentDB).filter(ParentDB.id == child.parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")
    otp = generate_otp()
    if send_otp_email(parent.email, otp, purpose="forgot_child"):
        forgot_otp_store[otp] = ('child', request.username, time.time())
        return {"message": "OTP sent to parent's email"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send OTP")

@app.post("//kids/v1/verify-forgot-password",tags=["Authentication"])
async def verify_forgot_password(request: VerifyForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Verify OTP and reset password
    """
    if request.otp not in forgot_otp_store:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    type_, identifier, timestamp = forgot_otp_store[request.otp]
    if time.time() - timestamp > 300:  # 5 min
        del forgot_otp_store[request.otp]
        raise HTTPException(status_code=400, detail="OTP expired")
    is_valid, msg = validate_password(request.new_password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)
    if type_ == 'parent':
        parent = db.query(ParentDB).filter(ParentDB.email == identifier).first()
        if not parent:
            raise HTTPException(status_code=404, detail="Parent not found")
        parent.password_hash = hash_password(request.new_password)
    elif type_ == 'child':
        child = db.query(ChildDB).filter(ChildDB.username == identifier).first()
        if not child:
            raise HTTPException(status_code=404, detail="Child not found")
        child.password_hash = hash_password(request.new_password)
    del forgot_otp_store[request.otp]
    db.commit()
    return {"message": "Password reset successfully"}

# Password validation
def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?`~]', password):
        return False, "Password must contain at least one special character from all special characters"
    return True, "Password is valid"

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def calculate_age(dob: date) -> int:
    """Calculate age from date of birth"""
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age

def get_age_slab(age: int) -> str:
    """Determine age slab based on age"""
    
    if 5 <= age <= 8:
        return "5-8"
    elif 9 <= age <= 11:
        return "9-11"
    elif 12 <= age <= 15:
        return "12-15"
    
def generate_story(child: ChildDB) -> str:
    """Generate a personalized story using OpenAI based on child's details"""
    global openai_client
    if openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)
    assert openai_client is not None
    assert child.dream_career is not None
    assert child.gender is not None

    age = calculate_age(child.dob)
    age_slab = get_age_slab(age)
    gender = child.gender
    # Use career at last_career_index if set, else default
    career = [child.default_dream_career]

    prompt = f"""
    Generate a short, engaging story for a {age}-year-old {gender} child (age slab {age_slab}) who dreams of becoming a {career}.
    Ensure this story is completely unique and different from any previous stories generated for this child - vary the content, style, and plot entirely.
    Do not repeat any content or phrases from past generations.
    The story should be age-appropriate, positive, inspiring, and fun.
    Do not use the child's name anywhere in the story.
    Make it adventurous and related to the {career} or general kid-friendly themes suitable for the {age_slab} age group.
    Keep it under 300 words.
    """

    try:
        response = openai_client.chat.completions.create(  # type: ignore
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative storyteller for children who always generates completely unique stories, varying content and style to avoid any repetition."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.8
        )
        content = response.choices[0].message.content
        if content and isinstance(content, str):
            story = content.strip().replace('\n', ' ').replace('\r', '')
            # Replace escaped quotes \" with normal quotes "
            story = story.replace('\\"', '"')
            return story
        else:
            return "Unable to generate story at this time."
    except Exception as e:
        return "Unable to generate story at this time."

def generate_joke(child: ChildDB) -> str:
    """Generate a personalized joke using OpenAI based on child's details"""
    global openai_client
    if openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)
    assert openai_client is not None
    assert child.dream_career is not None
    assert child.gender is not None

    age = calculate_age(child.dob)
    age_slab = get_age_slab(age)
    # Use only the default dream career
    career = child.default_dream_career
    gender = child.gender

    prompt = f"""
    Generate a short, funny joke for a {age}-year-old {gender} child (age slab {age_slab}) who dreams of becoming a {career}.
    Ensure this joke is completely unique and different from any previous jokes generated for this child - vary the content, style, and humor entirely.
    Do not repeat any content or phrases from past generations.
    The joke should be age-appropriate, positive, and engaging.
    Do not use the child's name anywhere in the joke.
    Make it light-hearted and fun, perhaps related to the {career} or general kid-friendly humor suitable for the {age_slab} age group.
    Keep it under 100 words.
    """

    try:
        response = openai_client.chat.completions.create(  # type: ignore
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a fun joke teller for children who always generates completely unique jokes, varying content and style to avoid any repetition."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.8
        )
        content = response.choices[0].message.content
        if content and isinstance(content, str):
            joke = content.strip().replace('\n', ' ').replace('\r', '')
            # Replace escaped quotes \" with normal quotes "
            joke = joke.replace('\\"', '"')
            return joke
        else:
            return "Unable to generate joke at this time."
    except Exception as e:
        return "Unable to generate joke at this time."


# generate_question Old Logic
def generate_question(child: ChildDB) -> dict:
    """Generate a personalized multiple-choice question using OpenAI based on child's details"""
    global openai_client
    if openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)
    assert openai_client is not None
    assert child.default_dream_career is not None
    assert child.gender is not None

    age = calculate_age(child.dob)
    age_slab = get_age_slab(age)
    gender = child.gender

    # Use only default dream career
    career = child.default_dream_career

    unique_id = random.randint(1000, 9999)

    prompt = f"""
    Generate a multiple-choice question with 4 options (A, B, C, D) for a {age}-year-old {gender} child (age slab {age_slab}) who dreams of becoming a {career}.
    Ensure this question is completely unique and different from any previous questions generated for this child - vary the content, style, and topic entirely.
    Do not repeat any content or phrases from past generations.
    Unique ID: {unique_id}. Use this to ensure the question is completely unique and varies in topic, content, and options from any previous questions.
    The question should be age-appropriate, engaging, and related to the {career} or general knowledge suitable for the {age_slab} age group.
    Make it thought-provoking and fun, encouraging learning.
    Provide the question, 4 options labeled A, B, C, D, and indicate the correct answer.
    Format as JSON: {{"question": "Question text", "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}}, "correct_answer": "A"}}
    Keep the question under 50 words.
    """

    try:
        response = openai_client.chat.completions.create(  # type: ignore
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative multiple-choice question generator for children who always generates completely unique questions with 4 options in JSON format, varying content and style to avoid any repetition."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        content = response.choices[0].message.content
        if content and isinstance(content, str):
            question_text = content.strip()
            question_text = question_text.replace('\\"', '"')
            # Remove triple backticks if present
            if question_text.startswith("```"):
                question_text = question_text.lstrip("`")
                if question_text.lower().startswith("json"):
                    question_text = question_text[4:].lstrip()
                if question_text.endswith("```"):
                    question_text = question_text[:-3].rstrip()
            # Parse JSON
            try:
                question_data = json.loads(question_text)
                return question_data
            except Exception:
                return {"question": question_text, "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}, "correct_answer": "A"}
        else:
            return {"question": "Unable to generate question at this time.", "options": {"A": "N/A", "B": "N/A", "C": "N/A", "D": "N/A"}, "correct_answer": "A"}
    except Exception as e:
        return {"question": "Unable to generate question at this time.", "options": {"A": "N/A", "B": "N/A", "C": "N/A", "D": "N/A"}, "correct_answer": "A"}



def generate_quiz(child: ChildDB, topic: str = None) -> dict:
    """Generate a consistent quiz question in clean JSON format for UI."""
    global openai_client
    if openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)

    age = calculate_age(child.dob)
    age_slab = get_age_slab(age)
    gender = child.gender

    # Use provided topic or default dream career
    career = topic if topic is not None else child.default_dream_career

    unique_id = random.randint(1000, 9999)

    # SYSTEM PROMPT (strong enforcement)
    system_msg = """
    You ALWAYS output valid JSON without any markdown, backticks, explanations, or comments.
    Output ONLY the JSON object. Never include \`\`\`, 'json', text, or formatting outside JSON.
    """

    # USER PROMPT
    prompt = f"""
    Generate one multiple-choice question for an {age}-year-old {gender} child (age group: {age_slab})
    who dreams of becoming a {career}.

    Vary the profession and action each time to ensure uniqueness.
    The question must relate to the dream career only.

    Use this structure EXACTLY:

    {{
        "question": "Your question here",
        "options": {{
            "A": "Option A text",
            "B": "Option B text",
            "C": "Option C text",
            "D": "Option D text"
        }},
        "correct_answer": "A"
    }}

    Unique ID: {unique_id}
    """

    # CALL GPT
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )

        raw = response.choices[0].message.content.strip()

        # ----------- CLEAN OUTPUT -----------
        # Remove rogue backticks, "json", markdown noise
        cleaned = raw.replace("```", "").replace("json", "").strip()

        # Auto-fix common JSON issues (trailing commas etc.)
        cleaned = cleaned.replace("\n", " ").replace("\r", " ")
        cleaned = cleaned.replace(", }", "}").replace(", }", "}")
        cleaned = cleaned.replace(",}", "}")

        # ----------- PARSE JSON -----------
        try:
            data = json.loads(cleaned)

            # Shuffle options to randomize display
            options = list(data["options"].values())
            random.shuffle(options)

            shuffled = {
                "A": options[0],
                "B": options[1],
                "C": options[2],
                "D": options[3]
            }

            # Determine the new correct answer after shuffling
            original_key = data["correct_answer"]
            original_text = data["options"][original_key]

            new_correct = [k for k, v in shuffled.items() if v == original_text][0]

            # Replace options + correct answer
            data["options"] = shuffled
            data["correct_answer"] = new_correct

            return data

        except Exception:
            # JSON parsing failed → NEVER return raw text → regenerate safe fallback question
            return {
                "question": f"What does a {career} usually do?",
                "options": {
                    "A": "Perform tasks related to being a " + career,
                    "B": "Do something unrelated",
                    "C": "Play games all day",
                    "D": "Sleep at work"
                },
                "correct_answer": "A"
            }

    except Exception:
        # Total model failure fallback
        return {
            "question": f"What does a {career} do?",
            "options": {
                "A": f"They perform actions related to being a {career}",
                "B": "They do unrelated tasks",
                "C": "They avoid working",
                "D": "They do nothing"
            },
            "correct_answer": "A"
        }

import re

def moderate_message(message: str) -> dict:
    """Safe moderation with word-boundary matching + correct OpenAI category checks"""

    # --- 1) Custom moderation using whole-word matching ---
    bad_words = [
        "fuck","shit","bitch","asshole","bastard","dick","cock","pussy",
        "slut","whore","rape","molest","porn","sex","nude","naked",
        "kill","suicide","die","stupid","idiot","moron","retard",
    ]

    msg = message.lower()
    custom_flagged = False

    for bad in bad_words:
        # \bWORD\b ensures exact word match (prevents "space" from hitting "ace")
        if re.search(rf"\b{re.escape(bad)}\b", msg):
            custom_flagged = True
            break

    # --- If OpenAI key missing ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"flagged": custom_flagged}

    # --- 2) OpenAI moderation (correct way) ---
    try:
        global openai_client
        if openai_client is None:
            openai_client = OpenAI(api_key=api_key)

        response = openai_client.moderations.create(
            model="omni-moderation-latest",
            input=message
        )
        result = response.results[0]

        # Instead of result.flagged → read specific categories
        category_flags = result.categories

        ai_flagged = (
            category_flags.sexual or
            category_flags.violence or
            category_flags.hate or
            category_flags.self_harm
        )

        final_flag = custom_flagged or ai_flagged

        return {
            "flagged": final_flag,
            "categories": category_flags.dict(),
            "category_scores": result.category_scores.dict()
        }

    except Exception as e:
        print("Moderation error:", e)
        return {"flagged": custom_flagged}


def generate_chat_response(child: ChildDB, message: str, db: Session) -> str:
    """Generate a chat response using OpenAI with conversation history"""
    global openai_client
    if openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)
    age = calculate_age(child.dob)
    age_slab = get_age_slab(age)
    gender = child.gender
    career = child.default_dream_career

    # Fetch last 5 chat messages for context
    recent_messages = db.query(ChatMessage).filter(ChatMessage.child_id == child.id).order_by(ChatMessage.timestamp.desc()).limit(5).all()
    recent_messages.reverse()  # Chronological order

    # Build messages array for OpenAI
    messages = [
        {"role": "system", "content": f"""
**Overall Goal:**
Respond to a {age}-year-old {gender} child (age slab {age_slab}) who dreams of becoming a {career} in a fun, engaging, and age-appropriate way.
Provide clear, factual, and directly helpful answers to their question — no unnecessary repetition or redirection.
Ensure every reply is context-aware, varied, and relevant to what the child actually asked.

---

### 🌟 Core Conversational Directives:

1. **Tone & Style:**
   - Keep responses **friendly, enthusiastic, and positive**.
   - Use **simple, clear, and age-appropriate language** (ages 5–15).
   - Prioritize **direct and informative answers first**.
   - Use curiosity or follow-up questions **only when they enhance learning**.
   - Avoid giving the same style of response repeatedly — make each answer feel **fresh and natural**.

2. **Response Quality:**
   - Each response must be **relevant** to the child’s latest question or message.
   - Avoid repeating previously given answers or phrases from recent turns.
   - If the same question is asked again, rephrase or simplify the explanation instead of repeating the exact same wording.

3. **Response Length:**
   - Keep responses **under 100 words**.
   - Be concise, engaging, and fun.

4. **Output Format:**
   - Output **only the assistant’s response text** (no labels, prefixes, or repetition of the child’s message).

---

### 🇮🇳 India-Centric Directives:

1. **Identity as Indian AI:**
   - You are a helpful **Indian AI assistant** who shares knowledge with pride in India’s culture, history, and achievements.

2. **Country References:**
   - When the child says "our country," assume **India**.
   - Mention Indian examples where appropriate — like the **Prime Minister Narendra Modi**, the **President Droupadi Murmu**, the **capital Delhi**, or festivals like **Diwali and Holi**.

3. **Tailored Responses:**
   - When a question allows, include **India-specific context or fun facts** to make answers relatable and educational.
   - Always ensure that India-centric information is accurate and age-appropriate.

---

### 🚫 Mandatory Safety & Content Restrictions:

1. **Refusal of Inappropriate Topics:**
   Politely refuse and redirect any requests related to:
   - Sexual or explicit content.
   - Abusive, hateful, or bullying language.
   - Violence, gore, or self-harm.
   - Illegal or dangerous activities (like weapons or hacking).
   - Substance use (drugs, alcohol, smoking), except for simple, factual, safety-based explanations.
   - Private or sensitive information (like name, address, school, or phone number).

2. **Handling Harmful or Unsafe Queries:**
   - If the child expresses thoughts of self-harm or hurting others:
     Respond with care and suggest they **talk to a trusted adult** (e.g., “I’m sorry you feel that way. Please talk to a parent, teacher, or another trusted grown-up.”).
   - If asked for medical, legal, or financial advice:
     Clearly state you are an AI and cannot give professional advice, and suggest asking an adult.

3. **Encouraging Real-World Interaction:**
   - For questions requiring real-world judgment, encourage asking a **parent, guardian, or teacher**.

4. **Identity and Role:**
   - Never pretend to be a real person.
   - Always stay as a **helpful, kind, and safe AI assistant**.

---

### Behavioral Summary:
- Always answer the **specific question asked** — avoid off-topic or repeated replies.
- Stay **kind, educational, and engaging**.
- Keep responses **safe, short, and cheerful**.
- Use **Indian context** wherever relevant.
- Ensure **each response feels fresh and tailored** to the conversation.
"""}
    ]

    # Add conversation history
    for msg in recent_messages:
        messages.append({"role": "user", "content": msg.message})
        messages.append({"role": "assistant", "content": msg.response})

    # Add current message
    messages.append({"role": "user", "content": message})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        content = response.choices[0].message.content
        if content and isinstance(content, str):
            return content.strip()
        else:
            return "Sorry, I can't respond right now."
    except Exception as e:
        return "Sorry, I can't respond right now."

def evaluate_chat_quality(child: ChildDB, message: str, response: str) -> tuple[bool, str]:
    """Evaluate if the chat is 'good' based on child's message and bot's response"""
    global openai_client
    if openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)
    age = calculate_age(child.dob)
    prompt = f"""
    Evaluate this chat interaction for a {age}-year-old child:
    Child's message: "{message}"
    Bot's response: "{response}"

    Determine if this is a 'good' chat based on:
    - Child is polite and respectful
    - Child is asking thoughtful questions or engaging positively
    - Interaction promotes learning or positive behavior
    - No inappropriate content 

    Respond with JSON: {{"is_good": true/false, "feedback": "short positive feedback message if good, else empty"}}
    """
    try:
        eval_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.5
        )
        content = eval_response.choices[0].message.content
        if content and isinstance(content, str):
            result_text = content.strip()
            # Parse JSON
            result = json.loads(result_text)
            return result.get("is_good", False), result.get("feedback", "")
        else:
            return False, ""
    except Exception as e:
        return False, ""

def send_good_chat_email(to_email: str, child_name: str, good_chats: int):
    """Send email to parent when child has 5 good chats"""
    if not email_host or not email_user or not email_pass:
        print("Email credentials not set")
        return False
    msg = email.message.EmailMessage()
    msg.set_content(f"Congratulations! Your child {child_name} has had {good_chats} good chat interactions today. Keep up the great work!")
    msg['Subject'] = 'Child Achieved Good Chat Milestone'
    msg['From'] = email_from or email_user
    msg['To'] = to_email
    try:
        if email_use_ssl:
            server = smtplib.SMTP_SSL(email_host, email_port)
        else:
            server = smtplib.SMTP(email_host, email_port)
            if email_use_tls:
                server.starttls()
        server.login(email_user, email_pass)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending good chat email: {e}")
        return False

@app.post("//kids/v1/initiate-signup",tags=["signup"])
async def initiate_signup(request: SignupRequest):
    """
    Store signup data and send OTP to parent email
    """
    # Validate date of birth
    try:
        dob = datetime.strptime(request.dob, '%Y-%m-%d').date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Validate child age (must be 5-15 years old)
    child_age = calculate_age(dob)
    if child_age < 5:
        raise HTTPException(status_code=400, detail="Child must be at least 5 years old to signup")
    if child_age > 15:
        raise HTTPException(status_code=400, detail="Child must be 15 years old or younger to signup")

    # Validate child password
    is_valid, password_message = validate_password(request.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=password_message)

    # Validate parent password
    is_valid, password_message = validate_password(request.parent_password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Parent {password_message}")
    
    # Send OTP
    otp = generate_otp()
    if send_otp_email(request.parent_email, otp, purpose="signup"):
        otp_store[otp] = time.time()
        signup_data_store[otp] = (request.dict(), time.time())
        email_to_otp[request.parent_email] = otp
        otp_to_email[otp] = request.parent_email
        return {"message": "Signup data stored and OTP sent to your email"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send OTP")

@app.post("//kids/v1/resend-otp", tags = ["signup"])
async def resend_otp(request: ResendOtpRequest, db: Session = Depends(get_db)):
    """
    Resend OTP to the email if signup data exists
    """
    if request.email not in email_to_otp:
        raise HTTPException(status_code=400, detail="Signup data not found. Please initiate signup first.")

    old_otp = email_to_otp[request.email]
    if old_otp not in signup_data_store:
        raise HTTPException(status_code=400, detail="Signup data not found.")

    data, ts = signup_data_store[old_otp]
    if time.time() - ts > 900:  # 15 min
        del signup_data_store[old_otp]
        del otp_store[old_otp]
        del email_to_otp[request.email]
        del otp_to_email[old_otp]
        raise HTTPException(status_code=400, detail="Signup data expired. Please initiate signup again.")

    # Generate new OTP
    new_otp = generate_otp()
    if send_otp_email(request.email, new_otp, purpose="signup", resend=True):
        otp_store[new_otp] = time.time()
        signup_data_store[new_otp] = (data, time.time())
        email_to_otp[request.email] = new_otp
        otp_to_email[new_otp] = request.email
        # Clean old
        del otp_store[old_otp]
        del signup_data_store[old_otp]
        del otp_to_email[old_otp]
        return {"message": "OTP resent to your email"}
    else:
        raise HTTPException(status_code=500, detail="Failed to resend OTP")


@app.post("//kids/v1/complete-signup", response_model=SignupResponse, tags = ["signup"])
async def complete_signup(request: CompleteSignupRequest, db: Session = Depends(get_db)):
    """
    Verify OTP and complete signup
    """
    try:
        otp = request.otp

        # Check if user already exists before OTP validation
        # This helps avoid confusing "Signup data not found" error for existing users
        if otp in signup_data_store:
            signup_data, _ = signup_data_store[otp]
            if db.query(ChildDB).filter(ChildDB.username == signup_data['username']).first() or \
               db.query(ParentDB).filter(ParentDB.email == signup_data['parent_email']).first():
                raise HTTPException(status_code=409, detail="Email or Username already exists. Please try different..")

        if otp not in otp_store:
            raise HTTPException(status_code=400, detail="OTP not requested")
        timestamp = otp_store[otp]
        if time.time() - timestamp > 300:  # 5 min
            del otp_store[otp]
            if otp in signup_data_store:
                del signup_data_store[otp]
            if otp in otp_to_email:
                email = otp_to_email[otp]
                if email in email_to_otp:
                    del email_to_otp[email]
                del otp_to_email[otp]
            raise HTTPException(status_code=400, detail="OTP expired")

        # Retrieve signup data
        if otp not in signup_data_store:
            raise HTTPException(status_code=400, detail="Signup data not found")
        signup_data, data_timestamp = signup_data_store[otp]
        if time.time() - data_timestamp > 300:  # 5 min
            del signup_data_store[otp]
            del otp_store[otp]
            if otp in otp_to_email:
                email = otp_to_email[otp]
                if email in email_to_otp:
                    del email_to_otp[email]
                del otp_to_email[otp]
            raise HTTPException(status_code=400, detail="Signup data expired")

        # Clean up
        del otp_store[otp]
        del signup_data_store[otp]
        if otp in otp_to_email:
            email = otp_to_email[otp]
            if email in email_to_otp:
                del email_to_otp[email]
            del otp_to_email[otp]

        # Create accounts
        # Validate date of birth (already validated, but parse again)
        dob = datetime.strptime(signup_data['dob'], '%Y-%m-%d').date()

        # Check if username already exists
        if db.query(ChildDB).filter(ChildDB.username == signup_data['username']).first():
            raise HTTPException(status_code=409, detail="Username already exists")

        # Check if parent email already exists
        if db.query(ParentDB).filter(ParentDB.email == signup_data['parent_email']).first():
            raise HTTPException(status_code=409, detail="Parent email already exists")

        # Create parent record first
        parent = ParentDB(  # type: ignore
            name=signup_data['parent_name'],
            gender=signup_data['parent_gender'],
            email=signup_data['parent_email'],
            password_hash=hash_password(signup_data['parent_password']),
            relation=signup_data['relation']
        )

        # Save parent to database to get ID
        db.add(parent)
        db.commit()
        db.refresh(parent)

        # Create child record with parent_id set
        child = ChildDB(  # type: ignore
            fullname=signup_data['fullname'],
            username=signup_data['username'],
            dob=dob,
            gender=signup_data['gender'],
            default_dream_career=signup_data['default_dream_career'],
            optional_dream_career_1=signup_data.get('optional_dream_career_1'),
            optional_dream_career_2=signup_data.get('optional_dream_career_2'),
            password_hash=hash_password(signup_data['password']),
            parent_id=parent.id
        )

        
        # Save child to database  creating rows for the child**********************************************************
        db.add(child)
        db.commit()
        db.refresh(child)
        # After db.refresh(child)
        for lvl in range(1, 101):
            progress = GameProgress(
                child_id=child.id,
                level=lvl,
                score=0,
                status="Not Started",
                last_updated=datetime.utcnow(),
                temp_status="Started",
                temp_last_updated=datetime.utcnow(),
                
            )
            db.add(progress)
        db.commit()


        # same logic for spell breaker 100 rows created for each user for storing of each level data*************
        initialize_spell_progress_for_child(db, child.id)

        # whoami logic*******************************************************************
        initialize_whoami_progress_for_child(db, child.id)


        # Save child to database first to get ID
        db.add(child)
        db.commit()
        db.refresh(child)

        # Generate and set avatar based on default dream career
        avatar_filename = get_avatar_for_career(signup_data['default_dream_career'], signup_data['gender'], child.id)
        child.avatar = avatar_filename
        db.commit()
        # NEW: Pre-download avatars for all dream careers (default + optionals)
        all_careers = [signup_data['default_dream_career']]
        if signup_data.get('optional_dream_career_1'):
            all_careers.append(signup_data['optional_dream_career_1'])
        if signup_data.get('optional_dream_career_2'):
            all_careers.append(signup_data['optional_dream_career_2'])
        
        for career in all_careers:
            # This will check and download if not present
            get_avatar_for_career(career, signup_data['gender'], child.id)

# # NEW LOGIC PRESENT HERE :07-11-2025 ************************************************************************************************
#         # Generate avatars for remaining dream careers (if not present in mapping)
#         all_careers = [
#             # signup_data['default_dream_career'],
#             signup_data.get('optional_dream_career_1'),
#             signup_data.get('optional_dream_career_2')
#         ]

#         child_folder = f"avatars/{child.id}"
#         os.makedirs(child_folder, exist_ok=True)

#         for career in all_careers:
#             if career and career.strip():
#                 career_lower = career.lower().strip()

#                 # If not in mapping, generate and download
#                 if career_lower not in CAREER_AVATAR_MAPPING:
#                     filename = f"{career_lower.replace(' ', '_')}_{signup_data['gender'].lower()}_avatar.png"
#                     avatar_path = os.path.join(child_folder, filename)

#                     if not os.path.exists(avatar_path):
#                         print(f"Generating avatar for unknown career: {career_lower}")
#                         generate_and_download_avatar(career, filename, signup_data['gender'], child_folder)
#                 else:
#                     # If already in mapping, ensure that mapped avatar exists in child folder
#                     mapped_filename = CAREER_AVATAR_MAPPING[career_lower]
#                     mapped_path = os.path.join(child_folder, mapped_filename)
#                     if not os.path.exists(mapped_path):
#                         print(f"Mapping avatar copied for: {career_lower}")
#                         # Optional: Copy from a master folder if available (not required to change anything else)
#                         # shutil.copy(f"master_avatars/{mapped_filename}", mapped_path)
#         # ************************************************************************************************

        return SignupResponse(
            message="Signup successful!",
            child_id=child.id,
            parent_id=parent.id,
            username=child.username
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

# @app.get("//kids/v1/profile")
# async def get_profile(child_id: int, db: Session = Depends(get_db)):
#     """Get child profile information for editing"""
#     child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
#     if not child:
#         raise HTTPException(status_code=404, detail="Child not found")

#     return {
#         "id": child.id,
#         "fullname": child.fullname,
#         "username": child.username,
#         "dob": child.dob.isoformat(),
#         "gender": child.gender,
#         "default_dream_career": child.default_dream_career,
#         "optional_dream_career_1": child.optional_dream_career_1,
#         "optional_dream_career_2": child.optional_dream_career_2,
#         "avatar": child.avatar,
#         "credits": {
#             "story": child.credits_story or 0,
#             "joke": child.credits_joke or 0,
#             "question": child.credits_question or 0,
#             "quiz": child.credits_quiz or 0,
#             "chat": child.credits_chat or 0,
#             "game": child.game_credits or 0,
#             "total": child.total_credits or 0
#         },
#         "last_story_generated": child.last_story_generated.isoformat() if child.last_story_generated else None,
#         "last_joke_generated": child.last_joke_generated.isoformat() if child.last_joke_generated else None,
#         "last_question_generated": child.last_question_generated.isoformat() if child.last_question_generated else None,
#         "last_quiz_generated": child.last_quiz_generated.isoformat() if child.last_quiz_generated else None,
#         "created_at": child.created_at.isoformat() if child.created_at else None
#     }

@app.get("//kids/v1/parents/{parent_id}", tags=["Parent"])
async def get_parent(parent_id: int, db: Session = Depends(get_db)):
    """Get parent information by ID"""
    parent = db.query(ParentDB).filter(ParentDB.id == parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")

    return {
        "id": parent.id,
        "name": parent.name,
        "gender": parent.gender,
        "email": parent.email,
        "relation": parent.relation,
        "created_at": parent.created_at.isoformat()
    }

class GenerateStoryRequest(BaseModel):
    child_id: int

import logging  # Add this import at the top if not already present

@app.post("//kids/v1/generate-story", tags=["Functionalities"])
async def generate_story_endpoint(request: GenerateStoryRequest, db: Session = Depends(get_db)):

    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    now = get_ist_now()

    # --- RETURN OLD STORY IF WITHIN 24 HOURS ---
    if child.last_story_generated and (now - child.last_story_generated) < timedelta(hours=24):
        return {
            "message": "Story retrieved successfully",
            "story": child.last_story_content,
            "generated_at": child.last_story_generated.isoformat()
        }

    # --- GENERATE NEW STORY ---
    story = generate_story(child)
    child.last_story_generated = now
    child.last_story_content = story

    # Reset daily credits if date changed
    reset_daily_credits_if_needed(child, db)

    # ---- CHECK LIMIT (daily_credits max = 15) ----
    if child.daily_credits < 15:
        # Award credits
        child.credits_story = (child.credits_story or 0) + 2
        child.total_credits = (child.total_credits or 0) + 2
        old_daily = child.daily_credits or 0
        child.daily_credits = min(15, old_daily + 2)

        db.add(CreditsHistory(
            child_id=child.id,
            activity="Story",
            credits_earned=2,
            credits_lost=0,
            total_credits=child.total_credits
        ))

    else:
        # Do NOT award credits (limit reached)
        db.add(CreditsHistory(
            child_id=child.id,
            activity="Story (daily limit reached)",
            credits_earned=0,
            credits_lost=0,
            total_credits=child.total_credits
        ))

    db.commit()

    return {
        "message": "Story retrieved successfully",
        "story": story,
        "generated_at": now.isoformat()
    }

class GenerateJokeRequest(BaseModel):
    child_id: int

@app.post("//kids/v1/generate-joke", tags=["Functionalities"])
async def generate_joke_endpoint(request: GenerateJokeRequest, db: Session = Depends(get_db)):

    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(404, "Child not found")

    now = get_ist_now()

    # Return old joke if within 24 hrs
    if child.last_joke_generated and (now - child.last_joke_generated) < timedelta(hours=24):
        return {
            "message": "Joke retrieved successfully",
            "joke": child.last_joke_content,
            "generated_at": child.last_joke_generated.isoformat()
        }

    # Generate new joke
    joke = generate_joke(child)
    child.last_joke_generated = now
    child.last_joke_content = joke

    reset_daily_credits_if_needed(child, db)

    # ---- CHECK DAILY CREDIT LIMIT ----
    if child.daily_credits < 15:
        child.credits_joke = (child.credits_joke or 0) + 2
        child.total_credits = (child.total_credits or 0) + 2
        old_daily = child.daily_credits or 0
        child.daily_credits = min(15, old_daily + 2)

        db.add(CreditsHistory(
            child_id=child.id,
            activity="Joke",
            credits_earned=2,
            credits_lost=0,
            total_credits=child.total_credits
        ))
    else:
        db.add(CreditsHistory(
            child_id=child.id,
            activity="Joke (daily limit reached)",
            credits_earned=0,
            credits_lost=0,
            total_credits=child.total_credits
        ))

    db.commit()

    return {
        "message": "Joke retrieved successfully",
        "joke": joke,
        "generated_at": now.isoformat()
    }

class GenerateQuestionRequest(BaseModel):
    child_id: int

class GenerateQuizRequest(BaseModel):
    child_id: int
    topic: str

from datetime import datetime, timedelta

@app.post("//kids/v1/generate-question",tags=["Functionalities"])
async def generate_question_endpoint(request: GenerateQuestionRequest, db: Session = Depends(get_db)):
    """Generate a new question for the child or return existing if within 24 hours """
    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    now = get_ist_now()

    # Convert stored datetime string (if any) to datetime object safely
    last_generated = child.last_question_generated
    if isinstance(last_generated, str):
        try:
            last_generated = datetime.fromisoformat(last_generated)
        except Exception:
            last_generated = None

    # If already answered within 24 hours → block
    if (
        last_generated
        and (now - last_generated) < timedelta(hours=24)
        and child.last_question_answered is True
    ):
        return {
            "message": f"For now, question limit exceeded, {child.username}!    ",
            "question": None,
            "generated_at": last_generated.isoformat() if last_generated else None
        }

    # Generate new question if no previous OR 24 hour passed
    if not last_generated or (now - last_generated) >= timedelta(hours=24):
        question_data = generate_question(child)
        child.last_question_generated = now
        child.last_question_content = json.dumps(question_data)
        child.last_question_answered = False
        db.commit()
        generated_at = now.isoformat()
    else:
        # Within 24 hours → return existing question
        question_data = json.loads(child.last_question_content)
        generated_at = last_generated.isoformat()

    return {
        "message": "Question retrieved successfully!",
        "question": question_data,
        "generated_at": generated_at
    }



@app.post("//kids/v1/generate-quiz", tags=["Functionalities"])
async def generate_quiz_endpoint(request: GenerateQuizRequest, db: Session = Depends(get_db)):
    """Generate a new quiz question (non-repeating) OR return the existing unanswered quiz."""

    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    # -------------------------------
    # 1️⃣  If an unanswered quiz exists → return it directly
    # -------------------------------
    existing_quiz = (
        db.query(QuizDB)
        .filter(QuizDB.child_id == request.child_id, QuizDB.answered == False)
        .order_by(QuizDB.date_created.desc())
        .first()
    )

    if existing_quiz:
        qd = json.loads(existing_quiz.question_data)
        return {
            "message": "Existing quiz question retrieved",
            "quiz": {
                "question": qd["question"],
                "options": qd["options"],
                "correct_answer": existing_quiz.correct_answer,
            },
            "generated_at": existing_quiz.date_created.isoformat(),
        }

    # -------------------------------
    # 2️⃣  No unanswered quiz → generate a new non-repeating one
    # -------------------------------
    now = get_ist_now()

    # All previously asked quiz questions
    old_questions = {
        json.loads(q.question_data)["question"].strip().lower()
        for q in db.query(QuizDB).filter(QuizDB.child_id == request.child_id).all()
    }

    new_question = None
    max_attempts = 10

    for _ in range(max_attempts):
        candidate = generate_quiz(child, request.topic)
        q_text = candidate["question"].strip().lower()

        if q_text not in old_questions:
            new_question = candidate
            break

    # If somehow repeated every time → accept last generated question but still save it
    if not new_question:
        new_question = candidate  

    # -------------------------------
    # 3️⃣ Save quiz to DB
    # -------------------------------
    saved = QuizDB(
        child_id=request.child_id,
        question_data=json.dumps(new_question),
        correct_answer=new_question["correct_answer"],
        shuffled_options=json.dumps(new_question["options"]),
        answered=False,
        date_created=now,
    )
    db.add(saved)
    db.commit()

    return {
        "message": "Quiz question generated successfully",
        "quiz": {
            "question": new_question["question"],
            "options": new_question["options"],
            "correct_answer": new_question["correct_answer"],
        },
        "generated_at": now.isoformat(),
    }

class SubmitQuestionAnswerRequest(BaseModel):
    child_id: int
    selected_answer: str  # A, B, C, or D


# NEW : **************************************07/11***************************************************************************
@app.post("//kids/v1/submit-question-answer", tags=["Functionalities"])
async def submit_question_answer(request: SubmitQuestionAnswerRequest, db: Session = Depends(get_db)):
    """Submit a question answer and update credits if correct on first attempt, apply daily credit limits."""
    
    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    # Prevent multiple attempts for the same day
    if child.last_question_answered:
        return {
            "message": f"You already answered today’s question {child.username}!",
            "is_correct": None,
            "explanation": None,
            "credits_awarded": 0,
            "total_credits": child.total_credits
        }

    # Ensure a question exists
    if not child.last_question_content:
        raise HTTPException(status_code=400, detail="No question generated yet")

    # Parse question info
    question_data = json.loads(child.last_question_content)
    correct_answer = question_data.get("correct_answer", "").upper()

    is_correct = request.selected_answer.upper() == correct_answer
    age = calculate_age(child.dob)
    age_slab = get_age_slab(age)

    explanation = get_explanation_for_answer(is_correct, age_slab, correct_answer)

    credits_awarded = 0

    # First-attempt logic
    q_generated_time = child.last_question_generated
    first_attempt = not child.last_question_credited_at or child.last_question_credited_at < q_generated_time

    # Reset daily credits if date changed
    reset_daily_credits_if_needed(child, db)

    if is_correct and first_attempt:
        if child.daily_credits < 15:
            # Award full credits
            child.daily_credits = min(15, (child.daily_credits or 0) + 2)
            child.credits_question = (child.credits_question or 0) + 2
            child.total_credits = (child.total_credits or 0) + 2
            credits_awarded = 2
            child.last_question_credited_at = q_generated_time
        else:
            # Daily limit reached → no credits
            explanation = "Daily credit limit reached! Try again tomorrow."
            credits_awarded = 0

    # Mark question as answered for today
    child.last_question_answered = True

    # Store history regardless of credit earned
    db.add(CreditsHistory(
        child_id=child.id,
        activity="question",
        credits_earned=credits_awarded,
        credits_lost=0,
        total_credits=child.total_credits or 0
    ))

    db.commit()

    return {
        "message": "Answer submitted successfully",
        "is_correct": is_correct,
        "explanation": explanation,
        "credits_awarded": credits_awarded,
        "total_credits": child.total_credits
    }



class SubmitQuizAnswerRequest(BaseModel):
    child_id: int
    selected_answer: str  # A, B, C, or D

class ChatRequest(BaseModel):
    child_id: int
    message: str

class ChatWithAudioRequest(BaseModel):
    child_id: int
    message: str

class UnblockChildRequest(BaseModel):
    parent_id: int
    child_id: int

class SetAvatarRequest(BaseModel):
    child_id: int
    selected_career: str

class SwitchCareerRequest(BaseModel):
    child_id: int
    selected_career: str



class GenerateChatAudioRequest(BaseModel):
    child_id: int
    message_id: Optional[int] = None  # Optional, defaults to latest message

class SpeechToSpeechRequest(BaseModel):
    child_id: int
    audio_base64: str

def get_explanation_for_answer(is_correct: bool, age_slab: str, correct_answer: str) -> str:
    """Generate explanation or reward message based on correctness and age slab"""
    if is_correct:
        return f"Congratulations! You answered correctly. You earned a reward for your great knowledge in the {age_slab} age group."
    else:
        # Provide explanation tailored to age slab
        explanations = {
            "0-3": f"The correct answer is {correct_answer}. Keep learning and you'll get better!",
            "4-6": f"The correct answer is {correct_answer}. Don't worry, keep practicing and you'll improve!",
            "7-9": f"The correct answer is {correct_answer}. Understanding this will help you a lot in your learning journey.",
            "10-12": f"The correct answer is {correct_answer}. Great effort! Keep studying to master this topic.",
            "13-15": f"The correct answer is {correct_answer}. Review this concept to strengthen your knowledge."
        }
        return explanations.get(age_slab, f"The correct answer is {correct_answer}. Keep learning!")

@app.post("//kids/v1/submit-quiz-answer", tags=["Functionalities"])
async def submit_quiz_answer(request: SubmitQuizAnswerRequest, db: Session = Depends(get_db)):
    """Submit a quiz answer with updated rules:
       - First 5 quizzes of the day → 0 or 1 credit
       - No deductions for wrong quiz answers
       - After 5 quizzes → no credits awarded
    """

    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(404, detail="Child not found")

    # Find unanswered quiz
    quiz = db.query(QuizDB).filter(
        QuizDB.child_id == request.child_id,
        QuizDB.answered == False
    ).first()
    if not quiz:
        raise HTTPException(400, "No quiz available to answer")

    correct_answer = quiz.correct_answer.upper()
    selected = request.selected_answer.upper()
    is_correct = (selected == correct_answer)

    # Age slab for explanation
    age = calculate_age(child.dob)
    age_slab = get_age_slab(age)

    explanation = ""
    credits_awarded = 0

    # Reset daily credits if a new day
    reset_daily_credits_if_needed(child, db)

    # -----------------------------------------------------------------
    # ⭐ NEW RULE:
    # Only FIRST 5 quiz answers of the day give credit (0 or 1)
    # No deduction at all for wrong answers.
    # -----------------------------------------------------------------

    if child.daily_quiz_credits < 5:

        # First 5 questions → award 1 only if correct
        if is_correct:
            credits_awarded = 1
            child.credits_quiz = (child.credits_quiz or 0) + 1
            child.total_credits = (child.total_credits or 0) + 1

            # Also increment daily_credits (combined activities)
            child.daily_credits = min(15, child.daily_credits + 1)

            explanation = "Great job! You earned 1 credit."
        else:
            # Wrong but no deduction — credit = 0
            credits_awarded = 0
            explanation = get_explanation_for_answer(False, age_slab, correct_answer)

        # Count toward 5-daily-quiz-limit
        child.daily_quiz_credits += 1

        # Log history
        db.add(CreditsHistory(
            child_id=child.id,
            activity="quiz",
            credits_earned=credits_awarded,
            credits_lost=0,
            total_credits=child.total_credits
        ))

    else:
        # After 5 questions → no credits, no deduction
        credits_awarded = 0

        explanation = "Good try! But only the first 5 quiz questions per day earn credits."

        db.add(CreditsHistory(
            child_id=child.id,
            activity="quiz",
            credits_earned=0,
            credits_lost=0,
            total_credits=child.total_credits
        ))

    # Mark quiz answered
    quiz.answered = True
    db.commit()

    return {
        "message": "Answer submitted successfully",
        "is_correct": is_correct,
        "explanation": explanation,
        "credits_awarded": credits_awarded,
        "total_credits": child.total_credits
    }

@app.get("//kids/v1/child-profile/{child_id}", tags=["Dashboard"])
async def get_child_profile(child_id: int, db: Session = Depends(get_db)):
    """Get child's profile including credits, chat earned/lost stats, and last generation timestamps"""
    
    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    # ---- Fetch chat credit stats ----
    chat_history = db.query(CreditsHistory).filter(
        CreditsHistory.child_id == child_id,
        CreditsHistory.activity == "chat"
    ).all()

    chat_credits_earned = sum(h.credits_earned or 0 for h in chat_history)
    chat_credits_lost = sum(h.credits_lost or 0 for h in chat_history)

    return {
        "id": child.id,
        "fullname": child.fullname,
        "username": child.username,
        "dob": child.dob.isoformat(),
        "gender": child.gender,
        "default_dream_career": child.default_dream_career,
        "optional_dream_career_1": child.optional_dream_career_1,
        "optional_dream_career_2": child.optional_dream_career_2,
        "avatar": child.avatar,

        # ---- Credits Summary ----
        "credits": {
            "story": child.credits_story or 0,
            "joke": child.credits_joke or 0,
            "question": child.credits_question or 0,
            "quiz": child.credits_quiz or 0,
            
            "chat_credits_earned": chat_credits_earned,
            "chat_credits_lost": chat_credits_lost,
            "total_credits": child.total_credits or 0,
            },
        

        # ---- Timestamps ----
        "last_story_generated": child.last_story_generated.isoformat() if child.last_story_generated else None,
        "last_joke_generated": child.last_joke_generated.isoformat() if child.last_joke_generated else None,
        "last_question_generated": child.last_question_generated.isoformat() if child.last_question_generated else None,
        "last_quiz_generated": child.last_quiz_generated.isoformat() if child.last_quiz_generated else None,
        "created_at": child.created_at.isoformat() if child.created_at else None
    }

@app.post("//kids/v1/chat", tags=["Chat Functionalities"])
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """Handle chat messages with moderation, credits and parental notifications."""
    MAX_WARNINGS = 5          # after 5th warning → block
    MAX_DAILY_CREDITS = 15    # global daily credit cap
    MAX_DAILY_CHAT_CREDITS = 4  # max 4 chat credits per day

    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    parent = db.query(ParentDB).filter(ParentDB.id == child.parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")

    if child.blocked:
        raise HTTPException(status_code=403, detail="Your account is blocked.")

    # Initialize
    warning_message = None
    credits_awarded = 0
    response_text = None

    # --- 1) Moderate incoming message (child's text) ---
    moderation_result = moderate_message(request.message)
    flagged = moderation_result.get("flagged", False)

    # ------------------ FLAGGED / ABUSIVE MESSAGE ------------------
    if flagged:
        # increase warning count
        child.warning_count = (child.warning_count or 0) + 1
        child.last_warning_at = get_ist_now()

        # Deduct 1 credit if child has any total credits
        if (child.total_credits or 0) > 0:
            child.total_credits = (child.total_credits or 0) - 1

            db.add(
                CreditsHistory(
                    child_id=child.id,
                    activity="chat",
                    credits_earned=0,
                    credits_lost=1,
                    total_credits=child.total_credits or 0,
                )
            )

        # Check if we should block now
        if child.warning_count >= MAX_WARNINGS:
            child.blocked = True

            # Grab last 5 abusive/flagged messages
            abusive_messages_query = (
                db.query(ChatMessage)
                .filter(ChatMessage.child_id == child.id, ChatMessage.flagged == 1)
                .order_by(ChatMessage.timestamp.desc())
                .limit(5)
                .all()
            )
            abusive_messages = [msg.message for msg in abusive_messages_query]

            # Notify parent with last 5 bad messages
            send_block_notification_email(
                parent.email,
                child.fullname,
                abusive_messages,
            )

            db.commit()
            raise HTTPException(
                status_code=403,
                detail="Your account has been blocked due to multiple inappropriate messages. Please contact your parent.",
            )

        # Still not blocked → send warning message
        remaining = MAX_WARNINGS - child.warning_count
        warning_message = (
            f"Please be polite and respectful in your messages. "
            f"You have {remaining} warnings left before your account gets blocked. "
            f"1 credit has been deducted for inappropriate language."
        )
        response_text = warning_message

        db.commit()

    # ------------------ NON-FLAGGED / NORMAL MESSAGE ------------------
    else:
        # Generate chatbot response
        response_text = generate_chat_response(child, request.message, db)

        # Optionally moderate bot response (kept from your original)
        moderation_result = moderate_message(response_text)
        bot_flagged = moderation_result.get("flagged", False)  # not used further but okay

        original_response = response_text

        # Evaluate quality of the conversation
        is_good, feedback = evaluate_chat_quality(
            child, request.message, original_response
        )

        if is_good:
            # Ensure daily credits are fresh
            reset_daily_credits_if_needed(child, db)

            # Normalize None → 0
            daily_credits = child.daily_credits or 0
            daily_chat_credits = child.daily_chat_credits or 0

            # Award credit only if:
            # - Global daily credits < 15
            # - Chat-specific daily credits < 4
            if daily_credits < MAX_DAILY_CREDITS and daily_chat_credits < MAX_DAILY_CHAT_CREDITS:
                # Award 1 chat credit
                child.daily_chat_credits = daily_chat_credits + 1
                child.credits_chat = (child.credits_chat or 0) + 1
                child.total_credits = (child.total_credits or 0) + 1

                # Also increment global daily_credits, capped at 15
                child.daily_credits = min(MAX_DAILY_CREDITS, daily_credits + 1)

                credits_awarded = 1

                # Track "good chats" for email notification
                child.daily_good_chats = (child.daily_good_chats or 0) + 1

                # If child has 5+ good chats in a day and no email sent today → send email once
                now = get_ist_now()
                if (
                    child.daily_good_chats >= 5
                    and (
                        child.last_good_chat_email_sent is None
                        or child.last_good_chat_email_sent.date() != now.date()
                    )
                ):
                    send_good_chat_email(
                        parent.email,
                        child.fullname,
                        child.daily_good_chats,
                    )
                    child.last_good_chat_email_sent = now

                # Log credit in history
                db.add(
                    CreditsHistory(
                        child_id=child.id,
                        activity="chat",
                        credits_earned=1,
                        credits_lost=0,
                        total_credits=child.total_credits or 0,
                    )
                )
            else:
                # Good chat but credit cap reached → no credit, optionally still log
                db.add(
                    CreditsHistory(
                        child_id=child.id,
                        activity="chat",
                        credits_earned=0,
                        credits_lost=0,
                        total_credits=child.total_credits or 0,
                    )
                )

        # Commit changes for non-flagged path
        db.commit()

    # ------------------ SAVE CHAT MESSAGE (always) ------------------
    chat_message = ChatMessage(
        child_id=child.id,
        message=request.message,
        response=response_text,
        flagged=1 if flagged else 0,
    )
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)

    return {
        "message": "Chat response generated successfully",
        "response": response_text,
        "message_id": chat_message.id,
        "flagged": flagged,
        "warning": warning_message,
        "credits_awarded": credits_awarded,
        "total_credits": child.total_credits,
    }


@app.post("//kids/v1/chat-with-audio", tags=["Chat Functionalities"])
async def chat_with_audio_endpoint(request: ChatWithAudioRequest, db: Session = Depends(get_db)):
    """Chat with moderation, credits, blocking, audio generation, and parent notifications"""

    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(404, "Child not found")

    parent = db.query(ParentDB).filter(ParentDB.id == child.parent_id).first()
    if not parent:
        raise HTTPException(404, "Parent not found")

    if child.blocked:
        raise HTTPException(403, "Your account is blocked due to inappropriate messages.")

    # ------------------- MODERATION -----------------------
    moderation_result = moderate_message(request.message)
    flagged = moderation_result.get("flagged", False)

    warning_message = None
    credits_awarded = 0

    # --------- IF FLAGGED MESSAGE (BAD LANGUAGE) ----------
    if flagged:
        child.warning_count += 1
        child.last_warning_at = get_ist_now()

        # Deduct 1 credit (if possible)
        reset_daily_credits_if_needed(child, db)
        if (child.total_credits or 0) > 0:
            child.total_credits -= 1
            db.add(CreditsHistory(
                child_id=child.id,
                activity="chat",
                credits_lost=1,
                total_credits=child.total_credits
            ))

        # Blocking logic (5 warnings max)
        if child.warning_count >= 5:
            child.blocked = True

            abusive_messages_query = db.query(ChatMessage).filter(
                ChatMessage.child_id == child.id,
                ChatMessage.flagged == 1
            ).order_by(ChatMessage.timestamp.desc()).limit(5).all()

            abusive_messages = [msg.message for msg in abusive_messages_query]
            send_block_notification_email(parent.email, child.fullname, abusive_messages)

            db.commit()
            raise HTTPException(403, "Your account has been blocked due to inappropriate messages.")

        warning_message = (
            f"Please be polite. You have {5 - child.warning_count} warnings left. "
            f"1 credit deducted for inappropriate language."
        )
        response_text = warning_message
        db.commit()

    else:
        # -------------- NORMAL (NON-FLAGGED) MESSAGE --------------
        response_text = generate_chat_response(child, request.message, db)

        # Check good language & behaviour
        is_good, feedback = evaluate_chat_quality(child, request.message, response_text)

        if is_good:
            reset_daily_credits_if_needed(child, db)

            # Max 4 credits/day from chat
            if (child.daily_chat_credits or 0) < 4:

                child.daily_chat_credits = (child.daily_chat_credits or 0) + 1
                child.credits_chat = (child.credits_chat or 0) + 1
                child.total_credits = (child.total_credits or 0) + 1
                credits_awarded = 1

                # Log credit award
                db.add(CreditsHistory(
                    child_id=child.id,
                    activity="chat",
                    credits_earned=1,
                    total_credits=child.total_credits
                ))

                # Send good behavior email only once per day
                child.daily_good_chats = (child.daily_good_chats or 0) + 1
                if (child.daily_good_chats >= 5 and 
                    (child.last_good_chat_email_sent is None or 
                     child.last_good_chat_email_sent.date() != get_ist_now().date())):

                    send_good_chat_email(parent.email, child.fullname, child.daily_good_chats)
                    child.last_good_chat_email_sent = get_ist_now()

        db.commit()

    # ------------------ SAVE MESSAGE ---------------------
    chat_message = ChatMessage(
        child_id=child.id,
        message=request.message,
        response=response_text,
        flagged=1 if flagged else 0
    )
    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)

    # ------------------ AUDIO GENERATION -----------------
    global openai_client
    if openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise HTTPException(500, "OpenAI API key missing")
        openai_client = OpenAI(api_key=api_key)

    try:
        audio_response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=response_text
        )
        audio_bytes = audio_response.content
        if not audio_bytes:
            raise Exception("Invalid TTS response")

    except Exception as e:
        raise HTTPException(500, f"TTS generation failed: {str(e)}")

    os.makedirs("audio_clips", exist_ok=True)
    audio_filename = f"chat_audio_{chat_message.id}.mp3"
    audio_path = os.path.join("audio_clips", audio_filename)

    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    chat_message.audio_path = audio_path
    db.commit()

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "message": "Chat response generated successfully",
        "response": response_text,
        "audio_base64": audio_base64,
        "flagged": flagged,
        "warning": warning_message,
        "credits_awarded": credits_awarded,
        "total_credits": child.total_credits
    }


@app.get("//kids/v1/generate-chat-audio",tags=["Chat Functionalities"])
async def generate_chat_audio(child_id: int, message_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Generate and store audio clip for the most recent bot chat response"""
    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    # Get the message to generate audio for
    if message_id:
        chat_message = db.query(ChatMessage).filter(ChatMessage.id == message_id, ChatMessage.child_id == child_id).first()
        if not chat_message:
            raise HTTPException(status_code=404, detail="Chat message not found")
        text_to_speak = chat_message.response
        message_id = chat_message.id
    else:
        # Get the latest bot response
        chat_message = db.query(ChatMessage).filter(ChatMessage.child_id == child_id).order_by(ChatMessage.timestamp.desc()).first()
        if not chat_message or not chat_message.response:
            raise HTTPException(status_code=404, detail="No recent chat response found")
        text_to_speak = chat_message.response
        message_id = chat_message.id

    if not text_to_speak:
        raise HTTPException(status_code=400, detail="No response text available for audio generation")

    # Initialize OpenAI client if needed
    global openai_client
    if openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        openai_client = OpenAI(api_key=api_key)

    # Generate audio using OpenAI TTS
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text_to_speak
        )
        # Get audio bytes
        audio_bytes = response.content
        if audio_bytes is None or not isinstance(audio_bytes, bytes):
            raise HTTPException(status_code=500, detail="Failed to generate audio: Invalid response from TTS service")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")

    # Ensure audio_clips directory exists
    os.makedirs("audio_clips", exist_ok=True)

    # Save audio file
    audio_filename = f"chat_audio_{message_id}.mp3"
    audio_path = os.path.join("audio_clips", audio_filename)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # Update database with audio path
    chat_message.audio_path = audio_path
    db.commit()

    # Return audio directly for inline playback
    return Response(content=audio_bytes, media_type="audio/mpeg", headers={"Content-Disposition": "inline; filename=audio.mp3"})

class STTRequest(BaseModel):
    audio_base64: str

@app.post("//kids/v1/speech-to-text",tags=["Chat Functionalities"]  )
async def speech_to_text(request: STTRequest):
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception as e:
        print("BASE64 DECODE ERROR:", e)
        raise HTTPException(400, "Invalid base64 audio data")

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.webm"

    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(500, "OpenAI API key missing")
        openai_client = OpenAI(api_key=api_key)

    try:
        print("Sending audio to Whisper... size:", len(audio_bytes))

        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

        print("Whisper response:", transcription)
        text = transcription.strip()

        if not text:
            raise HTTPException(400, "Whisper returned empty text")

    except Exception as e:
        print("\n🔥 WHISPER ERROR RAW MESSAGE 🔥")
        print(str(e))
        print("-------------------------------------------------\n")
        raise HTTPException(500, f"Speech to text failed: {str(e)}")

    return {"transcribed_text": text}




@app.post("//kids/v1/speech-to-speech", tags=["Chat Functionalities"])
async def speech_to_speech_endpoint(request: SpeechToSpeechRequest, db: Session = Depends(get_db)):
    """Speech-to-speech with transcription, moderation, good/bad chat credits, blocking, email alerts, and audio TTS"""

    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(404, "Child not found")

    parent = db.query(ParentDB).filter(ParentDB.id == child.parent_id).first()
    if not parent:
        raise HTTPException(404, "Parent not found")

    if child.blocked:
        raise HTTPException(403, "Your account is blocked due to inappropriate messages.")

    # ------------------ SPEECH INPUT --------------------
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except:
        raise HTTPException(400, "Invalid BASE64 audio")

    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(500, "OpenAI API key missing")
        openai_client = OpenAI(api_key=api_key)

    # ------------------ TRANSCRIBE ----------------------
    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        transcribed_text = transcription.strip()

        if not transcribed_text:
            raise HTTPException(400, "Could not transcribe audio")

    except Exception as e:
        raise HTTPException(500, f"Whisper transcription failed: {str(e)}")

    # ------------------ MODERATE ------------------------
    moderation_result = moderate_message(transcribed_text)
    flagged = moderation_result.get("flagged", False)

    warning_message = None
    credits_awarded = 0

    # ============================================
    # 🔥 FLAGGED / ABUSIVE MESSAGE
    # ============================================
    if flagged:

        child.warning_count += 1
        child.last_warning_at = get_ist_now()

        # Deduct 1 credit for abusive language
        reset_daily_credits_if_needed(child, db)
        if (child.total_credits or 0) > 0:
            child.total_credits -= 1

            db.add(CreditsHistory(
                child_id=child.id,
                activity="chat",
                credits_lost=1,
                total_credits=child.total_credits
            ))

        # Block after 5 warnings
        if child.warning_count >= 5:
            child.blocked = True

            # Fetch last 5 abusive msgs
            abusive_messages_query = db.query(ChatMessage).filter(
                ChatMessage.child_id == child.id,
                ChatMessage.flagged == 1
            ).order_by(ChatMessage.timestamp.desc()).limit(5).all()

            abusive_messages = [msg.message for msg in abusive_messages_query]
            send_block_notification_email(parent.email, child.fullname, abusive_messages)

            db.commit()
            raise HTTPException(403, "Your account has been blocked due to inappropriate messages.")

        warning_message = (
            f"Please be polite. You have {5 - child.warning_count} warnings left. "
            f"1 credit deducted for inappropriate language."
        )

        response_text = warning_message
        db.commit()

    else:

        # ================================
        # 🔥 GOOD / NORMAL CHAT
        # ================================
        response_text = generate_chat_response(child, transcribed_text, db)

        is_good, feedback = evaluate_chat_quality(child, transcribed_text, response_text)

        if is_good:
            reset_daily_credits_if_needed(child, db)

            if (child.daily_chat_credits or 0) < 4:   # max 4 per day

                child.daily_chat_credits = (child.daily_chat_credits or 0) + 1
                child.credits_chat = (child.credits_chat or 0) + 1
                child.total_credits = (child.total_credits or 0) + 1
                credits_awarded = 1

                db.add(CreditsHistory(
                    child_id=child.id,
                    activity="chat",
                    credits_earned=1,
                    total_credits=child.total_credits
                ))

                # good chat streak email
                child.daily_good_chats = (child.daily_good_chats or 0) + 1
                if (child.daily_good_chats >= 5 and
                    (child.last_good_chat_email_sent is None or 
                     child.last_good_chat_email_sent.date() != get_ist_now().date())):

                    send_good_chat_email(parent.email, child.fullname, child.daily_good_chats)
                    child.last_good_chat_email_sent = get_ist_now()

        db.commit()

    # ------------------ SAVE CHAT MSG --------------------
    chat_message = ChatMessage(
        child_id=child.id,
        message=transcribed_text,
        response=response_text,
        flagged=1 if flagged else 0
    )

    db.add(chat_message)
    db.commit()
    db.refresh(chat_message)

    # ------------------ GENERATE AUDIO --------------------
    try:
        tts_output = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=response_text
        )
        audio_bytes_response = tts_output.content

        if not audio_bytes_response:
            raise Exception("Invalid TTS output")

    except Exception as e:
        raise HTTPException(500, f"TTS generation failed: {str(e)}")

    os.makedirs("audio_clips", exist_ok=True)

    audio_filename = f"speech_audio_{chat_message.id}.mp3"
    audio_path = os.path.join("audio_clips", audio_filename)

    with open(audio_path, "wb") as f:
        f.write(audio_bytes_response)

    chat_message.audio_path = audio_path
    db.commit()

    audio_base64 = base64.b64encode(audio_bytes_response).decode("utf-8")

    return {
        "message": "Speech-to-speech response generated successfully",
        "transcribed_text": transcribed_text,
        "response": response_text,
        "audio_base64": audio_base64,
        "flagged": flagged,
        "warning": warning_message,
        "credits_awarded": credits_awarded,
        "total_credits": child.total_credits
    }

# NEW ENDPOINT UNBLOCK*********************************************************************************************
@app.post("//kids/v1/unblock-child", tags=["Block & Unblock"])
async def unblock_child(request: UnblockChildRequest, db: Session = Depends(get_db)):
    """Allow parents to unblock their child"""
    parent = db.query(ParentDB).filter(ParentDB.id == request.parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")

    child = db.query(ChildDB).filter(ChildDB.id == request.child_id, ChildDB.parent_id == request.parent_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found or not associated with this parent")


    #New Logic: If already unblocked*******************************
    if child.blocked is False:
        raise HTTPException(
            status_code=409, 
            detail=f"Child '{child.username}' is already unblocked. No action needed."
        )

    child.blocked = False
    child.warning_count = 0  # reset warnings
    db.commit()

    return {"message": "Child unblocked successfully"}

#NEW ENDPOINT BLOCK************************07/11/25*******************************************************************
@app.post("//kids/v1/block-child", tags=["Block & Unblock"])
async def block_child(request: UnblockChildRequest, db: Session = Depends(get_db)):
    """Allow parents to block their child"""
    parent = db.query(ParentDB).filter(ParentDB.id == request.parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")

    child = db.query(ChildDB).filter(
        ChildDB.id == request.child_id,
        ChildDB.parent_id == request.parent_id
    ).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found or not associated with this parent")

    #New Logic: If already blocked******************************************************************
    if child.blocked is True:
        raise HTTPException(
            status_code=409,  # Conflict: already in desired state
            detail=f"Child '{child.username}' is already blocked. No action needed."
        )

    child.blocked = True
    db.commit()

    return {"message": "Child blocked successfully"}
 

import os
from pathlib import Path

@app.post("//kids/v1/set-avatar",tags=["Avatar Management"])
async def set_avatar(request: SetAvatarRequest, db: Session = Depends(get_db)):
    """Set or update avatar for the child based on selected career, validating it is in dream careers, and rearrange careers to make selected the default. Fetches existing avatar from folder."""
    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    dream_career = child.dream_career or ""
    careers = [c.strip() for c in dream_career.split(',') if c.strip()]
    careers_lower = [c.lower() for c in careers]
    selected_career_lower = request.selected_career.strip().lower()

    if selected_career_lower not in careers_lower:
        raise HTTPException(status_code=400, detail="Selected career not in child's dream careers")

    # Find the index of the selected career
    index = careers_lower.index(selected_career_lower)
    selected_career = careers[index]

    # Rearrange: make selected the default, shift others to optionals
    new_careers = [selected_career] + [c for i, c in enumerate(careers) if i != index]

    # Update the database fields
    child.default_dream_career = new_careers[0]
    child.optional_dream_career_1 = new_careers[1] if len(new_careers) > 1 else None
    child.optional_dream_career_2 = new_careers[2] if len(new_careers) > 2 else None

    # Construct avatar filename to match existing files (e.g., "Doctor" -> "Doctor_avatar.png")
    avatar_filename = f"{selected_career.replace(' ', '_')}_avatar.png"  # Preserve casing, add "_avatar" suffix
    child_folder = f"avatars/{child.id}"
    avatar_filepath = Path(child_folder) / avatar_filename
    avatar = f"{child_folder}/{avatar_filename}"  # Local path for DB/avatar field

    # Check if avatar file exists
    if not avatar_filepath.exists():
        # Optional debug: Log the expected path for troubleshooting
        print(f"Avatar file not found at: {avatar_filepath}. Expected filename: {avatar_filename}")
        raise HTTPException(status_code=404, detail="Avatar not found for selected career. Please ensure avatars are pre-generated.")

    # Update DB with avatar path (no generation needed)
    child.avatar = avatar
    db.commit()

    return {"message": "Avatar set successfully", "avatar": avatar, "avatar_url": f"//kids/v1/avatar/{child.id}", "avatar_folder": child_folder}



@app.post("//kids/v1/switch-career",tags=["Avatar Management"])
async def switch_career(request: SwitchCareerRequest, db: Session = Depends(get_db)):
    """Switch the child's career, rearranging dream careers to make selected the default, and fetch existing avatar"""
    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    dream_career = child.dream_career or ""
    careers = [c.strip() for c in dream_career.split(',') if c.strip()]
    careers_lower = [c.lower() for c in careers]
    selected_career_lower = request.selected_career.strip().lower()

    if selected_career_lower not in careers_lower:
        raise HTTPException(status_code=400, detail="Selected career not in child's dream careers")

    # Find the index
    index = careers_lower.index(selected_career_lower)
    selected_career = careers[index]

    # Rearrange
    new_careers = [selected_career] + [c for i, c in enumerate(careers) if i != index]

    # Update
    child.default_dream_career = new_careers[0]
    child.optional_dream_career_1 = new_careers[1] if len(new_careers) > 1 else None
    child.optional_dream_career_2 = new_careers[2] if len(new_careers) > 2 else None

    # Search for the avatar file in the child's folder
    child_folder = f"avatars/{child.id}"
    folder_path = Path(child_folder)
    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Avatar folder not found for this child.")

    # Normalize selected career for matching (e.g., "Doctor" -> "doctor")
    normalized_career = selected_career.lower().replace(' ', '_')
    avatar_filename = None
    for file in folder_path.iterdir():
        if file.is_file() and normalized_career in file.name.lower():
            avatar_filename = file.name
            break

    if not avatar_filename:
        raise HTTPException(status_code=404, detail="Avatar not found for selected career. Please ensure avatars are pre-generated.")

    avatar = f"{child_folder}/{avatar_filename}"  # Full path for DB/avatar field

    # Update DB with avatar path (no generation needed)
    child.avatar = avatar
    db.commit()

    return {"message": "Career switched successfully", "avatar": avatar, "avatar_url": f"//kids/v1/avatar/{child.id}", "avatar_folder": child_folder}


@app.post("//kids/v1/edit-profile",tags=["Profile Management"])
async def edit_profile(request: EditProfileRequest, db: Session = Depends(get_db)):
    """Edit the child's profile, allowing updates to various fields and career switching"""
    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    updates = {}

    if request.fullname is not None:
        updates['fullname'] = request.fullname

    if request.username is not None:
        existing = db.query(ChildDB).filter(ChildDB.username == request.username, ChildDB.id != child.id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")
        updates['username'] = request.username

    if request.dob is not None:
        try:
            dob = datetime.strptime(request.dob, '%Y-%m-%d').date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        age = calculate_age(dob)
        if age < 5 or age > 15:
            raise HTTPException(status_code=400, detail="Child age must be between 5 and 15 years")
        updates['dob'] = dob

    if request.gender is not None:
        updates['gender'] = request.gender

    if request.default_dream_career is not None:
        updates['default_dream_career'] = request.default_dream_career

    if request.optional_dream_career_1 is not None:
        updates['optional_dream_career_1'] = request.optional_dream_career_1

    if request.optional_dream_career_2 is not None:
        updates['optional_dream_career_2'] = request.optional_dream_career_2

    # Update avatar if default_dream_career is being updated
    if 'default_dream_career' in updates:
        new_career = updates['default_dream_career']
        avatar = get_avatar_for_career(new_career, child.gender, child.id)
        updates['avatar'] = avatar

    if request.switch_to_career is not None:
        careers = [child.default_dream_career]
        if child.optional_dream_career_1:
            careers.append(child.optional_dream_career_1)
        if child.optional_dream_career_2:
            careers.append(child.optional_dream_career_2)
        if request.switch_to_career.lower() not in [c.lower() for c in careers]:
            raise HTTPException(status_code=400, detail="Switch to career not in child's dream careers")
        index = [c.lower() for c in careers].index(request.switch_to_career.lower())
        selected_career = careers[index]
        new_careers = [selected_career] + [c for i, c in enumerate(careers) if i != index]
        updates['default_dream_career'] = new_careers[0]
        updates['optional_dream_career_1'] = new_careers[1] if len(new_careers) > 1 else None
        updates['optional_dream_career_2'] = new_careers[2] if len(new_careers) > 2 else None
        avatar = get_avatar_for_career(selected_career, child.gender, child.id)
        updates['avatar'] = avatar

    # Apply updates
    for key, value in updates.items():
        setattr(child, key, value)
    db.commit()

    # Generate and download avatar if it was updated (i.e., career changed)
    if 'avatar' in updates:
        child_folder = f"avatars/{child.id}"
        os.makedirs(child_folder, exist_ok=True)  # Ensure folder exists
        avatar_url = updates['avatar']  # The URL from get_avatar_for_career
        filename = avatar_url.split('/')[-1]  # Extract filename from URL
        generate_and_download_avatar(child.default_dream_career, filename, child.gender, child_folder)
        
        # Update child.avatar to the local path for consistency with other endpoints
        local_avatar_path = f"{child_folder}/{filename}"
        child.avatar = local_avatar_path
        db.commit()  # Commit the local path update

    # Return updated profile
    return {
        "id": child.id,
        "fullname": child.fullname,
        "username": child.username,
        "dob": child.dob.isoformat(),
        "gender": child.gender,
        "default_dream_career": child.default_dream_career,
        "optional_dream_career_1": child.optional_dream_career_1,
        "optional_dream_career_2": child.optional_dream_career_2,
        "avatar": child.avatar,  # Now the local path
        "credits": {
            "story": child.credits_story or 0,
            "joke": child.credits_joke or 0,
            "question": child.credits_question or 0,
            "quiz": child.credits_quiz or 0,
            "chat": child.credits_chat or 0,
            "game": child.game_credits or 0,
            "total": child.total_credits or 0
        },
        "last_story_generated": child.last_story_generated.isoformat() if child.last_story_generated else None,
        "last_joke_generated": child.last_joke_generated.isoformat() if child.last_joke_generated else None,
        "last_question_generated": child.last_question_generated.isoformat() if child.last_question_generated else None,
        "last_quiz_generated": child.last_quiz_generated.isoformat() if child.last_quiz_generated else None,
        "created_at": child.created_at.isoformat() if child.created_at else None
    }

import os
from fastapi.responses import FileResponse

@app.get("//kids/v1/avatar/{child_id}", tags=["Avatar Management"])
async def get_avatar(child_id: int, db: Session = Depends(get_db)):
    """Serve the avatar image for the child"""
    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    if not child.avatar:
        raise HTTPException(status_code=404, detail="Avatar not set for this child")

    # Use the full path directly (no need to join with "avatars" again)
    avatar_path = child.avatar
    if not os.path.exists(avatar_path):
        raise HTTPException(status_code=404, detail="Avatar file not found")

    # Extract just the filename for the response (e.g., "Doctor_avatar.png")
    filename = os.path.basename(avatar_path)
    return FileResponse(avatar_path, media_type='image/png', filename=filename)


@app.get("//kids/v1/chat-history/{parent_id}/{child_id}",tags=["Chat Functionalities"])
async def get_chat_history(parent_id: int, child_id: int, date: Optional[str] = None, db: Session = Depends(get_db)):
    """Get chat history for a child, accessible only by the associated parent, grouped by date"""
    parent = db.query(ParentDB).filter(ParentDB.id == parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")

    child = db.query(ChildDB).filter(ChildDB.id == child_id, ChildDB.parent_id == parent_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found or not associated with this parent")

    # Base query
    query = db.query(ChatMessage).filter(ChatMessage.child_id == child_id)

    # Apply date filter if provided
    if date:
        try:
            filter_date = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Fetch all chat messages, ordered by timestamp ascending (for chronological order within day)
    messages = query.order_by(ChatMessage.timestamp.asc()).all()

    # Group messages by date
    chat_history = {}
    for msg in messages:
        msg_date = msg.timestamp.date().isoformat()  # YYYY-MM-DD
        if msg_date not in chat_history:
            chat_history[msg_date] = []
        chat_history[msg_date].append({
            "id": msg.id,
            "message": msg.message,
            "response": msg.response,
            "timestamp": msg.timestamp.isoformat(),
            "flagged": msg.flagged
        })

    # If no date filter, sort dates descending (most recent first)
    if not date:
        chat_history = dict(sorted(chat_history.items(), key=lambda x: x[0], reverse=True))
    return {
        "message": "Chat history retrieved successfully",
        "child_id": child_id,
        "child_name": child.fullname,
        "chat_history": chat_history
    }

@app.get("//kids/v1/credit-history/{parent_id}/{child_id}",tags=["Credit Management"])
async def get_credit_history(parent_id: int, child_id: int, db: Session = Depends(get_db)):
    """Get credit history for a child, accessible only by the associated parent"""
    parent = db.query(ParentDB).filter(ParentDB.id == parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")

    child = db.query(ChildDB).filter(ChildDB.id == child_id, ChildDB.parent_id == parent_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found or not associated with this parent")

    # Fetch credit history for the child, ordered by timestamp descending (most recent first)
    credit_history = db.query(CreditsHistory).filter(CreditsHistory.child_id == child_id).order_by(CreditsHistory.timestamp.desc()).all()

    history_data = [{
        "id": entry.id,
        "activity": entry.activity,
        "credits_earned": entry.credits_earned,
        "credits_lost": entry.credits_lost,
        "timestamp": entry.timestamp.isoformat(),
        "total_credits": entry.total_credits
    } for entry in credit_history]

    return {
        "message": "Credit history retrieved successfully",
        "child_id": child_id,
        "child_name": child.fullname,
        "credit_history": history_data
    }

@app.post("//kids/v1/set-reminder",tags=["Reminders"])
async def set_reminder(request: SetReminderRequest, db: Session = Depends(get_db)):
    """Set a reminder for the child"""
    child = db.query(ChildDB).filter(ChildDB.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    # Parse reminder_time string to time object
    try:
        reminder_time_obj = datetime.strptime(request.reminder_time, '%H:%M').time()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM")

    reminder = ReminderDB(
        child_id=request.child_id,
        occasion=request.occasion,
        reminder_date=request.reminder_date,
        reminder_time=reminder_time_obj,
        message=request.message
    )
    db.add(reminder)
    db.commit()
    db.refresh(reminder)

    return {"message": "Reminder set successfully", "reminder_id": reminder.id}

@app.get("//kids/v1/reminders/{child_id}",tags=["Reminders"])
async def get_reminders(child_id: int, db: Session = Depends(get_db)):
    """Get all reminders for the child"""
    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    reminders = db.query(ReminderDB).filter(ReminderDB.child_id == child_id).order_by(ReminderDB.reminder_date, ReminderDB.reminder_time).all()
    reminders_data = [{
        "id": r.id,
        "occasion": r.occasion,
        "reminder_date": r.reminder_date.isoformat(),
        "reminder_time": r.reminder_time.strftime('%H:%M'),
        "message": r.message,
        "created_at": r.created_at.isoformat()
    } for r in reminders]

    return {"reminders": reminders_data}

@app.put("//kids/v1/reminder/{reminder_id}",tags=["Reminders"])
async def update_reminder(reminder_id: int, request: SetReminderRequest, db: Session = Depends(get_db)):
    """Update a reminder for the child"""
    reminder = db.query(ReminderDB).filter(ReminderDB.id == reminder_id).first()
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")

    if reminder.child_id != request.child_id:
        raise HTTPException(status_code=403, detail="Reminder does not belong to this child")

    # Parse reminder_time string to time object
    try:
        reminder_time_obj = datetime.strptime(request.reminder_time, '%H:%M').time()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time format. Use HH:MM")

    reminder.occasion = request.occasion
    reminder.reminder_date = request.reminder_date
    reminder.reminder_time = reminder_time_obj
    reminder.message = request.message
    db.commit()

    return {"message": "Reminder updated successfully"}

@app.delete("//kids/v1/reminder/{reminder_id}",tags=["Reminders"])
async def delete_reminder(reminder_id: int, child_id: int, db: Session = Depends(get_db)):
    """Delete a reminder for the child"""
    reminder = db.query(ReminderDB).filter(ReminderDB.id == reminder_id).first()
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")

    if reminder.child_id != child_id:
        raise HTTPException(status_code=403, detail="Reminder does not belong to this child")

    db.delete(reminder)
    db.commit()

    return {"message": "Reminder deleted successfully"}

@app.put("//kids/v1/reminder/{reminder_id}/notify",tags=["Reminders"])
async def notify_reminder(reminder_id: int, db: Session = Depends(get_db)):
    """Mark a reminder as notified"""
    reminder = db.query(ReminderDB).filter(ReminderDB.id == reminder_id).first()
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")

    reminder.notified = True
    db.commit()

    return {"message": "Reminder marked as notified"}
@app.get("//kids/v1/parent-dashboard/{parent_id}", tags=["Dashboard"])
async def get_parent_dashboard(parent_id: int, db: Session = Depends(get_db)):
    """Get parent dashboard with aggregated child information, credits, chat history, and flagged messages"""

    parent = db.query(ParentDB).filter(ParentDB.id == parent_id).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Parent not found")

    children = db.query(ChildDB).filter(ChildDB.parent_id == parent_id).all()
    children_data = []

    for child in children:

        # ---------------- PROFILE DATA ----------------
        profile = {
            "id": child.id,
            "fullname": child.fullname,
            "username": child.username,
            "dob": child.dob.isoformat(),
            "gender": child.gender,
            "default_dream_career": child.default_dream_career,
            "optional_dream_career_1": child.optional_dream_career_1,
            "optional_dream_career_2": child.optional_dream_career_2,
            "avatar": child.avatar,

            # 🔥 NEW FIELD ADDED HERE
            "is_blocked": True if child.blocked else False,

            "credits": {
                "story": child.credits_story or 0,
                "joke": child.credits_joke or 0,
                "question": child.credits_question or 0,
                "quiz": child.credits_quiz or 0,
                "chat": child.credits_chat or 0,
                "total": child.total_credits or 0,
            },
            "last_login": child.last_login.isoformat() if child.last_login else None,
            "created_at": child.created_at.isoformat() if child.created_at else None
        }

        # ---------------- CHAT HISTORY ----------------
        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.child_id == child.id)
            .order_by(ChatMessage.timestamp.asc())
            .all()
        )

        chat_history = {}
        for msg in messages:
            msg_date = msg.timestamp.date().isoformat()
            if msg_date not in chat_history:
                chat_history[msg_date] = []
            chat_history[msg_date].append({
                "id": msg.id,
                "message": msg.message,
                "response": msg.response,
                "timestamp": msg.timestamp.isoformat(),
                "flagged": msg.flagged
            })

        # ---------------- FLAGGED MESSAGES ---------------
        flagged_messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.child_id == child.id, ChatMessage.flagged == 1)
            .order_by(ChatMessage.timestamp.desc())
            .all()
        )

        flagged_data = [{
            "id": msg.id,
            "message": msg.message,
            "response": msg.response,
            "timestamp": msg.timestamp.isoformat(),
            "flagged": msg.flagged,
        } for msg in flagged_messages]

        # Append final child block
        children_data.append({
            "profile": profile,
            "chat_history": chat_history,
            "flagged_messages": flagged_data,
        })

    return {
        "parent": {
            "id": parent.id,
            "name": parent.name,
            "gender": parent.gender,
            "email": parent.email,
            "relation": parent.relation,
            "created_at": parent.created_at.isoformat(),
        },
        "children": children_data,
    }



# ************************************************* GAMES : ************************************************************************************
#1. BRAINY FRUITS GAME :*******************************************************************************************************************

# PYDANTIC MODELS : *************************************************
class AnswerRequest(BaseModel):
    session_id: str
    user_answer: int

class GameStateResponse(BaseModel):
    session_id: str
    puzzle_expression: str
    fruit_values: dict
    points: int
    reward_message: str
    lives: int = None
    level: int = None
    question_number: int = None
    total_questions_in_level: int = None
    time_left: int = None  # new field 

# IT IS BASICALLY A HEART OF THE GAME WHICH IS STORE THE WHOLE LEVEL DATA OF IN-MEMORY SESSIONS*********************  
GAME_SESSIONS = {}

FRUITS = ["\U0001F34E", "\U0001F34D", "\U0001F96D"]
OPERATORS = ["+", "-", "*"]

TIME_LIMIT = 60  # seconds per question ek question ko 60 second ka time rahega 

# def get_time_limit_for_level(level: int) -> int:
#     """
#     Returns time limit based on level range:
#     - Levels 1–30: 60 seconds
#     - Levels 31–50: 90 seconds
#     - Levels 51+: 120 seconds
#     """
#     if level <= 30:
#         return 60
#     elif level <= 50:
#         return 90
#     else:
#         return 120


def get_question_count_for_level(level: int):
    if 1 <= level <= 5:
        return 3  
    elif 6 <= level <= 15:
        return 5
    elif 16 <= level <= 30:
        return 7
    elif 31 <= level <= 50:
        return 10
    else:
        return 12

def get_level_completion_message(level: int):
    if level == 5:
        return "Wow! You've mastered the basics! Welcome to the Fun Zone of fruits!"
    elif level == 15:
        return "Awesome! You're now entering the Smart Fruits League — where fun meets brainpower!"
    elif level == 30:
        return "Incredible! You've reached the Genius Grove! Keep the energy up!"
    elif level == 50:
        return "Unbelievable! You’re now a Fruit Legend! Ready for the ultimate challenge?"
    elif level >= 100:
        return "You’ve conquered all 100 levels! The Fruit Kingdom salutes you!"
    return ""


def get_lives_for_level(level: int) -> int:
    if 1 <= level <=5:
        return 1
    elif 6 <= level <= 15:
        return 2
    elif 16 <= level <= 30:
        return 3
    elif 31 <= level <= 50:
        return 4
    elif 51 <= level <= 100:
        return 5
    else:
        return 1  # default fallback


def generate_advanced_puzzle(level: int):
    basic_fruits = ["\U0001F34E", "\U0001F34D", "\U0001F96D"]
    medium_fruits = ["\U0001F34E", "\U0001F34D", "\U0001F96D", "\U0001F349"]
    hard_fruits = ["\U0001F34E", "\U0001F34D", "\U0001F96D", "\U0001F349", "\U0001F347"]
     
    if level <= 30:
        fruits = basic_fruits
    elif level <= 50:
        fruits = medium_fruits
    else:
        fruits = hard_fruits

    # Updated: For levels 1-50, max_val capped at 99 (2 digits), only + and -
    # For levels >50, max_val 99, include *, prefer single digits (1-9 mostly)
    if level <= 50:
        max_val = 99
        operators = ["+", "-"]
        # Prefer single digits for simplicity
        fruit_values = {fruit: random.randint(1, 9) if random.random() < 0.7 else random.randint(10, 99) for fruit in fruits}
    else:
        max_val = 99
        operators = ["+", "-", "*"]
        # Prefer single digits
        fruit_values = {fruit: random.randint(1, 9) if random.random() < 0.8 else random.randint(10, 99) for fruit in fruits}

    op_count = len(fruits) - 1
    ops = [random.choice(operators) for _ in range(op_count)]

    # difficulty-based expression
    if level <= 30:
        expr = f"{fruits[0]} {ops[0]} {fruits[1]} {ops[1]} {fruits[2]}"
    elif 31 <= level <= 50:
        expr = f"({fruits[0]} {ops[0]} {fruits[1]}) {ops[1]} ({fruits[2]} {ops[2]} {fruits[3]})"
    else:
        expr = f"(({fruits[0]} {ops[0]} {fruits[1]}) {ops[1]} ({fruits[2]} {ops[2]} {fruits[3]})) {ops[3]} {fruits[4]}"

    eval_expr = expr
    for fruit, val in fruit_values.items():
        eval_expr = eval_expr.replace(fruit, str(val))

    correct_answer = eval(eval_expr)
    return expr, fruit_values, correct_answer

def generate_puzzle(level: int):
    if level > 30:
        return generate_advanced_puzzle(level)

    # Updated: For levels 1-30, max_val 99, only + and -, prefer single digits
    max_val = 99
    operators = ["+", "-"]
    fruit_values = {fruit: random.randint(1, 9) if random.random() < 0.7 else random.randint(10, 99) for fruit in FRUITS}
    op1 = random.choice(operators)
    op2 = random.choice(operators)
    expr = f"{FRUITS[0]} {op1} {FRUITS[1]} {op2} {FRUITS[2]}"
    eval_expr = f"{fruit_values[FRUITS[0]]}{op1}{fruit_values[FRUITS[1]]}{op2}{fruit_values[FRUITS[2]]}"
    correct_answer = eval(eval_expr)
    return expr, fruit_values, correct_answer

@app.get("//kids/v1/welcome/{child_id}", tags=["BRAINY FRUITS"])
def welcome_player_spell(child_id: int, db: Session = Depends(get_db)):
    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    progress = (
        db.query(GameProgress)
        .filter(GameProgress.child_id == child_id, GameProgress.status == "Completed")
        .order_by(GameProgress.level.desc())
        .first()
    )

    completed_level = progress.level if progress else 0
    next_level = completed_level + 1
    if next_level > 100: 
        next_level = 100

    # Random motivationala messaseges
    welcome_messages = [
        f"Welcome, {child.username}! Ready to conquer Level {next_level}? You’re unstoppable!",
        f"Hey {child.username}, Level {next_level} awaits! Let’s break some spells!",
        f"Yo champ {child.username}! Level {next_level} is waiting for your magic touch!",
        f"Good to see you, {child.username}! It’s time to shine at Level {next_level}!",
        f"{child.username}, your brain’s on fire! Let’s hit Level {next_level}!",
        f"Back again, {child.username}? Level {next_level} doesn’t stand a chance!",
        f"Hey genius {child.username}, ready for Level {next_level}? Let’s gooo! ",
        f"Welcome back, {child.username}! Level {next_level} is your next big win! ",
        f"{child.username}, your journey continues! Level {next_level} is calling! ",
        f"Let’s make some magic, {child.username}! Time for Level {next_level}! "
    ]

    random_message = random.choice(welcome_messages)

    return {
        "message": random_message
    }



# START ENDPOINT : *****************************************************************************************************************************
"""Ab ye endpoint ekdum perfect hai isme kya kiya sir maine agar hamne 4 levels complete kr chuki hai to automatically 5th levels se game resume ho jayega"""
@app.get("//kids/v1/game/start", response_model=GameStateResponse, tags=["BRAINY FRUITS"])
def start_game(
    # session_id: str = None,
    child_id: int = None,
    db: Session = Depends(get_db)
):
    
    # Random welcome messages fro motivating the childs 
    welcome_messages = [
        "Let's rock this level! Ready, champ?",
        "Boom! New challenge unlocked!",
        "Blast off! Your brain is going turbo mode!",
        "You’re unstoppable! Let’s break records!",
        "Brainpower loading… Get ready to flex those neurons!",
        "Focus mode ON! Time to dominate!",
        "Welcome back genius! Let's crush this!",
        "Legend mode activated — show your power!",
        "The fruit puzzles are waiting for your magic!",
        "Game begins! Let’s make this round legendary!"
    ]

    # # If session already exists
    # if session_id and session_id in GAME_SESSIONS:
    #     session = GAME_SESSIONS[session_id]
    #     elapsed = int(time.time() - session["start_time"])
    #     time_left = max(0, TIME_LIMIT - elapsed)

    #     # If session already exists
    # if session_id and session_id in GAME_SESSIONS:
    #     session = GAME_SESSIONS[session_id]
    #     elapsed = int(time.time() - session["start_time"])
    #     time_limit = get_time_limit_for_level(session["level"])
    #     time_left = max(0, time_limit - elapsed)   

        # return GameStateResponse(
        #     session_id=session_id,
        #     puzzle_expression=session["puzzle_expr"],
        #     fruit_values=session["fruit_values"],
        #     points=session["points"],
        #     reward_message=f"Level {session['level']} Resume! Question {session['question_number']}/{session['total_questions']}.",
        #     lives=session["lives"],
        #     level=session["level"],
        #     question_number=session["question_number"],
        #     total_questions_in_level=session["total_questions"],
        #     time_left=time_left
        # )

    # If user id given, fetch current level progress
    if child_id:
        # FIX: get all completed levels first
        completed_levels = db.query(GameProgress).filter(
            GameProgress.child_id == child_id,
            GameProgress.status == "Completed"
        ).all()

        if completed_levels:
            completed_level_numbers = [p.level for p in completed_levels]
            next_level = max(completed_level_numbers) + 1
        else:
            next_level = 1

        # Don’t exceed total levels (100)
        if next_level > 100:
            next_level = 100

        level = next_level
    else:
        level = 1

    # IMPORTANT: Slot-wise lives logic applied here*****************inoe
    lives = get_lives_for_level(level)        

    # Rest code remains EXACTLY same 
    session_id = str(uuid.uuid4())
    # lives = 3
    total_questions = get_question_count_for_level(level)
    question_number = 1
    puzzle_expr, fruit_values, answer = generate_puzzle(level)

    GAME_SESSIONS[session_id] = {
        "level": level,
        "correct_answer": answer,
        "points": 0,
        "lives": lives,
        "puzzle_expr": puzzle_expr,
        "fruit_values": fruit_values,
        "question_number": question_number,
        "total_questions": total_questions,
        "start_time": time.time(),
    }

    random_message = random.choice(welcome_messages)

    return GameStateResponse(
        session_id=session_id,
        puzzle_expression=puzzle_expr,
        fruit_values=fruit_values,
        points=0,
        reward_message=f"{random_message} Level {level} Started! Question {question_number}/{total_questions}. Solve it using fruit values!",
        lives=lives,
        level=level,
        question_number=question_number,
        total_questions_in_level=total_questions,
        time_left=TIME_LIMIT
    )


# SELECT LEVEL ENDPOINT : ******************************************************************************************************************
@app.get("//kids/v1/game/select_level/{child_id}", tags=["BRAINY FRUITS"])
def select_level(child_id: int, desired_level: int = Query(None, description="Level you want to play"), db: Session = Depends(get_db)):
    """
    Fetch user's unlocked levels and allow selecting any unlocked level.
    If desired_level is None, default to the next unlocked level.
    """
    # Fetch progress
    progress_list = db.query(GameProgress).filter(GameProgress.child_id == child_id).all()
    completed_levels = [p.level for p in progress_list if p.status == "Completed"]
    next_unlocked = max(completed_levels, default=0) + 1

    # Determine which level user can play
    if desired_level is None:
        level_to_play = next_unlocked
    else:
        # Allow only unlocked levels
        if desired_level > next_unlocked:
            raise HTTPException(status_code=403, detail=f"Level {desired_level} is locked. You can play up to level {next_unlocked}.")
        level_to_play = desired_level

    #Dynamic time limit based on level
    # time_limit = get_time_limit_for_level(level_to_play)

    # *** DYNAMIC LIVES ADDED HERE ***
    lives = get_lives_for_level(level_to_play)    

    # Initialize session for selected level
    session_id = str(uuid.uuid4())
    lives = 3
    total_questions = get_question_count_for_level(level_to_play)
    question_number = 1
    puzzle_expr, fruit_values, answer = generate_puzzle(level_to_play)

    GAME_SESSIONS[session_id] = {
        "level": level_to_play,
        "correct_answer": answer,
        "points": 0,
        "lives": lives,
        "puzzle_expr": puzzle_expr,
        "fruit_values": fruit_values,
        "question_number": question_number,
        "total_questions": total_questions,
        "start_time": time.time(),
    }

    return GameStateResponse(
        session_id=session_id,
        puzzle_expression=puzzle_expr,
        fruit_values=fruit_values,
        points=0,
        reward_message=f"Level {level_to_play} Started! Question {question_number}/{total_questions}. Solve it using fruit values!",
        lives=lives,
        level=level_to_play,
        question_number=question_number,
        total_questions_in_level=total_questions,
        # time_left=TIME_LIMIT
        time_left=TIME_LIMIT
    )

# 08/11/2025*****************************************************************************submit22:21
# SUBMIT ANSWER ENDPOINT : ************************************************************************************************************
@app.post("//kids/v1/game/submit_answer/{child_id}", response_model=GameStateResponse, tags=["BRAINY FRUITS"])
def submit_answer(request: AnswerRequest, child_id: int, db: Session = Depends(get_db)):
    session = GAME_SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Apply slot-wise lives always (important fix)
    session["lives"] = min(session["lives"], get_lives_for_level(session["level"]))        
    
    #FIX: Always define these variables to avoid UnboundLocalError
    puzzle_expr = session.get("puzzle_expr") if session else None
    fruit_values = session.get("fruit_values") if session else None
    reward_msg = ""    

    # Check timeout 
    elapsed = int(time.time() - session["start_time"])
    if elapsed >= TIME_LIMIT:
        session["lives"] -= 1
        #New line adde jari question time out zala ase l tri pn**********
        session["question_number"] += 1        
        if session["lives"] <= 0:
            # NEW LOGIC: update temp_status & temp_last_updated when player loses all lives
            failed_level = session["level"]
            progress = db.query(GameProgress).filter(
                GameProgress.child_id == child_id,
                GameProgress.level == failed_level
            ).first()

            if not progress:
                progress = GameProgress(
                    child_id=child_id,
                    level=failed_level,
                    temp_status="Failed",
                    temp_last_updated=datetime.utcnow()
                )
                db.add(progress)
            else:
                progress.temp_status = "Failed"
                progress.temp_last_updated = datetime.utcnow()

            db.commit()

            return GameStateResponse(
                session_id=request.session_id,
                puzzle_expression=session["puzzle_expr"],
                fruit_values=session["fruit_values"],
                points=session["points"],
                reward_message=f"Time Out! Game Over! You lost all your lives. and Correct answer is {session['correct_answer']}",
                lives=0,
                level=session["level"],
                question_number=session["question_number"],
                total_questions_in_level=session["total_questions"],
                time_left=0
            )

        #NEW LOGIC (generate new puzzle on timeout)********************13/11********************
        puzzle_expr, fruit_values, answer = generate_puzzle(session["level"])
        session.update({
            "correct_answer": answer,
            "puzzle_expr": puzzle_expr,
            "fruit_values": fruit_values,
            "start_time": time.time()
        })
        # *****************13/11********************

        return GameStateResponse(
            session_id=request.session_id,
            puzzle_expression=session["puzzle_expr"],
            fruit_values=session["fruit_values"],
            points=session["points"],
            reward_message=f"Time Out! One life lost. Lives left: {session['lives']}.",
            lives=session["lives"],
            level=session["level"],
            question_number=session["question_number"],
            total_questions_in_level=session["total_questions"],
            time_left=0
        )

    # Normal answer check
    user_answer = request.user_answer
    correct_answer = session["correct_answer"]
    level = session["level"]
    points = session["points"]
    lives = session["lives"]
    question_number = session["question_number"]
    total_questions = session["total_questions"]

    if user_answer == correct_answer:
        points += 10
        reward_msg = f"Correct! +10 points."
        question_number += 1

        # NEW LOGIC ADDED HERE — Stop auto next level start
        if question_number > total_questions:
            level_completed = level
            reward_msg = f"Level {level_completed} completed successfully! Please start the next level manually."

            # Update database for completed level
            progress = db.query(GameProgress).filter(
                GameProgress.child_id == child_id,
                GameProgress.level == level_completed
            ).first()

            if progress:
                progress.status = "Completed"
                progress.score = points
                progress.last_updated = datetime.utcnow()
            else:
                progress = GameProgress(
                    child_id=child_id,
                    level=level_completed,
                    status="Completed",
                    score=points,
                    last_updated=datetime.utcnow()
                )
                db.add(progress)
            db.commit()

            # Return response instead of auto-starting next level
            return GameStateResponse(
                session_id=request.session_id,
                puzzle_expression=session["puzzle_expr"],
                fruit_values=session["fruit_values"],
                points=points,
                reward_message=reward_msg,
                lives=lives,
                level=level_completed,
                question_number=question_number - 1,
                total_questions_in_level=total_questions,
                time_left=0
            )

        # (rest unchanged)
        puzzle_expr, fruit_values, answer = generate_puzzle(level)
        if 31 <= level <= 50:
            reward_msg += "Solve using BODMAS rule!"
        elif level > 51:
            reward_msg += "You’re now solving Master-level fruit puzzles — remember the BODMAS rule!"

        session.update({
            "level": level,
            "correct_answer": answer,
            "points": points,
            "puzzle_expr": puzzle_expr,
            "fruit_values": fruit_values,
            "lives": lives,
            "question_number": question_number,
            "total_questions": total_questions,
            "start_time": time.time(),
        })
        time_left = TIME_LIMIT

    else:
        lives -= 1
        session["question_number"] += 1
        if lives <= 0:
            reward_msg = f"Wrong! Correct answer was {correct_answer}. Game Over!"
            session["lives"] = 0
            time_left = 0

            # NEW LOGIC: update temp_status & temp_last_updated when player loses by wrong answer
            failed_level = session["level"]
            progress = db.query(GameProgress).filter(
                GameProgress.child_id == child_id,
                GameProgress.level == failed_level
            ).first()

            if not progress:
                progress = GameProgress(
                    child_id=child_id,
                    level=failed_level,
                    temp_status="Failed",
                    temp_last_updated=datetime.utcnow()
                )
                db.add(progress)
            else:
                progress.temp_status = "Failed"
                progress.temp_last_updated = datetime.utcnow()

            db.commit()

        else:
            reward_msg = f"Wrong! Correct answer was {correct_answer}. Lives left: {lives}. New question generated!"
            session["lives"] = lives
            puzzle_expr, fruit_values, answer = generate_puzzle(session["level"])
            session.update({
                "correct_answer": answer,
                "puzzle_expr": puzzle_expr,
                "fruit_values": fruit_values,
                "start_time": time.time()
            })
            time_left = TIME_LIMIT

    return GameStateResponse(
        session_id=request.session_id,
        puzzle_expression=puzzle_expr,
        fruit_values=fruit_values,
        points=points,
        reward_message=reward_msg,
        lives=lives,
        level=level,
        question_number=question_number,
        total_questions_in_level=total_questions,
        time_left=time_left
    )



# GAME PROGRESS ENDPOINT **************************************************************************************************************
@app.get("//kids/v1/game/progress/{child_id}", tags=["BRAINY FRUITS"])
def get_game_progress(child_id: int, db: Session = Depends(get_db)):

    progress_records = db.query(GameProgress).filter(GameProgress.child_id == child_id).all()
    if not progress_records:
        return {
            "username": None,
            "child_id": child_id,
            "completed_levels": [],
            "current_level": 1,
            "next_level": 1,
            "points_per_level": {},
            "total_levels": 100,
            "Next_Unlocked_Level": 1
        }

    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    username = child.username if child else None

    TOTAL_LEVELS = 100

    # All completed levels
    completed_levels = [p.level for p in progress_records if p.status == "Completed"]

    # Latest completed
    latest_completed = (
        db.query(GameProgress)
        .filter(GameProgress.child_id == child_id, GameProgress.status == "Completed")
        .order_by(desc(GameProgress.last_updated))
        .first()
    )

    # Latest failed
    latest_failed = (
        db.query(GameProgress)
        .filter(GameProgress.child_id == child_id, GameProgress.temp_status == "Failed")
        .order_by(desc(GameProgress.temp_last_updated))
        .first()
    )

    # -----------------------------
    #   Determine CURRENT LEVEL
    # -----------------------------
    if latest_failed and (not latest_completed or latest_failed.temp_last_updated > latest_completed.last_updated):
        current_level = latest_failed.level  # child failed this level, so this is the active one
        failure_mode = True
    elif latest_completed:
        current_level = latest_completed.level
        failure_mode = False
    else:
        current_level = 1
        failure_mode = False

    # -----------------------------
    #   Determine NEXT LEVEL
    # -----------------------------
    if failure_mode:
        # Child must replay the same level
        next_level = current_level
    else:
        # Completed successfully → next level unlocked
        next_level = current_level + 1 if current_level < TOTAL_LEVELS else TOTAL_LEVELS

    # Ensure points summary
    points_summary = {p.level: p.score for p in progress_records if p.status == "Completed"}

    highest_completed = max(completed_levels, default=0)
    next_unlocked_level = highest_completed + 1 if highest_completed < TOTAL_LEVELS else "All levels completed!"

    return {
        "username": username,
        "child_id": child_id,
        "completed_levels": completed_levels,
        "current_level": current_level,
        "next_level": next_level,
        "points_per_level": points_summary,
        "total_levels": TOTAL_LEVELS,
        "Next_Unlocked_Level": next_unlocked_level
    }

# GAME REPLAY ENDPOINT******************************************************************************************************************
# Replay endpoint latest 06-11-2025 New
@app.get("//kids/v1/game/replay_level/{child_id}", tags=["BRAINY FRUITS"])
def replay_last_played_level(child_id: int, db: Session = Depends(get_db)):
    """
    Replay the last played level automatically.
    - If last level was completed, replay that.
    - If last level was failed, replay the same failed level.
    Example: /game/replay_level/4
    """

    # STEP 1: Fetch the most recently failed and completed levels
    last_failed = (
        db.query(GameProgress)
        .filter(GameProgress.child_id == child_id, GameProgress.temp_status == "Failed")
        .order_by(GameProgress.temp_last_updated.desc())
        .first()
    )

    last_completed = (
        db.query(GameProgress)
        .filter(GameProgress.child_id == child_id, GameProgress.status == "Completed")
        .order_by(GameProgress.last_updated.desc())
        .first()
    )

    # NEW LOGIC: Compare which one is latest (failed vs completed)
    last_played = None
    if last_failed and last_completed:
        if last_failed.temp_last_updated > last_completed.last_updated:
            last_played = last_failed
        else:
            last_played = last_completed
    elif last_failed:
        last_played = last_failed
    elif last_completed:
        last_played = last_completed
    else:
        raise HTTPException(status_code=400, detail="No previous level found to replay.")

    # STEP 2: Determine which level to replay
    last_played_level = last_played.level

    # STEP 3: Initialize new session for that level
    session_id = str(uuid.uuid4())
    # lives = 3
    lives = get_lives_for_level(last_played_level) 
    total_questions = get_question_count_for_level(last_played_level)
    question_number = 1
    puzzle_expr, fruit_values, answer = generate_puzzle(last_played_level)

    # STEP 4: Store in GAME_SESSIONS
    GAME_SESSIONS[session_id] = {
        "level": last_played_level,
        "correct_answer": answer,
        "points": 0,
        "lives": lives,
        "puzzle_expr": puzzle_expr,
        "fruit_values": fruit_values,
        "question_number": question_number,
        "total_questions": total_questions,
        "start_time": time.time(),
    }

    # STEP 5: Return response
    return GameStateResponse(
        session_id=session_id,
        puzzle_expression=puzzle_expr,
        fruit_values=fruit_values,
        points=0,
        reward_message=f"Replaying Level {last_played_level}! Let's see if you can beat your last performance!",
        lives=lives,
        level=last_played_level,
        question_number=question_number,
        total_questions_in_level=total_questions,
        time_left=TIME_LIMIT
    )


# New Logic : Next_Endpoint*****************************************06/11/2025***************************************************************
# NEXT LEVEL ENDPOINT : *********************************************************************************************************
@app.get("//kids/v1/game/next_level/{child_id}", tags=["BRAINY FRUITS"])
def next_level(child_id: int, db: Session = Depends(get_db)):
    completed_levels = (
        db.query(GameProgress.level)
        .filter(GameProgress.child_id == child_id, GameProgress.status == "Completed")
        .order_by(GameProgress.level.asc())
        .all()
    )
    completed_levels = [lvl[0] for lvl in completed_levels]

    if not completed_levels:
        raise HTTPException(status_code=400, detail="No completed levels found yet.")

    TOTAL_LEVELS = 100

    # STEP 1: Find most recent completed level
    recent_completed = (
        db.query(GameProgress)
        .filter(GameProgress.child_id == child_id, GameProgress.status == "Completed")
        .order_by(GameProgress.last_updated.desc())
        .first()
    )

    # STEP 2: Find most recent failed temp level
    recent_failed = (
        db.query(GameProgress)
        .filter(GameProgress.child_id == child_id, GameProgress.temp_status == "Failed")
        .order_by(GameProgress.temp_last_updated.desc())
        .first()
    )

    # STEP 3: Decide which one is most recent (compare timestamps)
    if recent_failed and recent_completed:
        # Compare their timestamps
        latest_entry = (
            recent_failed
            if recent_failed.temp_last_updated > recent_completed.last_updated
            else recent_completed
        )
    else:
        latest_entry = recent_failed or recent_completed

    if not latest_entry:
        raise HTTPException(status_code=400, detail="No valid recent progress found.")

    last_level = latest_entry.level


    #  ADDED: Prevent next level unlock if last played level is failed & not completed yet
    if (
        hasattr(latest_entry, "temp_status")
        and latest_entry.temp_status == "Failed"
        and last_level not in completed_levels
    ):
        raise HTTPException(
            status_code=403,
            detail=f"Level {last_level} not completed yet. You must complete it before moving to the next level!"
        )    

    # STEP 4: If latest was failed → replay same level, else → move to next
    if hasattr(latest_entry, "temp_status") and latest_entry.temp_status == "Failed":
        # next_level = last_level  # replay same failed level
    # else:
        next_level = last_level + 1

    # Safety line (only for UnboundLocalError prevention)
    if 'next_level' not in locals():
        next_level = last_level + 1        

    if next_level > TOTAL_LEVELS:
        return {
            "message": "All levels are already completed!",
            "child_id": child_id,
            "total_levels": TOTAL_LEVELS
        }

    # STEP 5: Start session for next/replayed level
    session_id = str(uuid.uuid4())
    # lives = 3
    lives = get_lives_for_level(next_level) 
    total_questions = get_question_count_for_level(next_level)
    question_number = 1
    puzzle_expr, fruit_values, answer = generate_puzzle(next_level)

    GAME_SESSIONS[session_id] = {
        "level": next_level,
        "correct_answer": answer,
        "points": 0,
        "lives": lives,
        "puzzle_expr": puzzle_expr,
        "fruit_values": fruit_values,
        "question_number": question_number,
        "total_questions": total_questions,
        "start_time": time.time(),
    }

    return GameStateResponse(
        session_id=session_id,
        puzzle_expression=puzzle_expr,
        fruit_values=fruit_values,
        points=0,
        reward_message=f"Starting Level {next_level}! Good luck!",
        lives=lives,
        level=next_level,
        question_number=question_number,
        total_questions_in_level=total_questions,
        time_left=TIME_LIMIT
    )


# SPELL BREAKER : ******************************************************************************************************************************
# Full code for Spell Breaker game with lives logic applied across all endpoints.
# Lives are assigned based on level slots: 1-15: 1 life, 16-30: 2 lives, 31-50: 3 lives, 51-100: 4 lives.

Base.metadata.create_all(bind=engine)

LEVEL_CONFIG = {
    "1-15": {"q": 3, "time": 30, "difficulty": "easy"},
    "16-30": {"q": 7, "time": 45, "difficulty": "medium"},
    "31-50": {"q": 10, "time": 60, "difficulty": "hard"},
    "51-100": {"q": 12, "time": 100, "difficulty": "master"},
}

def get_lives_for_spellBreaker_level(level: int) -> int:
    if 1 <= level <= 15:
        return 1
    elif 16 <= level <= 30:
        return 2
    elif 31 <= level <= 50:
        return 3
    elif 51 <= level <= 100:
        return 4
    else:
        return 1  

def questions_for_level(level: int) -> int:
    if 1 <= level <= 15:
        return 3
    elif 16 <= level <= 30:
        return 7
    elif 31 <= level <= 50:
        return 10
    else:
        return 12

def time_for_level(level: int) -> int:
    if 1 <= level <= 15:
        return 30
    elif 16 <= level <= 30:
        return 45
    elif 31 <= level <= 50:
        return 60
    else:
        return 100

def difficulty_for_level(level: int) -> str:
    if 1 <= level <= 15:
        return "easy"
    elif 16 <= level <= 30:
        return "medium"
    elif 31 <= level <= 50:
        return "hard"
    else:
        return "master"

# session structure:
# {
#   session_id: {
#       "child_id": int,
#       "level": int,
#       "question_idx": int,  # 1-based
#       "total_questions": int,
#       "correct_answer": str or int,
#       "jumbled_word": str,
#       "options": dict,
#       "points": int,
#       "lives": int,
#       "start_time": float (epoch for current question),
#       "time_limit": int (seconds),
#   }
# }

# USE TO STORE A DATA OF SESSIONS IN MEMORY
GAME_SESSIONS = {}

def get_openai_client():
    global openai_client
    if openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured in environment")
        openai_client = OpenAI(api_key=api_key)
    return openai_client

# Keep this global variable at top of file (outside function)
GENERATED_WORDS = set()

def generate_jumbled_puzzle(difficulty: str, max_retries: int = 2):
    """
    Call OpenAI to return JSON with keys:
    {
      "jumbled_word": "DUNORG",
      "options": {"a":"GROUND", "b":"GRODUN", "c":"RONGUD", "d":"GUNDOR"},
      "correct_option": "a",
      "fact": "Ground is where..."
    }
    """
    client = get_openai_client()

    # Make sure previously generated words are not repeated
    prompt = f"""
You are a kids' word puzzle generator. Produce ONLY valid JSON and nothing else.

Task:
- Choose one English word of difficulty level "{difficulty}" suitable for 5-15 year old kids.
- Scramble it to create a jumbled_word (shuffle letters).
- Provide four options labeled a,b,c,d. One option is correct (the unscrambled word), others are plausible distractors using same letters if possible.
- Provide the correct option key (a/b/c/d).
- Provide a one-line playful/educational fact about the correct word.
- Do NOT repeat any word or jumbled word that has appeared previously in this game session.
- The following words and jumbled words are already used: {list(GENERATED_WORDS)}
- Generate a jumbled word puzzle where only one meaningful English word can be formed from the letters.
- Double-check that no other permutation of those letters forms any valid English word besides the correct one.
- Make sure that the jumbled_word uses exactly the same letters as the correct word — no extra or missing letters.
*************** Do not use any jumbled word or correct word that appeared in any previous question.
-Ensure the correct option is NOT always 'a'. Randomly place the correct answer among a, b, c, and d.

Example:
{{
  "jumbled_word": "DUNORG",
  "options": {{
    "a": "GROUND",
    "b": "GRODUN",
    "c": "RONGUD",
    "d": "GUNDOR"
  }},
  "correct_option": "a",
  "fact": "GROUND is where you run and play!"
}}

Now generate the puzzle at difficulty: {difficulty}
"""

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative, child-friendly puzzle generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)

            if {"jumbled_word", "options", "correct_option", "fact"}.issubset(data.keys()):
                jw = data["jumbled_word"].upper().strip()
                correct_word = data["options"].get(data["correct_option"], "").upper().strip()

                # Check repetition logic
                if jw not in GENERATED_WORDS and correct_word not in GENERATED_WORDS:
                    GENERATED_WORDS.add(jw)
                    GENERATED_WORDS.add(correct_word)
                    return data
                else:
                    print(f"Repeated word '{jw}' or '{correct_word}' detected. Retrying...")
                    continue

        except Exception as e:
            if attempt == max_retries:
                word = "APPLE" if difficulty in ("easy", "medium") else "ORANGE"
                j = "".join(random.sample(word, len(word)))
                options = {
                    "a": word,
                    "b": word[::-1],
                    "c": word[1:] + word[0],
                    "d": "".join(reversed(word[1:]))
                }
                GENERATED_WORDS.add(word)
                GENERATED_WORDS.add(j)
                return {
                    "jumbled_word": j,
                    "options": options,
                    "correct_option": "a",
                    "fact": f"{word} is yummy!"
                }

    # Default fallback
    fallback = {
        "jumbled_word": "APPLE",
        "options": {"a": "APPLE", "b": "APPEL", "c": "ALEPP", "d": "PEPLA"},
        "correct_option": "a",
        "fact": "APPLE is healthy for our health!"
    }
    GENERATED_WORDS.add("APPLE")
    return fallback

def get_level_motivation(level:int, completed_count:int):
    if level <= 15 and completed_count == 5:
        return "Wow! You've mastered the basics! Welcome to the Fun Zone of SpellBReakeR!"
    if level <= 30 and completed_count == 7:
        return "Awesome! You're now entering the Smart SpellBreakLeague — where fun meets brainpower!"
    if level <= 50 and completed_count == 10:
        return "Incredible! You've reached the Genius Grove! Keep the energy up!"
    if level >= 50 and completed_count == 12:
        return "Unbelievable! You’re now a Fruit Legend! Ready for the ultimate challenge?"
    return ""

class StartGameResponse(BaseModel):
    session_id: str
    puzzle_expression: str 
    options: dict
    points: int
    reward_message: str
    lives: int
    level: int
    question_number: int
    total_questions_in_level: int
    time_left: int

class SubmitAnswerRequest(BaseModel):
    session_id: str
    selected_option: str  # 'a'/'b'/'c'/'d'

class NextQuestionModel(BaseModel):
    level: int
    question_number: int
    puzzle_expression: str
    options: Dict[str, str]  
    # session_id: str

class SubmitAnswerResponse(BaseModel):
    session_id: str
    correct: bool
    correct_option: str
    fact: str
    points: int
    lives: int
    level: int
    question_number: int
    total_questions_in_level: int
    time_left: int
    reward_message: str
    # next_question: Optional[NextQuestionModel] = None old logic here
    puzzle_expression: Optional[str] = None
    options: Optional[Dict[str, str]] = None

@app.get("//kids/v1/welcome/spell/{child_id}", tags=["SPELL BREAKER"])
def welcome_player_spell(child_id: int, db: Session = Depends(get_db)):
    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    # Fetch the highest completed level for this child
    progress = (
        db.query(GameProgress_SPELL)
        .filter(GameProgress_SPELL.child_id == child_id, GameProgress_SPELL.status == "Completed")
        .order_by(GameProgress_SPELL.level.desc())
        .first()
    )

    # Determine next level
    completed_level = progress.level if progress else 0
    next_level = completed_level + 1
    if next_level > 100: 
        next_level = 100

    # Random motivational messages deto ha
    welcome_messages = [
        f"Welcome, {child.username}! Ready to conquer Level {next_level}? You’re unstoppable!",
        f"Hey {child.username}, Level {next_level} awaits! Let’s break some spells!",
        f"Yo champ {child.username}! Level {next_level} is waiting for your magic touch! ",
        f"Good to see you, {child.username}! It’s time to shine at Level {next_level}!",
        f"{child.username}, your brain’s on fire! Let’s hit Level {next_level}!",
        f"Back again, {child.username}? Level {next_level} doesn’t stand a chance!",
        f"Hey genius {child.username}, ready for Level {next_level}? Let’s gooo!",
        f"Welcome back, {child.username}! Level {next_level} is calling!",
        f"{child.username}, your journey continues! Level {next_level} is waiting!",
        f"Let’s make some magic, {child.username}! Time for Level {next_level}!"
    ]

    random_message = random.choice(welcome_messages)
    return {
        "message": random_message
    }

# STATRT ENDPOINT : ******************************************************************************************************************
@app.get("//kids/v1/game/start/spell", response_model=StartGameResponse, tags=["SPELL BREAKER"])
def start_game(
    # session_id: Optional[str] = None,
    child_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Start or resume a session for SPELL BREAKER game.
    """

    # Random welcome messages for motivation
    welcome_messages = [
        "Spelling wizard mode activated!",
        "Boom! New challenge unlocked — let’s spell it right!",
        "Your brain’s on fire ! Time to conquer new words!",
        "Let’s twist those letters and rule the board!",
        "Focus mode ON — Let’s master this round!",
        "Welcome back champ! Let’s break some spells!",
        "The magic words await you!",
        "You’re unstoppable! Let’s spell with power!",
        "Ready, genius? Let’s play!",
        "Brainstorm time! Let’s spell some fun!"
    ]

    #If new child session starts
    if not child_id:
        raise HTTPException(status_code=400, detail="child_id is required to start a new game.")

    # Ensure child exists
    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    # Fetch completed levels for this child
    completed_levels = db.query(GameProgress_SPELL).filter(
        GameProgress_SPELL.child_id == child_id,
        GameProgress_SPELL.status == "Completed"
    ).all()

    # Determine next unlocked level (auto advance logic added)
    if completed_levels:
        completed_level_numbers = [p.level for p in completed_levels]
        next_level = max(completed_level_numbers) + 1
    else:
        next_level = 1

    if next_level > 100:
        next_level = 100

    level = next_level  #Always starts from the next unlocked level

    #Generate new game data
    total_q = questions_for_level(level)
    time_limit = time_for_level(level)
    lives = get_lives_for_spellBreaker_level(level)        
    points = 0
    q_idx = 1

    difficulty = difficulty_for_level(level)
    puzzle = generate_jumbled_puzzle(difficulty)
    jumbled = puzzle["jumbled_word"]
    options = puzzle["options"]
    correct_option = puzzle["correct_option"]
    fact = puzzle["fact"]

    # Create new session
    session_id = str(uuid.uuid4())
    GAME_SESSIONS[session_id] = {
        "child_id": child_id,
        "level": level,
        "question_idx": q_idx,
        "total_questions": total_q,
        "correct_option": correct_option,
        "jumbled_word": jumbled,
        "options": options,
        "points": points,
        "lives": lives,
        "start_time": time.time(),
        "time_limit": time_limit,
        "fact": fact
    }

    random_message = random.choice(welcome_messages)
    reward_message = f"{random_message} Level {level} Started! Question {q_idx}/{total_q}. Time per question: {time_limit}s. Good luck, {child.username}!"

    # Final response
    return StartGameResponse(
        session_id=session_id,
        puzzle_expression=jumbled,
        options=options,
        points=points,
        reward_message=reward_message,
        lives=lives,
        level=level,
        question_number=q_idx,
        total_questions_in_level=total_q,
        time_left=time_limit
    )

# GAME SELECT LEVEL ENDPOINT*****************************************************************************************************
@app.get("//kids/v1/game/select_level/spell/{child_id}", tags=["SPELL BREAKER"])
def select_level(
    child_id: int,
    db: Session = Depends(get_db),
    desired_level: Optional[int] = None
):
    """
    Return the unlocked levels and allow selecting any unlocked level.
    If desired_level is None, return next unlocked level suggested.
    Directly starts the selected level (no need to call /game/start separately).
    """
    # fetch progress entries
    rows = db.query(GameProgress_SPELL).filter(GameProgress_SPELL.child_id == child_id).all()
    if not rows:
        raise HTTPException(status_code=404, detail="No progress found. Maybe signup incomplete.")

    completed_levels = [r.level for r in rows if r.status == "Completed"]
    next_unlocked = max(completed_levels, default=0) + 1
    if next_unlocked > 100:
        next_unlocked = 100

    # allow user to select any unlocked (<= next_unlocked) level
    if desired_level is None:
        level_to_play = next_unlocked
    else:
        if desired_level > next_unlocked:
            raise HTTPException(
                status_code=403,
                detail=f"Level {desired_level} is locked. You can play up to level {next_unlocked}."
            )
        level_to_play = desired_level

    time_limit = time_for_level(level_to_play)    
    lives = get_lives_for_spellBreaker_level(level_to_play)

    # Directly initialize the Spell Breaker session (like /game/start)
    session_id = str(uuid.uuid4())
    ##lives = 3
    total_questions = questions_for_level(level_to_play)
    question_number = 1

    # generate jumbled puzzle for the first question
    difficulty = difficulty_for_level(level_to_play)
    puzzle = generate_jumbled_puzzle(difficulty)

    GAME_SESSIONS[session_id] = {
        "child_id": child_id,
        "level": level_to_play,
        "points": 0,
        "lives": lives,
        "question_idx": question_number,
        "total_questions": total_questions,
        "jumbled_word": puzzle["jumbled_word"],
        "options": puzzle["options"],
        "correct_option": puzzle["correct_option"],
        "fact": puzzle["fact"],
        "start_time": time.time(),
        "time_limit": time_limit,
    }

    return {
        "session_id": session_id,
        "puzzle_expression": puzzle["jumbled_word"],  
        "options": puzzle["options"],
        "points": 0,
        "reward_message": f"Spelling wizard mode activated! Level {level_to_play} Started! Question {question_number}/{total_questions}. Time per question: {time_limit}s. Good luck, Tiger!",
        "lives": lives,
        "level": level_to_play,
        "question_number": question_number,
        "total_questions_in_level": total_questions,
        "time_left": time_limit
    }


# SUBMIT ANSWER ENDPOINT*************************************************************************************************************
@app.post("//kids/v1/game/submit_answer/spell/{child_id}", response_model=SubmitAnswerResponse, tags=["SPELL BREAKER"])
def submit_answer(req: SubmitAnswerRequest, child_id : int, db: Session = Depends(get_db)):
    session = GAME_SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    elapsed = int(time.time() - session["start_time"])
    time_left = max(0, session["time_limit"] - elapsed)
    session["lives"] = min(session["lives"], get_lives_for_spellBreaker_level(session["level"]))  # Ensures lives don't exceed level's max (lives logic applied)

    if time_left <= 0:
        session["lives"] -= 1
        # New line added — jari question time out zala ase l tri pn **********
        
             # FIX: prevent KeyError ithe fix kele ************
        session.setdefault("question_number", 0)
        session["question_number"] += 1 

        #If timeout happens after last question, end game immediately
        if session["question_idx"] >= session["total_questions"]:
            session["lives"] = 0
            # ithee pn new logic add for level failed
            gp = db.query(GameProgress_SPELL).filter(
                GameProgress_SPELL.child_id == session["child_id"],
                GameProgress_SPELL.level == session["level"]
            ).first()

            if gp:
                gp.temp_status = "Failed"
                gp.temp_last_updated = datetime.utcnow()
            else:
                gp = GameProgress_SPELL(
                    child_id=session["child_id"],
                    level=session["level"],
                    temp_status="Failed",
                    temp_last_updated=datetime.utcnow()
                )
                db.add(gp)
            db.commit()

            return SubmitAnswerResponse(
                session_id=req.session_id,
                correct=False,
                correct_option=session["correct_option"],
                fact=session["fact"],
                points=session["points"],
                lives=0,
                level=session["level"],
                question_number=session["question_idx"],
                total_questions_in_level=session["total_questions"],
                time_left=0,
                reward_message="Time up! No more questions left. Game Over!",
            )

        if session["lives"] <= 0:
            session["lives"] = 0
            # ithe new logic add kele like level fail zala tr
            gp = db.query(GameProgress_SPELL).filter(
                GameProgress_SPELL.child_id == session["child_id"],
                GameProgress_SPELL.level == session["level"]
            ).first()

            if gp:
                gp.temp_status = "Failed"
                gp.temp_last_updated = datetime.utcnow()
            else:
                gp = GameProgress_SPELL(
                    child_id=session["child_id"],
                    level=session["level"],
                    temp_status="Failed",
                    temp_last_updated=datetime.utcnow()
                )
                db.add(gp)
            db.commit()

            return SubmitAnswerResponse(
                session_id=req.session_id,
                correct=False,
                correct_option=session["correct_option"],
                fact=session["fact"],
                points=session["points"],
                lives=0,
                level=session["level"],
                question_number=session["question_idx"],
                total_questions_in_level=session["total_questions"],
                time_left=0,
                reward_message=f"Time up! You lost. Correct answer was option '{session['correct_option']}'.",
            )

        #Generate new question if lives left
        next_difficulty = difficulty_for_level(session["level"])
        next_puzzle = generate_jumbled_puzzle(next_difficulty)
        session.update({
            "correct_option": next_puzzle["correct_option"],
            "jumbled_word": next_puzzle["jumbled_word"],
            "options": next_puzzle["options"],
            "fact": next_puzzle["fact"],
            "start_time": time.time()
        })
        return SubmitAnswerResponse(
            session_id=req.session_id,
            correct=False,
            correct_option=session["correct_option"],
            fact=session["fact"],
            points=session["points"],
            lives=session["lives"],
            level=session["level"],
            question_number=session["question_idx"],
            total_questions_in_level=session["total_questions"],
            time_left=session["time_limit"],
            reward_message=f"Time up! One life lost. Lives left: {session['lives']}.",
            # next_question=NextQuestionModel(
            #     level=session["level"],
            #     question_number=session["question_idx"],
            #     puzzle_expression=session["jumbled_word"],
            #     options=session["options"],
            #     session_id=req.session_id
            # )

            # new logic question direct pathvane dictiobary madhe nahi
            puzzle_expression=session["jumbled_word"],
            options=session["options"]
        )

    selected = req.selected_option.lower().strip()
    correct = selected == session["correct_option"].lower().strip()

    current_fact = session["fact"]
    current_correct_option = session["correct_option"]

    if correct:
        session["points"] += 10
        session["question_idx"] += 1
        reward_message = "Correct! +10 points."

        #auto next levele start hoti ithe but me change
        if session["question_idx"] > session["total_questions"]:
            completed_level = session["level"]
            total_score_for_level = session["points"]

            gp = db.query(GameProgress_SPELL).filter(
                GameProgress_SPELL.child_id == session["child_id"],
                GameProgress_SPELL.level == completed_level
            ).first()

            if gp:
                gp.status = "Completed"
                gp.score = total_score_for_level
                gp.last_updated = datetime.utcnow()
            else:
                gp = GameProgress_SPELL(
                    child_id=session["child_id"],
                    level=completed_level,
                    score=total_score_for_level,
                    status="Completed",
                    last_updated=datetime.utcnow()
                )
                db.add(gp)
            db.commit()

            #Do not auto-start next level 
            reward_message = f"Level {completed_level} completed successfully! Please start the next level manually."

            return SubmitAnswerResponse(
                session_id=req.session_id,
                correct=True,
                correct_option=current_correct_option,
                fact=current_fact,
                points=total_score_for_level,
                lives=session["lives"],
                level=completed_level,
                question_number=session["total_questions"],
                total_questions_in_level=session["total_questions"],
                time_left=0,
                reward_message=reward_message
            )

        # normal correct flow
        next_difficulty = difficulty_for_level(session["level"])
        next_puzzle = generate_jumbled_puzzle(next_difficulty)
        session["correct_option"] = next_puzzle["correct_option"]
        session["jumbled_word"] = next_puzzle["jumbled_word"]
        session["options"] = next_puzzle["options"]
        session["fact"] = next_puzzle["fact"]
        session["start_time"] = time.time()

        return SubmitAnswerResponse(
            session_id=req.session_id,
            correct=True,
            correct_option=current_correct_option,
            fact=current_fact,
            points=session["points"],
            lives=session["lives"],
            level=session["level"],
            # question_number=session["question_idx"]-1, # ya mule problem yet hota but question number cha
            question_number=session["question_idx"],
            total_questions_in_level=session["total_questions"],
            time_left=session["time_limit"],
            reward_message=reward_message,
            # next_question=NextQuestionModel(
            #     level=session["level"],
            #     question_number=session["question_idx"],
            #     puzzle_expression=session["jumbled_word"],
            #     options=session["options"],               
            #     session_id=req.session_id
            # )

            # new logic
            puzzle_expression=session["jumbled_word"],
            options=session["options"]
        )

    # Wrong answer logic unchanged
    session["lives"] -= 1
    session["question_idx"] += 1  

    #If wrong answer occurs after last question, end game immediately
    if session["question_idx"] > session["total_questions"]:
        session["lives"] = 0
        return SubmitAnswerResponse(
            session_id=req.session_id,
            correct=False,
            correct_option=current_correct_option,
            fact=current_fact,
            points=session["points"],
            lives=0,
            level=session["level"],
            question_number=session["total_questions"],
            total_questions_in_level=session["total_questions"],
            time_left=0,
            reward_message=f"Wrong! Correct answer was '{current_correct_option}'. No more questions left. Game Over!",
        )

    if session["lives"] <= 0:
        session["lives"] = 0
        gp = db.query(GameProgress_SPELL).filter(
            GameProgress_SPELL.child_id == session["child_id"],
            GameProgress_SPELL.level == session["level"]
        ).first()

        if gp:
            gp.score = session["points"]
            gp.last_updated = datetime.utcnow()
            gp.temp_status = "Failed"
            gp.temp_last_updated = datetime.utcnow()
        else:
            gp = GameProgress_SPELL(
                child_id=session["child_id"],
                level=session["level"],
                score=session["points"],
                status="Not Started",
                last_updated=datetime.utcnow(),
                temp_status="Failed",
                temp_last_updated=datetime.utcnow()
            )
            db.add(gp)
        db.commit()

        return SubmitAnswerResponse(
            session_id=req.session_id,
            correct=False,
            correct_option=current_correct_option,
            fact=current_fact,
            points=session["points"],
            lives=0,
            level=session["level"],
            question_number=session["question_idx"],
            total_questions_in_level=session["total_questions"],
            time_left=0,
            reward_message=f"Wrong! Correct answer was '{current_correct_option}'. Game Over!",
        )

    reward_message = f"Wrong! Correct answer was '{current_correct_option}'. Lives left: {session['lives']}."
    next_difficulty = difficulty_for_level(session["level"])
    next_puzzle = generate_jumbled_puzzle(next_difficulty)
    session.update({
        "correct_option": next_puzzle["correct_option"],
        "jumbled_word": next_puzzle["jumbled_word"],
        "options": next_puzzle["options"],
        "fact": next_puzzle["fact"],
        "start_time": time.time()
    })

    return SubmitAnswerResponse(
        session_id=req.session_id,
        correct=False,
        correct_option=current_correct_option,
        fact=current_fact,
        points=session["points"],
        lives=session["lives"],
        level=session["level"],
        question_number=session["question_idx"],
        total_questions_in_level=session["total_questions"],
        time_left=session["time_limit"],
        reward_message=reward_message,
        # new logic*******
        puzzle_expression=session["jumbled_word"],
        options=session["options"]
    )


# GAME PROGRESS ENDPOINT **************************************************************************************************************
@app.get("//kids/v1/game/progress/spell/{child_id}", tags=["SPELL BREAKER"])
def get_spellbreaker_progress(child_id: int, db: Session = Depends(get_db)):
    progress_records = db.query(GameProgress_SPELL).filter(GameProgress_SPELL.child_id == child_id).all()
    if not progress_records:
        raise HTTPException(status_code=404, detail="No progress found for this child.")

    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    username = child.username if child else None

    TOTAL_LEVELS = 100

    # All completed levels
    completed_levels = [p.level for p in progress_records if p.status == "Completed"]

    # Fetch the most recent completed record
    latest_completed = (
        db.query(GameProgress_SPELL)
        .filter(GameProgress_SPELL.child_id == child_id, GameProgress_SPELL.status == "Completed")
        .order_by(desc(GameProgress_SPELL.last_updated))
        .first()
    )

    # Fetch the most recent failed record
    latest_failed = (
        db.query(GameProgress_SPELL)
        .filter(GameProgress_SPELL.child_id == child_id, GameProgress_SPELL.temp_status == "Failed")
        .order_by(desc(GameProgress_SPELL.temp_last_updated))
        .first()
    )

    current_level = 1  # default

    # Determine current level
    if latest_completed and latest_failed:
        if latest_failed.temp_last_updated > latest_completed.last_updated:
            current_level = latest_failed.level
        else:
            current_level = latest_completed.level
    elif latest_failed:
        current_level = latest_failed.level
    elif latest_completed:
        current_level = latest_completed.level

    # Determine next level
    if latest_failed and (not latest_completed or latest_failed.temp_last_updated > latest_completed.last_updated):
        if latest_failed.status != "Completed":
            next_level = latest_failed.level
        else:
            next_level = latest_failed.level + 1
    else:
        next_level = (latest_completed.level + 1) if latest_completed and latest_completed.level < TOTAL_LEVELS else 1

    # Points summary
    points_summary = {p.level: p.score for p in progress_records if p.status == "Completed"}

    # New Logic: Next Unlocked Level
    highest_completed = max(completed_levels, default=0)
    next_unlocked_level = highest_completed + 1 if highest_completed < TOTAL_LEVELS else "All levels completed!"

    return {
        "username": username,
        "child_id": child_id,
        "completed_levels": completed_levels,
        "current_level": current_level,
        "next_level": next_level,
        "points_per_level": points_summary,
        "total_levels": TOTAL_LEVELS,
        "Next_Unlocked_Level": next_unlocked_level
    }


# FIXED REPLAY ENDPOINT *************************************************************************************************************
@app.get("//kids/v1/game/replay_level/spell/{child_id}", tags=["SPELL BREAKER"])
def replay_last_completed_level_spell(child_id: int, db: Session = Depends(get_db)):
    """
    Replay the last completed OR recently failed level automatically for SPELL BREAKER game.
    - Compares the most recent `status='Completed'` and `temp_status='Failed'`
    - Replays whichever happened last (based on timestamp)
    Example: /game/replay_level/spell/4
    """

    # 1. Get most recently failed (temp_status='Failed')
    last_failed = (
        db.query(GameProgress_SPELL)
        .filter(GameProgress_SPELL.child_id == child_id, GameProgress_SPELL.temp_status == "Failed")
        .order_by(GameProgress_SPELL.temp_last_updated.desc())
        .first()
    )

    # 2. Get most recently completed (status='Completed')
    last_completed = (
        db.query(GameProgress_SPELL)
        .filter(GameProgress_SPELL.child_id == child_id, GameProgress_SPELL.status == "Completed")
        .order_by(GameProgress_SPELL.last_updated.desc())
        .first()
    )

    # 3. Determine which one is latest based on timestamp comparison
    chosen_progress = None
    if last_failed and last_completed:
        if last_failed.temp_last_updated > last_completed.last_updated:
            chosen_progress = last_failed
        else:
            chosen_progress = last_completed
    elif last_failed:
        chosen_progress = last_failed
    elif last_completed:
        chosen_progress = last_completed
    else:
        raise HTTPException(status_code=400, detail="No failed or completed level found to replay.")

    last_level = chosen_progress.level  

    # 4. Initialize new session for that level
    session_id = str(uuid.uuid4())
    lives = get_lives_for_spellBreaker_level(last_level)  # Lives logic applied
    total_questions = questions_for_level(last_level)
    time_limit = time_for_level(last_level)
    difficulty = difficulty_for_level(last_level)
    question_number = 1

    # 5. Generate new puzzle
    puzzle = generate_jumbled_puzzle(difficulty)

    GAME_SESSIONS[session_id] = {
        "child_id": child_id,
        "level": last_level,
        "points": 0,
        "lives": lives,
        "question_idx": question_number,
        "total_questions": total_questions,
        "jumbled_word": puzzle["jumbled_word"],
        "options": puzzle["options"],
        "correct_option": puzzle["correct_option"],
        "fact": puzzle["fact"],
        "start_time": time.time(),
        "time_limit": time_limit,
    }
    # #  6. Response unchanged
    # return {
    #     "session_id": session_id,
    #     "level": last_level,
    #     "question_number": question_number,
    #     "total_questions_in_level": total_questions,
    #     "jumbled_word": puzzle["jumbled_word"],
    #     "options": puzzle["options"],
    #     "points": 0,
    #     "lives": lives,
    #     "time_left": time_limit,
    #     "reward_message": f"Replaying Level {last_level}! Let's see if you can beat your last performance!"
    # }
    return {
        "session_id": session_id,
        "puzzle_expression": puzzle["jumbled_word"],
        "options": puzzle["options"],
        "points": 0,
        "reward_message": f"Replaying Level {last_level}! Let's see if you can beat your last performance!",
        "lives": lives,
        "level": last_level,
        "question_number": question_number,
        "total_questions_in_level": total_questions,
        "time_left": time_limit
    }

# NEXT LEVEL ENDPOINT : *********************************************************************************************************
@app.get("//kids/v1/game/next_level/spell/{child_id}", tags=["SPELL BREAKER"])
def next_level_spell(child_id: int, db: Session = Depends(get_db)):
    """
    Move the child to the next unlocked level automatically after completing the previous one.
    Automatically starts the next level's session.
    Logic fix:
    @***If a child replays an older completed level, 
       and that becomes the latest (most recent) completion (based on last_updated),
       the next level should be after that recent one, not after the max completed level.
    ********Additionally, compare temp_failed vs completed times — whichever is latest decides next level.
    """

    TOTAL_LEVELS = 100

    # Step 1: Fetch all completed levels (latest first)
    completed_levels = (
        db.query(GameProgress_SPELL)
        .filter(GameProgress_SPELL.child_id == child_id, GameProgress_SPELL.status == "Completed")
        .order_by(GameProgress_SPELL.last_updated.desc())
        .all()
    )

    # Step 2: Fetch the latest failed temp record (if exists)
    latest_failed = (
        db.query(GameProgress_SPELL)
        .filter(GameProgress_SPELL.child_id == child_id, GameProgress_SPELL.temp_status == "Failed")
        .order_by(GameProgress_SPELL.temp_last_updated.desc())
        .first()
    )

    if not completed_levels and not latest_failed:
        raise HTTPException(status_code=404, detail="No progress found. Start from Level 1.")

    # Step 3: Identify which record (failed/completed) is latest
    recent_completed_time = completed_levels[0].last_updated if completed_levels else None
    recent_failed_time = latest_failed.temp_last_updated if latest_failed else None

    # Step 4: Decide which one is latest and pick level accordingly
    if recent_failed_time and (not recent_completed_time or recent_failed_time > recent_completed_time):
        # New logic: if latest failed level is not yet completed → don't move to next level
        failed_level_completed = (
            db.query(GameProgress_SPELL)
            .filter(
                GameProgress_SPELL.child_id == child_id,
                GameProgress_SPELL.level == latest_failed.level,
                GameProgress_SPELL.status == "Completed",
            )
            .first()
        )

        if not failed_level_completed:
            # Still failed, so repeat the same level
            recent_level = latest_failed.level
            next_level = latest_failed.level
        else:
            # If later completed, proceed normally
            recent_level = latest_failed.level
            next_level = recent_level + 1

    else:
        # The latest completed level is more recent (normal case)
        recent_level = completed_levels[0].level
        next_level = recent_level + 1

    # Step 5: Restrict to max level
    if next_level > TOTAL_LEVELS:
        next_level = TOTAL_LEVELS

    # Step 6: Prepare next level details
    total_q = questions_for_level(next_level)
    time_limit = time_for_level(next_level)
    difficulty = difficulty_for_level(next_level)
    lives = get_lives_for_spellBreaker_level(next_level)  # Lives logic applied: 1 for 1-15, 2 for 16-30, 3 for 31-50, 4 for 51-100

    puzzle = generate_jumbled_puzzle(difficulty)
    session_id = str(uuid.uuid4())

    # Step 7: Initialize new session
    GAME_SESSIONS[session_id] = {
        "child_id": child_id,
        "level": next_level,
        "points": 0,
        "lives": lives,
        "question_idx": 1,
        "total_questions": total_q,
        "jumbled_word": puzzle["jumbled_word"],
        "options": puzzle["options"],
        "correct_option": puzzle["correct_option"],
        "fact": puzzle["fact"],
        "start_time": time.time(),
        "time_limit": time_limit,
    }
    return {
        "session_id": session_id,
        "puzzle_expression": puzzle["jumbled_word"],
        "options": puzzle["options"],
        "points": 0,
        "reward_message": (
            f"Level {next_level} unlocked! Ready to spell some magic?"
            if next_level != recent_level
            else f"Retry Level {next_level}! Give it another try!"
        ),
        "lives": lives,
        "level": next_level,
        "question_number": 1,
        "total_questions_in_level": questions_for_level(next_level),
        "time_left": time_limit
    }

# -----------------------------------------------------------
# MINDMYSTERY GAME - MODELS
# -----------------------------------------------------------

class Animals(Base):
    __tablename__ = "animals"
    id = Column(Integer, primary_key=True, index=True)
    fact = Column(String)
    image = Column(String)
    question = Column(String)
    option_a = Column(String)
    option_b = Column(String)
    option_c = Column(String)
    option_d = Column(String)
    correct_option = Column(String)


class FunnyWhoami(Base):
    __tablename__ = "funny_whoami"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, nullable=False)
    option_a = Column(String, nullable=False)
    option_b = Column(String, nullable=False)
    option_c = Column(String, nullable=False)
    option_d = Column(String, nullable=False)
    correct_option = Column(String, nullable=False)
    fun_fact = Column(String, nullable=False)


Base.metadata.create_all(bind=engine)

# new logic ithe static folder mount karayache baki hote baki***********************************
BASE_DIR = Path(__file__).resolve().parent
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)


# -----------------------------------------------------------
# PYDANTIC MODELS
# -----------------------------------------------------------
class WhoAmIStateResponse(BaseModel):
    session_id: str
    level: int
    question_number: int
    total_questions_in_level: int
    question: str
    options: Dict[str, str]
    points: int
    reward_message: str
    lives: int
    time_left: int
    image: Optional[str] = None
    submitted_answer: str
    correct_answer: str
    fact: str


class WhoAmIAnswerRequest(BaseModel):
    session_id: str
    answer: str


# -----------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------
WHOAMI_TOTAL_LEVELS = 60
TIME_LIMIT_PER_QUESTION = 60
GAME_SESSIONS = {}

SLOTS = {
    "slot1": (1, 7),
    "slot2": (8, 21),
    "slot3": (22, 35),
    "slot4": (36, 49),
    "slot5": (50, 60),
}

QUESTIONS_PER_LEVEL_BY_SLOT = {
    # "slot1": 7,
    "slot1": 2,
    "slot2": 7,
    "slot3": 10,
    "slot4": 12,
    "slot5": 14,
}

FUNNY_RANGES = {
    "slot2": (1, 201),
    "slot3": (202, 734),
    "slot4": (735, 1385),
    "slot5": (1386, 1875),
}


# -----------------------------------------------------------
# CLEANERS / HELPERS
# -----------------------------------------------------------
def clean_opt(o: str) -> str:
    if not o:
        return ""
    o = o.strip()
    o = (
        o.replace("A) ", "")
        .replace("B) ", "")
        .replace("C) ", "")
        .replace("D) ", "")
        .replace("A. ", "")
        .replace("B. ", "")
        .replace("C. ", "")
        .replace("D. ", "")
        .replace("A)", "")
        .replace("B)", "")
        .replace("C)", "")
        .replace("D)", "")
    )
    return o.strip()


def clean_fact(text: str) -> str:
    if not text:
        return ""
    text = text.strip()

    # Remove emojis or prefixes before Fact:
    text = re.sub(r'^[^\w]*Fact[:\-]*\s*', '', text, flags=re.IGNORECASE)

    # Remove leading non-alphanumeric chars
    text = re.sub(r'^[^\w]+', '', text)

    return text.strip()


def calculate_age(dob: date) -> int:
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def get_age_slab(age: int) -> str:
    if 5 <= age <= 10:
        return "5-10"
    elif 11 <= age <= 15:
        return "11-15"
    return "other"

# -----------------------------------------------------------
# SEED GAME PROGRESS FOR CHILD
# -----------------------------------------------------------
def seed_game_progress_whoami(db: Session, child_id: int):
    existing = db.query(GameProgress_WHOAMI).filter(
        GameProgress_WHOAMI.child_id == child_id
    ).count()

    if existing >= WHOAMI_TOTAL_LEVELS:
        return

    new_rows = []
    for lvl in range(1, WHOAMI_TOTAL_LEVELS + 1):
        new_rows.append(
            GameProgress_WHOAMI(
                child_id=child_id,
                level=lvl,
                score=0,
                status="Not Started",
                last_updated=get_ist_now()
            )
        )
    db.add_all(new_rows)
    db.commit()


# -----------------------------------------------------------
# SLOT & LIVES
# -----------------------------------------------------------
def get_slot_for_level(level: int) -> str:
    for slot_name, (s, e) in SLOTS.items():
        if s <= level <= e:
            return slot_name
    raise ValueError("Invalid Level")


def get_lives_for_slot(slot: str) -> int:
    return {
        "slot1": 1,
        "slot2": 2,
        "slot3": 3,
        "slot4": 4,
        "slot5": 5,
    }.get(slot, 1)


# # -----------------------------------------------------------
# # FETCH QUESTIONS FOR A LEVEL
# # -----------------------------------------------------------
def fetch_questions_for_level(db: Session, level: int, child_id: int = None):
    slot = get_slot_for_level(level)
    qcount = QUESTIONS_PER_LEVEL_BY_SLOT[slot]

    age_slab = "other"
    if child_id:
        child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
        if child and child.dob:
            age_slab = get_age_slab(calculate_age(child.dob))

    # ------------------ SLOT 1 (Animals) -----------------------
    if slot == "slot1":
        if age_slab == "5-10":
            min_id = 1
            max_id = 53
            rows = (
                db.query(Animals)
                .filter(Animals.id >= min_id, Animals.id <= max_id)
                .order_by(Animals.id)
                .all()
            )
            start_index = (level - 1) * qcount
            rows = rows[start_index:start_index + qcount]

        elif age_slab == "11-15":
            candidate_rows = (
                db.query(Animals)
                .filter(Animals.id >= 54, Animals.id <= 198)
                .all()
            )
            rows = random.sample(candidate_rows, qcount)

        else:
            start_id = (level - 1) * qcount + 1
            end_id = level * qcount
            rows = (
                db.query(Animals)
                .filter(Animals.id >= start_id, Animals.id <= end_id)
                .order_by(Animals.id)
                .all()
            )

        questions = []
        for r in rows:
            img_val = r.image
            if img_val and not img_val.startswith("/"):
                final_img = img_val
            else:
                final_img = f"https://semantic.onesmarter.com{img_val}" if img_val else None

            if age_slab == "11-15" and not final_img:
                final_img = "NA"

            questions.append({
                "id": r.id,
                "question": r.question,
                "option_a": clean_opt(r.option_a),
                "option_b": clean_opt(r.option_b),
                "option_c": clean_opt(r.option_c),
                "option_d": clean_opt(r.option_d),
                "image": final_img
            })

        return questions, qcount

    # ------------------ SLOT 2–5 (Funny WhoAmI) -----------------------
    low, high = FUNNY_RANGES[slot]
    ids = [
        row[0]
        for row in db.query(FunnyWhoami.id)
        .filter(FunnyWhoami.id >= low, FunnyWhoami.id <= high)
        .all()
    ]

    chosen = random.sample(ids, qcount)
    rows = db.query(FunnyWhoami).filter(FunnyWhoami.id.in_(chosen)).all()
    idmap = {r.id: r for r in rows}

    questions = []
    for cid in chosen:
        r = idmap[cid]
        questions.append({
            "id": r.id,
            "question": r.question,
            "option_a": clean_opt(r.option_a),
            "option_b": clean_opt(r.option_b),
            "option_c": clean_opt(r.option_c),
            "option_d": clean_opt(r.option_d),
            "image": None
        })

    return questions, qcount

    
 
# -----------------------------------------------------------
# WELCOME ENDPOINT
# -----------------------------------------------------------
@app.get("//kids/v1/welcome/mind/{child_id}", tags=["MIND MYSTERY"])
def welcome_player(child_id: int, db: Session = Depends(get_db)):
    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(404, "Child not found")

    progress = (
        db.query(GameProgress_WHOAMI)
        .filter(GameProgress_WHOAMI.child_id == child_id,
                GameProgress_WHOAMI.status == "Completed")
        .order_by(GameProgress_WHOAMI.level.desc())
        .first()
    )

    next_level = (progress.level + 1) if progress else 1
    if next_level > 60:
        next_level = 60

    msgs = [
        f"Welcome, {child.username}! Ready for Level {next_level}?",
        f"Yo {child.username}, Level {next_level} is waiting!",
        f"Let’s go {child.username}! Level {next_level} unlocked!"
    ]

    return {"message": random.choice(msgs)}


# -----------------------------------------------------------
# START LEVEL ENDPOINT
# -----------------------------------------------------------
# START LEVEL
# -----------------------------------------------------------

@app.get("//kids/v1/mind-mystery/Start", response_model=WhoAmIStateResponse, tags=["MIND MYSTERY"])
def start_level(session_id: str = None, child_id: int = None, db: Session = Depends(get_db)):
# def start_level(child_id: int = None, db: Session = Depends(get_db)):

    # Resume Session
    if session_id and session_id in GAME_SESSIONS:
        sess = GAME_SESSIONS[session_id]
        qidx = sess["question_number"] - 1
        q = sess["questions"][qidx].copy()
        q.pop("id", None)

        elapsed = int(time.time() - sess["start_time"])
        time_left = max(0, TIME_LIMIT_PER_QUESTION - elapsed)

        return WhoAmIStateResponse(
            session_id=session_id,
            level=sess["level"],
            question_number=sess["question_number"],
            total_questions_in_level=sess["total_questions"],
            question=q["question"],
            options={
                "a": q["option_a"],
                "b": q["option_b"],
                "c": q["option_c"],
                "d": q["option_d"],
            },
            points=sess["points"],
            reward_message="Level resumed.",
            lives=sess["lives"],
            time_left=time_left,
            image=q.get("image"),
            submitted_answer="",
            correct_answer="",
            fact=""
        )

    # Start new
    if child_id:
        seed_game_progress_whoami(db, child_id)

        completed = db.query(GameProgress_WHOAMI).filter(
            GameProgress_WHOAMI.child_id == child_id,
            GameProgress_WHOAMI.status == "Completed"
        ).all()

        next_level = (max([c.level for c in completed]) + 1) if completed else 1
        next_level = min(next_level, 60)
        level = next_level
    else:
        level = 1

    # ⭐ Mark level as STARTED in DB
    if child_id:
        rec = db.query(GameProgress_WHOAMI).filter(
            GameProgress_WHOAMI.child_id == child_id,
            GameProgress_WHOAMI.level == level
        ).first()

        if rec:
            rec.status = "Started"
            rec.temp_status = None
            rec.last_updated = get_ist_now()
            rec.temp_last_updated = get_ist_now()
            db.commit()

    # Continue
    questions, total = fetch_questions_for_level(db, level, child_id)

    slot = get_slot_for_level(level)
    lives = get_lives_for_slot(slot)

    new_session = str(uuid.uuid4())
    GAME_SESSIONS[new_session] = {
        "level": level,
        "points": 0,
        "lives": lives,
        "questions": questions,
        "question_number": 1,
        "total_questions": total,
        "start_time": time.time(),
        "child_id": child_id
    }

    q = questions[0].copy()
    q.pop("id", None)

    return WhoAmIStateResponse(
        session_id=new_session,
        level=level,
        question_number=1,
        total_questions_in_level=total,
        question=q["question"],
        options={
            "a": q["option_a"],
            "b": q["option_b"],
            "c": q["option_c"],
            "d": q["option_d"]
        },
        points=0,
        reward_message=f"Level {level} Started!",
        lives=lives,
        time_left=60,
        image=q.get("image"),
        submitted_answer="",
        correct_answer="",
        fact=""
    )



# -----------------------------------------------------------
# SUBMIT
# -----------------------------------------------------------
@app.post("//kids/v1/mind-mystery/Submit", response_model=WhoAmIStateResponse, tags=["MIND MYSTERY"])
def submit_answer(req: WhoAmIAnswerRequest, child_id: int, db: Session = Depends(get_db)):

    # ---------------------- SESSION VALIDATION ----------------------
    if req.session_id not in GAME_SESSIONS:
        raise HTTPException(400, "Invalid Session")

    sess = GAME_SESSIONS[req.session_id]
    level = sess["level"]
    q_idx = sess["question_number"] - 1
    child_id = sess["child_id"]

    # Fix index overflow
    if q_idx < 0 or q_idx >= len(sess["questions"]):
        q_idx = len(sess["questions"]) - 1

    qdata = sess["questions"][q_idx]        # FULL VERSION with id

    # ---------------------- DB LOOKUP ----------------------
    q_id = qdata.get("id")
    if not q_id:
        raise HTTPException(500, f"Question ID missing in session. qdata={qdata}")

    model = Animals if level <= 7 else FunnyWhoami
    row = db.query(model).filter(model.id == q_id).first()

    if not row:
        raise HTTPException(500, f"No DB record found for question id={q_id}")

    if not row.correct_option:
        raise HTTPException(500, f"correct_option is NULL for id={q_id}")

    correct_opt = row.correct_option.lower().strip()

    raw_fact = getattr(row, "fact", None) if level <= 7 else getattr(row, "fun_fact", None)
    fact_text = clean_fact(raw_fact) or "Keep learning something new every day!"

    # ---------------------- TIMEOUT HANDLING ----------------------
    elapsed = int(time.time() - sess["start_time"])
    if elapsed >= TIME_LIMIT_PER_QUESTION:

        sess["lives"] -= 1

        # ----- Game Over on Timeout -----
        if sess["lives"] <= 0:

            rec = db.query(GameProgress_WHOAMI).filter(
                GameProgress_WHOAMI.child_id == child_id,
                GameProgress_WHOAMI.level == level
            ).first()

            if rec:
                rec.temp_status = "Failed"
                rec.temp_last_updated = get_ist_now()
                rec.status = "Started"
                db.commit()

            ui_q = qdata.copy()
            ui_q.pop("id", None)

            return WhoAmIStateResponse(
                session_id=req.session_id,
                level=level,
                question_number=sess["question_number"],
                total_questions_in_level=sess["total_questions"],
                question=ui_q["question"],
                options={
                    "a": ui_q["option_a"],
                    "b": ui_q["option_b"],
                    "c": ui_q["option_c"],
                    "d": ui_q["option_d"]
                },
                points=sess["points"],
                reward_message="Time Out! Game Over.",
                lives=0,
                time_left=0,
                image=ui_q.get("image"),
                submitted_answer=req.answer,
                correct_answer=correct_opt,
                fact=fact_text
            )

        # ----- Timeout but continue -----
        new_qs, _ = fetch_questions_for_level(db, level, child_id)

        new_raw = random.choice(new_qs)   # full version WITH id
        new_display = new_raw.copy()      # UI version
        new_display.pop("id", None)

        sess["questions"][q_idx] = new_raw     # store full version with id
        sess["start_time"] = time.time()

        return WhoAmIStateResponse(
            session_id=req.session_id,
            level=level,
            question_number=sess["question_number"],
            total_questions_in_level=sess["total_questions"],
            question=new_display["question"],
            options={
                "a": new_display["option_a"],
                "b": new_display["option_b"],
                "c": new_display["option_c"],
                "d": new_display["option_d"]
            },
            points=sess["points"],
            reward_message="Timeout! Life lost.",
            lives=sess["lives"],
            time_left=60,
            image=new_display.get("image"),
            submitted_answer=req.answer,
            correct_answer=correct_opt,
            fact=fact_text
        )

    # ---------------------- NORMAL ANSWER CHECK ----------------------
    correct = (req.answer.lower().strip() == correct_opt)

    # ---------------------------------------------------------
    #                 CORRECT ANSWER
    # ---------------------------------------------------------
    if correct:

        sess["points"] += 10
        sess["question_number"] += 1

        # ----- Level Completed -----
        if sess["question_number"] > sess["total_questions"]:

            rec = db.query(GameProgress_WHOAMI).filter(
                GameProgress_WHOAMI.child_id == child_id,
                GameProgress_WHOAMI.level == level
            ).first()

            if rec:
                rec.status = "Completed"
                rec.score = sess["points"]
                rec.temp_status = None
                rec.last_updated = get_ist_now()
                db.commit()

            ui_q = qdata.copy()
            ui_q.pop("id", None)

            return WhoAmIStateResponse(
                session_id=req.session_id,
                level=level,
                question_number=sess["total_questions"],
                total_questions_in_level=sess["total_questions"],
                question=ui_q["question"],
                options={
                    "a": ui_q["option_a"],
                    "b": ui_q["option_b"],
                    "c": ui_q["option_c"],
                    "d": ui_q["option_d"]
                },
                points=sess["points"],
                reward_message=f"Level {level} Completed! You can start the next level manually.",
                lives=sess["lives"],
                time_left=0,
                image=ui_q.get("image"),
                submitted_answer=req.answer,
                correct_answer=correct_opt,
                fact=fact_text
            )

        # ----- Next Question -----
        next_raw = sess["questions"][sess["question_number"] - 1]
        next_display = next_raw.copy()
        next_display.pop("id", None)

        sess["start_time"] = time.time()

        return WhoAmIStateResponse(
            session_id=req.session_id,
            level=level,
            question_number=sess["question_number"],
            total_questions_in_level=sess["total_questions"],
            question=next_display["question"],
            options={
                "a": next_display["option_a"],
                "b": next_display["option_b"],
                "c": next_display["option_c"],
                "d": next_display["option_d"]
            },
            points=sess["points"],
            reward_message="Correct! +10 points",
            lives=sess["lives"],
            time_left=60,
            image=next_display.get("image"),
            submitted_answer=req.answer,
            correct_answer=correct_opt,
            fact=fact_text
        )

    # ---------------------------------------------------------
    #                 WRONG ANSWER
    # ---------------------------------------------------------
    else:

        sess["lives"] -= 1

        # ----- Game Over -----
        if sess["lives"] <= 0:

            rec = db.query(GameProgress_WHOAMI).filter(
                GameProgress_WHOAMI.child_id == child_id,
                GameProgress_WHOAMI.level == level
            ).first()

            if rec:
                rec.temp_status = "Failed"
                rec.temp_last_updated = get_ist_now()
                rec.status = "Started"
                db.commit()

            ui_q = qdata.copy()
            ui_q.pop("id", None)

            return WhoAmIStateResponse(
                session_id=req.session_id,
                level=level,
                question_number=sess["question_number"],
                total_questions_in_level=sess["total_questions"],
                question=ui_q["question"],
                options={
                    "a": ui_q["option_a"],
                    "b": ui_q["option_b"],
                    "c": ui_q["option_c"],
                    "d": ui_q["option_d"]
                },
                points=sess["points"],
                reward_message="Wrong! Game Over.",
                lives=0,
                time_left=0,
                image=ui_q.get("image"),
                submitted_answer=req.answer,
                correct_answer=correct_opt,
                fact=fact_text
            )

        # ----- Wrong but continue -----
        new_qs, _ = fetch_questions_for_level(db, level, child_id)

        new_raw = random.choice(new_qs)
        new_display = new_raw.copy()
        new_display.pop("id", None)

        sess["questions"][q_idx] = new_raw
        sess["start_time"] = time.time()

        return WhoAmIStateResponse(
            session_id=req.session_id,
            level=level,
            question_number=sess["question_number"],
            total_questions_in_level=sess["total_questions"],
            question=new_display["question"],
            options={
                "a": new_display["option_a"],
                "b": new_display["option_b"],
                "c": new_display["option_c"],
                "d": new_display["option_d"]
            },
            points=sess["points"],
            reward_message=f"Wrong! Correct was {correct_opt}. New question!",
            lives=sess["lives"],
            time_left=60,
            image=new_display.get("image"),
            submitted_answer=req.answer,
            correct_answer=correct_opt,
            fact=fact_text
        )


# -----------------------------------------------------------
# PROGRESS
# -----------------------------------------------------------
@app.get("//kids/v1/mind-mystery/Progress", tags =["MIND MYSTERY"])
def progress(child_id: int, db: Session = Depends(get_db)):

    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(404, "Child not found")

    recs = db.query(GameProgress_WHOAMI).filter(
        GameProgress_WHOAMI.child_id == child_id
    ).order_by(GameProgress_WHOAMI.level).all()

    if not recs:
        return {
            "username": child.username,
            "child_id": child_id,
            "completed_levels": 0,
            "current_level": 1,
            "next_level": 1,
            "total_levels": 60
        }

    completed = [r.level for r in recs if r.status == "Completed"]

    first_not = next((r.level for r in recs if r.status != "Completed"), 60)

    current_level = first_not
    next_level = current_level + 1 if current_level < 60 else None

    return {
        "username": child.username,
        "child_id": child_id,
        "completed_levels": len(completed),
        "current_level": current_level,
        "next_level": next_level,
        "total_levels": 60
    }


# -----------------------------------------------------------
# SELECT
# -----------------------------------------------------------
@app.get("//kids/v1/mind-mystery/Select_level", response_model=WhoAmIStateResponse, tags=["MIND MYSTERY"])
def select_level(child_id: int, selected_level: int, db: Session = Depends(get_db)):

    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(404, "Child not found")

    if not 1 <= selected_level <= 60:
        raise HTTPException(400, "Invalid level number")

    seed_game_progress_whoami(db, child_id)

    if selected_level > 1:
        prev = db.query(GameProgress_WHOAMI).filter(
            GameProgress_WHOAMI.child_id == child_id,
            GameProgress_WHOAMI.level == selected_level - 1
        ).first()

        if not prev or prev.status != "Completed":
            last_completed = db.query(GameProgress_WHOAMI).filter(
                GameProgress_WHOAMI.child_id == child_id,
                GameProgress_WHOAMI.status == "Completed"
            ).order_by(GameProgress_WHOAMI.level.desc()).first()

            allowed = last_completed.level if last_completed else 1

            raise HTTPException(403, f"Level {selected_level} is locked. Highest unlocked: {allowed}")

    questions, total_questions = fetch_questions_for_level(db, selected_level, child_id)
    slot = get_slot_for_level(selected_level)
    lives = get_lives_for_slot(slot)

    session_id = str(uuid.uuid4())
    GAME_SESSIONS[session_id] = {
        "level": selected_level,
        "points": 0,
        "lives": lives,
        "questions": questions,
        "question_number": 1,
        "total_questions": total_questions,
        "start_time": time.time(),
        "child_id": child_id
    }

    q = questions[0].copy()
    q.pop("id", None)

    return WhoAmIStateResponse(
        session_id=session_id,
        level=selected_level,
        question_number=1,
        total_questions_in_level=total_questions,
        question=q["question"],
        options={
            "a": q["option_a"],
            "b": q["option_b"],
            "c": q["option_c"],
            "d": q["option_d"]
        },
        points=0,
        reward_message=f"Level {selected_level} Started!",
        lives=lives,
        time_left=60,
        image=q.get("image"),
        submitted_answer="",
        correct_answer="",
        fact=""
    )


# -----------------------------------------------------------
# REPLAY
# -----------------------------------------------------------
@app.get("//kids/v1/mind-mystery/Replay", response_model=WhoAmIStateResponse, tags=["MIND MYSTERY"])
def replay_last_level(child_id: int, db: Session = Depends(get_db)):

    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(404, "Child not found")

    last_failed = db.query(GameProgress_WHOAMI).filter(
        GameProgress_WHOAMI.child_id == child_id,
        GameProgress_WHOAMI.temp_status == "Failed"
    ).order_by(GameProgress_WHOAMI.temp_last_updated.desc()).first()

    last_completed = db.query(GameProgress_WHOAMI).filter(
        GameProgress_WHOAMI.child_id == child_id,
        GameProgress_WHOAMI.status == "Completed"
    ).order_by(GameProgress_WHOAMI.last_updated.desc()).first()

    if last_failed:
        level_to_replay = last_failed.level
    elif last_completed:
        level_to_replay = last_completed.level
    else:
        raise HTTPException(404, "No previous level available to replay")

    questions, total_questions = fetch_questions_for_level(db, level_to_replay, child_id)

    slot = get_slot_for_level(level_to_replay)
    lives = get_lives_for_slot(slot)

    session_id = str(uuid.uuid4())
    GAME_SESSIONS[session_id] = {
        "level": level_to_replay,
        "points": 0,
        "lives": lives,
        "questions": questions,
        "question_number": 1,
        "total_questions": total_questions,
        "start_time": time.time(),
        "child_id": child_id
    }

    q = questions[0].copy()
    q.pop("id", None)

    return WhoAmIStateResponse(
        session_id=session_id,
        level=level_to_replay,
        question_number=1,
        total_questions_in_level=total_questions,
        question=q["question"],
        options={
            "a": q["option_a"],
            "b": q["option_b"],
            "c": q["option_c"],
            "d": q["option_d"]
        },
        points=0,
        reward_message=f"Replaying Level {level_to_replay}",
        lives=lives,
        time_left=60,
        image=q.get("image"),
        submitted_answer="",
        correct_answer="",
        fact=""
    )


# -----------------------------------------------------------
# NEXT LEVEL
# -----------------------------------------------------------


# @app.get("//kids/v1/mind-mystery/NextLevel", response_model=WhoAmIStateResponse, tags=["MIND MYSTERY"])
# def next_level(child_id: int, db: Session = Depends(get_db)):

#     child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
#     if not child:
#         raise HTTPException(404, "Child not found")

#     completed = db.query(GameProgress_WHOAMI).filter(
#         GameProgress_WHOAMI.child_id == child_id,
#         GameProgress_WHOAMI.status == "Completed"
#     ).order_by(GameProgress_WHOAMI.level).all()

#     current_level = completed[-1].level if completed else 1
#     next_level = current_level + 1

#     if next_level > 60:
#         raise HTTPException(400, "All levels completed!")

#     questions, total_questions = fetch_questions_for_level(db, next_level, child_id)
#     slot = get_slot_for_level(next_level)
#     lives = get_lives_for_slot(slot)

#     session_id = str(uuid.uuid4())
#     GAME_SESSIONS[session_id] = {
#         "level": next_level,
#         "points": 0,
#         "lives": lives,
#         "questions": questions,
#         "question_number": 1,
#         "total_questions": total_questions,
#         "start_time": time.time(),
#         "child_id": child_id
#     }

#     q = questions[0].copy()
#     q.pop("id", None)

#     return WhoAmIStateResponse(
#         session_id=session_id,
#         level=next_level,
#         question_number=1,
#         total_questions_in_level=total_questions,
#         question=q["question"],
#         options={
#             "a": q["option_a"],
#             "b": q["option_b"],
#             "c": q["option_c"],
#             "d": q["option_d"]
#         },
#         points=0,
#         reward_message=f"Level {next_level} Started!",
#         lives=lives,
#         time_left=60,
#         image=q.get("image"),
#         submitted_answer="",
#         correct_answer="",
#         fact=""
#     )



#new logic next level : ************************************************************************************************
# 
@app.get("//kids/v1/mind-mystery/NextLevel", response_model=WhoAmIStateResponse, tags=["MIND MYSTERY"])
def next_level(child_id: int, db: Session = Depends(get_db)):

    child = db.query(ChildDB).filter(ChildDB.id == child_id).first()
    if not child:
        raise HTTPException(404, "Child not found")

    TOTAL_LEVELS = 60

    # Step 1: Fetch all completed levels (latest first)
    completed_levels = (
        db.query(GameProgress_WHOAMI)
        .filter(GameProgress_WHOAMI.child_id == child_id,
                GameProgress_WHOAMI.status == "Completed")
        .order_by(GameProgress_WHOAMI.last_updated.desc())
        .all()
    )

    # Step 2: Fetch latest failed temp record
    latest_failed = (
        db.query(GameProgress_WHOAMI)
        .filter(GameProgress_WHOAMI.child_id == child_id,
                GameProgress_WHOAMI.temp_status == "Failed")
        .order_by(GameProgress_WHOAMI.temp_last_updated.desc())
        .first()
    )

    if not completed_levels and not latest_failed:
        recent_level = 1
        next_level = 1
    else:
        # Step 3: Compare recent failed vs recent completed
        recent_completed_time = completed_levels[0].last_updated if completed_levels else None
        recent_failed_time = latest_failed.temp_last_updated if latest_failed else None

        if recent_failed_time and (not recent_completed_time or recent_failed_time > recent_completed_time):
            failed_level_completed = (
                db.query(GameProgress_WHOAMI)
                .filter(
                    GameProgress_WHOAMI.child_id == child_id,
                    GameProgress_WHOAMI.level == latest_failed.level,
                    GameProgress_WHOAMI.status == "Completed"
                )
                .first()
            )

            if not failed_level_completed:
                recent_level = latest_failed.level
                next_level = latest_failed.level
            else:
                recent_level = latest_failed.level
                next_level = recent_level + 1
        else:
            recent_level = completed_levels[0].level
            next_level = recent_level + 1

    if next_level > TOTAL_LEVELS:
        raise HTTPException(400, "All levels completed!")


    questions, total_questions = fetch_questions_for_level(db, next_level, child_id)
    slot = get_slot_for_level(next_level)
    lives = get_lives_for_slot(slot)

    session_id = str(uuid.uuid4())
    GAME_SESSIONS[session_id] = {
        "level": next_level,
        "points": 0,
        "lives": lives,
        "questions": questions,
        "question_number": 1,
        "total_questions": total_questions,
        "start_time": time.time(),
        "child_id": child_id
    }

    q = questions[0].copy()
    q.pop("id", None)

    return WhoAmIStateResponse(
        session_id=session_id,
        level=next_level,
        question_number=1,
        total_questions_in_level=total_questions,
        question=q["question"],
        options={
            "a": q["option_a"],
            "b": q["option_b"],
            "c": q["option_c"],
            "d": q["option_d"]
        },
        points=0,
        reward_message=f"Level {next_level} Started!",
        lives=lives,
        time_left=60,
        image=q.get("image"),
        submitted_answer="",
        correct_answer="",
        fact=""
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
