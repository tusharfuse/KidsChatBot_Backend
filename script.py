import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import os

# Database setup (copied from your main.py for SQLite)
SQLALCHEMY_DATABASE_URL = "sqlite:///./kids_chatbot.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# Define the models (copied from your main.py)
Base = declarative_base()

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

# Create the tables if they don't exist
Base.metadata.create_all(bind=engine)

# Create a session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def load_csv_to_animals(csv_file_path: str):
    """
    Load data from CSV into the animals table.
    Assumes CSV columns: fact, image, question, option_a, option_b, option_c, option_d, correct_option.
    Skips rows with missing required data and handles duplicates gracefully.
    """
    try:
        # Read CSV into a DataFrame
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} rows from {csv_file_path}")
        
        # Validate required columns (all fields except id)
        required_columns = ['fact', 'image', 'question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        # Create a session
        db = SessionLocal()
        
        # Prepare data for bulk insert
        animals_data = []
        skipped_rows = 0
        for index, row in df.iterrows():
            # Skip rows with missing critical data (e.g., empty question or correct_option)
            if pd.isna(row['question']) or pd.isna(row['correct_option']):
                print(f"Skipped row {index + 1}: Missing question or correct_option")
                skipped_rows += 1
                continue
            
            animal = {
                'fact': str(row['fact']).strip() if pd.notna(row['fact']) else None,
                'image': str(row['image']).strip() if pd.notna(row['image']) else None,
                'question': str(row['question']).strip(),
                'option_a': str(row['option_a']).strip() if pd.notna(row['option_a']) else None,
                'option_b': str(row['option_b']).strip() if pd.notna(row['option_b']) else None,
                'option_c': str(row['option_c']).strip() if pd.notna(row['option_c']) else None,
                'option_d': str(row['option_d']).strip() if pd.notna(row['option_d']) else None,
                'correct_option': str(row['correct_option']).strip(),
            }
            animals_data.append(animal)
        
        # Bulk insert (efficient for large CSVs)
        if animals_data:
            db.bulk_insert_mappings(Animals, animals_data)
            db.commit()
            print(f"Successfully inserted {len(animals_data)} animals into the database. Skipped {skipped_rows} invalid rows.")
        else:
            print("No valid data to insert.")
    
    except IntegrityError as e:
        print(f"Integrity error (e.g., duplicate data): {e}. Rolling back.")
        db.rollback()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        db.rollback()
    finally:
        db.close()

def load_csv_to_funny_whoami(csv_file_path: str):
    """
    Load data from CSV into the funny_whoami table.
    Assumes CSV columns: question, option_a, option_b, option_c, option_d, correct_option, fun_fact.
    Skips rows with missing required data and handles duplicates gracefully.
    """
    try:
        # Read CSV into a DataFrame
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} rows from {csv_file_path}")
        
        # Validate required columns (all fields except id)
        required_columns = ['question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option', 'fun_fact']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        # Create a session
        db = SessionLocal()
        
        # Prepare data for bulk insert
        funny_whoami_data = []
        skipped_rows = 0
        for index, row in df.iterrows():
            # Skip rows with missing critical data (e.g., empty question or correct_option)
            if pd.isna(row['question']) or pd.isna(row['correct_option']) or pd.isna(row['fun_fact']):
                print(f"Skipped row {index + 1}: Missing question, correct_option, or fun_fact")
                skipped_rows += 1
                continue
            
            funny_item = {
                'question': str(row['question']).strip(),
                'option_a': str(row['option_a']).strip(),
                'option_b': str(row['option_b']).strip(),
                'option_c': str(row['option_c']).strip(),
                'option_d': str(row['option_d']).strip(),
                'correct_option': str(row['correct_option']).strip(),
                'fun_fact': str(row['fun_fact']).strip(),
            }
            funny_whoami_data.append(funny_item)
        
        # Bulk insert (efficient for large CSVs)
        if funny_whoami_data:
            db.bulk_insert_mappings(FunnyWhoami, funny_whoami_data)
            db.commit()
            print(f"Successfully inserted {len(funny_whoami_data)} funny_whoami entries into the database. Skipped {skipped_rows} invalid rows.")
        else:
            print("No valid data to insert.")
    
    except IntegrityError as e:
        print(f"Integrity error (e.g., duplicate data): {e}. Rolling back.")
        db.rollback()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # Paths to your CSV files
    animals_csv = "data-1762339147907.csv"  # For animals table
    funny_whoami_csv = "data-1762339583691.csv"  # For funny_whoami table
    
    # Load animals data
    if os.path.exists(animals_csv):
        load_csv_to_animals(animals_csv)
    else:
        print(f"Error: CSV file '{animals_csv}' not found.")
    
    # Load funny_whoami data
    if os.path.exists(funny_whoami_csv):
        load_csv_to_funny_whoami(funny_whoami_csv)
    else:
        print(f"Error: CSV file '{funny_whoami_csv}' not found.")