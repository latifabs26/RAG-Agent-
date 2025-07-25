import argparse
import os
import shutil
from pathlib import Path
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Supported file extensions and their corresponding loaders
SUPPORTED_EXTENSIONS = {
    '.pdf': 'pdf',
    '.csv': 'csv',
    '.md': 'text',
    '.markdown': 'text',
    '.txt': 'text',
}

def main():
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    if documents:
        chunks = split_documents(documents)
        add_to_chroma(chunks)
    else:
        print("❌ No documents found to process")

def get_files_by_type(directory_path):
    """Get all supported files grouped by type"""
    files_by_type = {
        'pdf': [],
        'csv': [],
        'text': [],
    }
    
    data_path = Path(directory_path)
    if not data_path.exists():
        print(f"❌ Directory {directory_path} does not exist")
        return files_by_type
    
    for file_path in data_path.rglob('*'):
        if file_path.is_file():
            extension = file_path.suffix.lower()
            if extension in SUPPORTED_EXTENSIONS:
                file_type = SUPPORTED_EXTENSIONS[extension]
                files_by_type[file_type].append(str(file_path))
    
    return files_by_type

def load_single_csv(file_path):
    """Load a single CSV file with error handling"""
    try:
        # Try with default settings first
        loader = CSVLoader(file_path=file_path)
        return loader.load()
    except Exception as e1:
        try:
            # Try with different encoding
            loader = CSVLoader(
                file_path=file_path,
                encoding='utf-8',
                csv_args={'delimiter': ','}
            )
            return loader.load()
        except Exception as e2:
            try:
                # Try with semicolon delimiter (common in European CSVs)
                loader = CSVLoader(
                    file_path=file_path,
                    encoding='utf-8',
                    csv_args={'delimiter': ';'}
                )
                return loader.load()
            except Exception as e3:
                print(f"❌ Failed to load CSV {file_path}:")
                print(f"   Error 1: {e1}")
                print(f"   Error 2: {e2}")
                print(f"   Error 3: {e3}")
                return []

def load_single_text(file_path):
    """Load a single text file with encoding handling"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            loader = TextLoader(file_path, encoding=encoding)
            return loader.load()
        except Exception as e:
            continue
    
    print(f"❌ Failed to load text file {file_path} with any encoding")
    return []

def load_documents():
    """Load documents from various file types"""
    all_documents = []
    files_by_type = get_files_by_type(DATA_PATH)
    
    # Load PDF files using the original method
    if files_by_type['pdf']:
        print(f"📄 Loading {len(files_by_type['pdf'])} PDF files...")
        try:
            pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
            pdf_docs = pdf_loader.load()
            all_documents.extend(pdf_docs)
            print(f"✅ Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            print(f"❌ Error loading PDF files: {e}")
    
    # Load CSV files individually
    if files_by_type['csv']:
        print(f"📊 Loading {len(files_by_type['csv'])} CSV files...")
        for csv_file in files_by_type['csv']:
            csv_docs = load_single_csv(csv_file)
            if csv_docs:
                all_documents.extend(csv_docs)
                print(f"✅ Loaded CSV file: {Path(csv_file).name} ({len(csv_docs)} documents)")
    
    # Load text files individually
    if files_by_type['text']:
        print(f"📝 Loading {len(files_by_type['text'])} text files...")
        for text_file in files_by_type['text']:
            text_docs = load_single_text(text_file)
            if text_docs:
                all_documents.extend(text_docs)
                print(f"✅ Loaded text file: {Path(text_file).name}")
    
    print(f"📚 Total documents loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents: list[Document]):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    
    # For CSV files, use larger chunks since they contain structured data
    csv_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    
    for doc in documents:
        source = doc.metadata.get("source", "")
        
        try:
            # Use different splitter for CSV files
            if source.lower().endswith('.csv'):
                chunks = csv_splitter.split_documents([doc])
            else:
                chunks = text_splitter.split_documents([doc])
            
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"❌ Error splitting document {source}: {e}")
    
    print(f"🔪 Split into {len(all_chunks)} chunks")
    return all_chunks

def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )

    # Calculate chunk IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    """Calculate unique IDs for chunks"""
    last_source_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        
        # For PDF files, use page number
        if source.lower().endswith('.pdf'):
            page = chunk.metadata.get("page", 0)
            current_source_id = f"{source}:{page}"
        # For CSV files, use row number if available
        elif source.lower().endswith('.csv'):
            row = chunk.metadata.get("row", 0)
            current_source_id = f"{source}:{row}"
        # For other files, just use the source
        else:
            current_source_id = f"{source}:0"

        # If the source ID is the same as the last one, increment the index.
        if current_source_id == last_source_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_source_id}:{current_chunk_index}"
        last_source_id = current_source_id

        # Add it to the metadata.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()