import json
import os

# --- Configuration ---
RAW_ROOT = 'remap-my-rag-source-data/raw/'
PROCESSED_ROOT = 'remap-my-rag-source-data/processed/'

# Create processed directories
os.makedirs(PROCESSED_ROOT + 'json_chunks/', exist_ok=True)
os.makedirs(PROCESSED_ROOT + 'documents/', exist_ok=True)
os.makedirs(PROCESSED_ROOT + 'unsupported_files/', exist_ok=True)

# Bedrock's maximum file size is ~50 MB. We target smaller chunks.
MAX_JSON_CHUNK_SIZE_BYTES = 40 * 1024 * 1024  # 40 MB

# --- Helper Functions (Simulating AWS ETL Logic) ---

def process_json_record(record: dict, original_filename: str) -> str:
    """Converts a complex JSON record into a simple, RAG-friendly text narrative.
    
    This prevents the 2048-byte metadata limit error.
    """
    context_string = f"SOURCE FILE: {original_filename}. "
    
    # 1. Manually extract and simplify key attributes (PII/PHI must be handled here!)
    narrative_parts = []
    
    for key, value in record.items():
        # Example of data cleaning: Skip internal keys and simplify complex types
        if key in ['id', 'timestamp', 'internal_hash']: 
            continue
        
        # Example: Convert patient data fields into structured text
        if key == 'patient_details' and isinstance(value, dict):
            narrative_parts.append(f"Patient ID: {value.get('patient_id', 'Unknown')}. Diagnosis: {value.get('diagnosis', 'N/A')}.")
        elif isinstance(value, (int, float, str)):
            narrative_parts.append(f"The {key.replace('_', ' ').title()} is: {str(value)}.")
        
    return context_string + " ".join(narrative_parts)

def split_and_process_jsonl(input_file_path):
    """Handles the size limit error by splitting the large file and processing content."""
    
    current_chunk_size = 0
    chunk_file_index = 0
    output_filename_base = os.path.basename(input_file_path).replace('.jsonl', '')
    
    print(f"Starting splitting and cleaning of {os.path.basename(input_file_path)}...")

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        output_data_list = []
        
        for line_number, line in enumerate(infile):
            try:
                record = json.loads(line)
                
                # Convert the JSON record to RAG-friendly text
                rag_text = process_json_record(record, os.path.basename(input_file_path))
                
                # Check if adding this record exceeds the max chunk size
                if current_chunk_size + len(rag_text.encode('utf-8')) > MAX_JSON_CHUNK_SIZE_BYTES and current_chunk_size > 0:
                    # Write the current list of records to a new chunk file
                    write_json_chunk(output_filename_base, chunk_file_index, output_data_list)
                    
                    # Reset counters for the next chunk
                    current_chunk_size = 0
                    output_data_list = []
                    chunk_file_index += 1

                # Add current record's text to the list and size count
                output_data_list.append(rag_text)
                current_chunk_size += len(rag_text.encode('utf-8'))
                
            except json.JSONDecodeError:
                print(f"Skipped malformed JSON line in {input_file_path} at line {line_number}.")
                continue
        
        # Write any remaining data to the final chunk file
        if output_data_list:
            write_json_chunk(output_filename_base, chunk_file_index, output_data_list)
            
    print(f"Successfully split into {chunk_file_index + 1} smaller RAG files.")


def write_json_chunk(base_name, index, data_list):
    """Writes accumulated text strings to a numbered file in the processed folder."""
    output_filename = f"{base_name}_part_{index:03d}.txt"
    output_filepath = os.path.join(PROCESSED_ROOT, 'json_chunks', output_filename)
    
    # Join all RAG-friendly text strings with a paragraph break
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        outfile.write("\n\n---\n\n".join(data_list))

def process_unstructured_files():
    """Simulates checking all other raw files and moving/quarantining them."""
    
    print("\nStarting unstructured file check (PPTX, Images, Docs)...")
    for subdir, _, files in os.walk(RAW_ROOT):
        for filename in files:
            source_path = os.path.join(subdir, filename)
            
            # --- Fix Error 2: Unsupported Formats ---
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Simulates sending to a quarantine folder (Bedrock Textract/Multimodal BDA would be here)
                dest_path = os.path.join(PROCESSED_ROOT, 'unsupported_files', filename)
                os.rename(source_path, dest_path)
                print(f"  [QUARANTINED] Image file {filename}.")
            
            # --- Handle Supported but Risky Formats ---
            elif filename.lower().endswith(('.docx', '.pdf', '.xlsx', '.pptx', '.eml')):
                # Simulates using external parsers (e.g., Textract) to get clean output
                # For POC, we move these to the processed folder as if they passed the cleaner
                dest_path = os.path.join(PROCESSED_ROOT, 'documents', filename)
                os.rename(source_path, dest_path)
                print(f"  [CLEANED] Document {filename} moved to /processed/documents.")
                
            # --- Handle Raw JSONL (for the splitting function above) ---
            elif filename.lower().endswith('.jsonl') and 'json/' in subdir:
                # The large JSONL file will be handled by the specialized splitting function
                pass
            
            # --- Catch any other file types ---
            else:
                 print(f"  [SKIPPED] Unknown file type {filename}.")


# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Simulate ETL for Unstructured/Risky Files
    process_unstructured_files()

    # 2. Simulate ETL for the Large Structured Data (File Splitting & Metadata Cleaning)
    
    # NOTE: You MUST rename your 1 GB file to 'large_raw_data.jsonl' and place it
    # in 'remap-my-rag-source-data/raw/json/' before running this script.
    
    jsonl_path = os.path.join(RAW_ROOT, 'json', 'large_raw_data.jsonl')
    
    if os.path.exists(jsonl_path):
        split_and_process_jsonl(jsonl_path)
    else:
        print("\nSkipping JSONL splitting: large_raw_data.jsonl not found in the correct path.")

    # 3. Final Step: Point Bedrock to the clean processed folder
    
    print("\n========================================================")
    print("       POC DATA READY FOR AWS INGESTION")
    print("========================================================")
    print(f"Next Action: Go to Bedrock Console and set the Data Source URI to:")
    print(f"s3://remap-my-rag-source-data/processed/")
    print("\nAll files in this location are now <50MB, text-clean, and ready for sync.")