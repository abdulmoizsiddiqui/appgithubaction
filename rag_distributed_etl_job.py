import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, pandas_udf
import pandas as pd
from docx import Document # Requires installing external library in Glue environment

# 1. Initialize Glue Context for parallel processing
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Define S3 paths (dynamically passed to the Glue job)
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'S3_RAW_PATH', 'S3_PROCESSED_PATH'])
raw_s3_path = args['S3_RAW_PATH']
processed_s3_path = args['S3_PROCESSED_PATH']

# --- ETL STEP 1: PARSING, CLEANING, AND PII REDACTION (Parallel Step) ---

# UDF to simulate document parsing and PII redaction on each file chunk
@pandas_udf('string')
def process_content_udf(file_content: pd.Series, file_extension: pd.Series) -> pd.Series:
    """
    This function simulates cleaning, parsing, and PII detection on worker nodes.
    Each row in the Pandas Series represents a distributed chunk of data.
    """
    cleaned_texts = []
    for content, ext in zip(file_content, file_extension):
        text = str(content)
        
        # --- Simulate Multi-Modal Parsing (PPTX, DOCX, XLSX to Text) ---
        if ext in ['.docx', '.pptx']:
            # In production, specialized libraries or Amazon Textract API calls run here
            text = f"DOCUMENT TEXT EXTRACTED: {text[:500]}..." # Placeholder for actual extraction
        
        elif ext in ['.xlsx', '.csv']:
            # Simulate converting tabular data into narrative text
            text = f"TABULAR DATA NARRATIVE: Summary of sheet: {text[:500]}..."

        # --- Simulate PII/PHI Redaction (HIPAA Compliance) ---
        # AWS Glue has a built-in PII transform, or you use Amazon Comprehend in a batch step
        if "patient" in text.lower() or "ssn" in text.lower():
             text = text.replace("John Smith", "[PHI_NAME_MASKED]")
             text = f"[HIPAA REDACTION APPLIED] {text}"
        
        cleaned_texts.append(text)
        
    return pd.Series(cleaned_texts)

# 1. Load data from the raw S3 path, handling various formats
data_frame = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": [raw_s3_path], "recurse": True},
    format="json" # Glue can read JSONL as a distributed object
).toDF()

# 2. Add metadata columns for file type and content
data_frame = data_frame.withColumn("file_extension", spark.functions.element_at(spark.functions.split(col("s3_uri"), "\."), -1))
data_frame = data_frame.withColumn("file_content", col("raw_data_column")) # Assume raw content column

# 3. Apply the distributed cleaning and parsing UDF
data_frame = data_frame.withColumn("cleaned_text", process_content_udf(col("file_content"), col("file_extension")))

# --- ETL STEP 2: DISTRIBUTED SPLITTING (Solving the 50MB Limit) ---

# This step solves the 50MB limit by forcing Spark to write the output into many small files
# The .repartition(N) function forces Spark to use N partitions (or workers) to write data.
# This ensures a large file is read once and written out as many small files in parallel.

# NOTE: The number of partitions directly controls the number of output files.
# For a 1 GB file, we might aim for 100 small files to prove parallelism.
data_frame = data_frame.repartition(100) 

# Select only the cleaned text and write as small, parallel chunks
glueContext.write_dynamic_frame.from_options(
    frame=DynamicFrame.fromDF(data_frame.select("cleaned_text"), glueContext, "cleaned_data"),
    connection_type="s3",
    connection_options={"path": processed_s3_path},
    format="text",  # Output format is simple text, ready for Bedrock
    format_options={"writeHeader": False}
)

job.commit()