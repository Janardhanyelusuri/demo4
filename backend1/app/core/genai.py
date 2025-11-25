import os
import json
import re
import time
import logging
from openai import AzureOpenAI, RateLimitError
from dotenv import load_dotenv

# Set up basic logging configuration
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()  # ✅ REQUIRED to load variables from .env
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")

# --- Configuration for Resilience ---
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # Starting wait time in seconds
REQUEST_DELAY = 3  # Delay in seconds between successful LLM calls to avoid rate limits (set to 0 to disable)

def llm_call(prompt: str) -> str:
    """
    Calls the Azure OpenAI service, sets the response token limit, and 
    implements exponential backoff to handle RateLimitError (HTTP 429).
    Returns the extracted JSON string or an empty string on failure.
    """
    
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_VERSION
        )
    except Exception as e:
        logging.error(f"Error initializing AzureOpenAI client: {e}")
        return ""

    for attempt in range(MAX_RETRIES):
        try:
            # 1. Set Token Limit: max_tokens is crucial for cost and speed.
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful FinOps and cloud optimization assistant. Your response must be in the specified JSON format only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=800,  # ✅ Token Limit Set (Adjust as needed)
            )

            # --- Success handling ---
            output_text = response.choices[0].message.content
            
            # Clean output_text to extract only the JSON
            # This regex captures the outermost JSON object starting with { and ending with }
            match = re.search(r"(\{.*\})", output_text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                # Fallback to the whole output text if the pattern is not found
                json_str = output_text

            # Add delay after successful request to prevent rate limit exhaustion
            if REQUEST_DELAY > 0:
                time.sleep(REQUEST_DELAY)

            # Return the raw JSON string for the caller (llm_analysis.py) to process
            return json_str
            
        except RateLimitError as e:
            # 2. Rate Limit Handling (Exponential Backoff)
            if attempt < MAX_RETRIES - 1:
                # Calculate backoff time: 2^attempt * INITIAL_BACKOFF
                backoff_time = INITIAL_BACKOFF * (2 ** attempt)
                logging.warning(f"Rate limit hit (429). Retrying in {backoff_time:.2f} seconds (Attempt {attempt + 1}/{MAX_RETRIES}).")
                time.sleep(backoff_time)
            else:
                # Max retries reached
                logging.error(f"Rate limit hit, max retries reached after {MAX_RETRIES} attempts. Error: {e}")
                return "" # Return empty string on exhausted retries

        except Exception as e:
            # Handle all other non-rate-limit errors (network, auth, JSON, etc.)
            logging.error(f"Unforeseen Error during LLM processing (Attempt {attempt + 1}): {e}")
            # For non-recoverable errors, stop and return empty string
            return ""
            
    # Should only be reached if the loop finished due to returning on success
    # or if all retries failed and the exception was logged/handled inside the loop.
    return ""