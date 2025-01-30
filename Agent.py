import requests
import pandas as pd
import logging
import time
from typing import Dict, List, Optional
from huggingface_hub import InferenceClient
import os

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HashtagAnalyzer:
    def __init__(self, huggingface_api_key: str, serper_api_key: str):
        self.serper_api_key = serper_api_key
        self.client = InferenceClient(api_key=huggingface_api_key)
        logger.info("Hugging Face client initialized successfully")

    def search_serper(self, query: str) -> List[Dict]:
        """Search using Serper API"""
        try:
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            payload = {
                'q': query,
                'num': 3  
            }
            response = requests.post(
                'https://google.serper.dev/search',
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json().get('organic', [])
        except Exception as e:
            logger.error(f"Serper search failed for query '{query}': {e}")
            return []

    def generate_with_llama(self, prompt: str) -> str:
        """Generate response using Llama model via Hugging Face"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            completion = self.client.chat.completions.create(
                model="meta-llama/Llama-3.1-3B-Instruct",
                messages=messages,
                max_tokens=500
            )
            
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Llama generation failed: {e}")
            return f"Generation failed: {str(e)}"

    def analyze_hashtag(self, hashtag: str) -> str:
        """Analyze a single hashtag using Serper and Llama"""
        try:

            search_query = f"{hashtag} Tamil politics meaning context"
            search_results = self.search_serper(search_query)

       
            context = ""
            for result in search_results:
                context += f"{result.get('snippet', '')} "

            # Create prompt for Llama
            prompt = f"""As a Tamil political expert, provide a one-line explanation for this hashtag: {hashtag}

Context: {context}

Rules for the response:
1. Must be ONE line only
2. If it's a Tamil word, provide the expansion/meaning
3. If it's a person's name, add their role/position
4. If it's a party name, add "(Political Party)"
5. If meaning is unclear, make a reasonable assumption based on common Tamil political context
6. Maximum 10-12 words only
7. For abbreviated Tamil words, provide full Tamil word meaning

Example format:
"#DMK" -> "Dravida Munnetra Kazhagam (Political Party)"
"#MKStalin" -> "MK Stalin, Chief Minister of Tamil Nadu"

Your one-line explanation:"""

            explanation = self.generate_with_llama(prompt)
            time.sleep(5)
            
            return explanation

        except Exception as e:
            logger.error(f"Failed to analyze hashtag {hashtag}: {e}")
            return f"Analysis failed: {str(e)}"

    def process_excel(self, input_file: str, output_file: Optional[str] = None) -> None:
        """Process Excel file containing hashtags"""
        try:
            df = pd.read_excel(input_file)
            if 'hashtag' not in df.columns:
                raise ValueError("Excel file must contain a 'hashtag' column")

            logger.info(f"Processing {len(df)} hashtags...")

            explanations = []
            for index, hashtag in df['hashtag'].items():
                logger.info(f"Analyzing hashtag {index + 1}/{len(df)}: {hashtag}")
                explanation = self.analyze_hashtag(hashtag)
                explanations.append(explanation)

            df['explanation'] = explanations

            if output_file is None:
                output_file = input_file.replace('.xlsx', '_analyzed.xlsx')
            df.to_excel(output_file, index=False)
            logger.info(f"Analysis complete. Results saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to process Excel file: {e}")
            raise

def main():
    HUGGINGFACE_API_KEY = "<enter your api key>"
    SERPER_API_KEY = "enter your api key"
    
    try:
        analyzer = HashtagAnalyzer(HUGGINGFACE_API_KEY, SERPER_API_KEY)
        analyzer.process_excel('hash.xlsx')
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
