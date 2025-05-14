"""
Generation Agent for The Second Mind
Creates initial hypotheses based on query and web data.
"""
import time
from typing import Dict, Any, List, Optional
import random
import json
from .base_agent import BaseAgent
from web.google_search_wrapper import SearchAPI  # ✅ CORRECT
from web.scraper import WebScraper
from utils.gemini_client import generate_text
import re

class GenerationAgent(BaseAgent):
    """
    Generation Agent creates initial hypotheses based on input queries
    and real-time web data extraction, enhanced with Gemini LLM.
    """
    
    def __init__(self, web_scraper: WebScraper, search_api: SearchAPI):
        """
        Initialize the Generation Agent.
        
        Args:
            web_scraper: Web scraper instance for extracting data
            search_api: Search API instance for web queries
        """
        super().__init__("generation", "Generation Agent")
        self.web_scraper = web_scraper
        self.search_api = search_api
        self.required_context_keys = ["query"]

    def process(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process the input query and generate initial hypotheses while ensuring proper context validation.
        
        Args:
            context: Current context containing the query
        
        Returns:
            Updated context with extracted web data and generated hypotheses
        """
        if not isinstance(context, dict):
            raise ValueError(f"Expected dictionary as context, got {type(context)}")

        generated_data = {
            "web_data": [],
            "hypotheses": [],
            "cycle": context.get("cycle", 0) + 1,
            "generation_timestamp": time.time(),
        }

        try:
            is_valid, error_msg = self.validate_context(context, self.required_context_keys)
            if not is_valid:
                self.logger.error(f"Context validation failed: {error_msg}")
                raise ValueError(error_msg)

            query = context["query"]
            self.logger.info(f"Generating hypotheses for query: {query}")

            search_results = self.search_api.search(query, num_results=5)
            web_data = []
            if isinstance(search_results, list):
                for result in search_results:
                    try:
                        url = result.get("link", "")
                        if url:
                            extracted_data = self.web_scraper.scrape(url)
                            
                            # More robust error handling for extracted_data
                            if extracted_data and isinstance(extracted_data, dict):
                                # Check if content exists and is not None
                                content = extracted_data.get("content")
                                if content is not None:
                                    # Check if content is a string or has a text attribute
                                    if isinstance(content, str):
                                        content_text = content
                                    elif isinstance(content, dict) and "text" in content:
                                        content_text = content["text"]
                                    else:
                                        content_text = str(content)  # Convert to string as fallback
                                    
                                    web_data.append({
                                        "source": url,
                                        "title": result.get("title", ""),
                                        "content": content_text,
                                        "metadata": extracted_data.get("metadata", {})
                                    })
                                else:
                                    self.logger.warning(f"No content found in extracted data for URL: {url}")
                            else:
                                self.logger.warning(f"Invalid extracted data format for URL: {url}")
                    except Exception as e:
                        self.logger.warning(f"Error scraping {result.get('link', 'Unknown URL')}: {str(e)}")

            hypotheses = self._generate_hypotheses_with_gemini(query, web_data)
            generated_data["web_data"] = web_data
            generated_data["hypotheses"] = hypotheses

        except Exception as e:
            self.logger.error(f"Error in Generation Agent: {str(e)}")
            generated_data["hypotheses"] = [{
                "id": f"fallback-{int(time.time())}",
                "statement": f"General hypothesis about {context.get('query', 'the topic')}",
                "confidence": 0.3,
                "sources": []
            }]

        return {"results": generated_data}
    
    def _sanitize_json_response(self, response: str) -> str:
        """
        Cleans and sanitizes the Gemini response to ensure proper JSON formatting.
        
        Args:
            response: The raw response from Gemini API.
        
        Returns:
            A sanitized JSON string ready for parsing.
        """
        if not response:
            return "[]"

        # First try to extract JSON content from markdown code blocks
        code_block_match = re.search(r"```(?:json)?\n(.*?)\n```", response, flags=re.DOTALL)
        if code_block_match:
            response = code_block_match.group(1).strip()
        
        # Handle common formatting issues
        try:
            # Try to directly parse the JSON first
            json.loads(response)
            return response
        except json.JSONDecodeError:
            # If it fails, apply the sanitization steps
            
            # Fix unescaped quotes in string values
            response = re.sub(r'(?<!")(?<!\\)"(?![\s,\]}])', r'\"', response)
            
            # Fix apostrophes that look like quotes
            response = re.sub(r'(\w)"s', r'\1\'s', response)
            
            # Ensure valid JSON format - convert single quotes to double quotes
            response = response.replace("'", '"')
            
            # Handle escape sequences properly
            response = response.replace('\\"', '\\\\"')  # Properly escape quotes
            
            # Remove any trailing commas before closing brackets
            response = re.sub(r",\s*([\]}])", r"\1", response)
            
            # Ensure the response is wrapped in square brackets if it looks like JSON objects
            if response.strip().startswith('{') and response.strip().endswith('}'):
                if not (response.strip().startswith('[') and response.strip().endswith(']')):
                    response = f"[{response}]"
            
            return response

    def _generate_hypotheses_with_gemini(self, query: str, web_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate hypotheses using Gemini LLM based on the query and extracted web data.
        Fixes JSON parsing issues caused by Markdown formatting.
        """
        web_content_summary = ""
        for i, data in enumerate(web_data[:3]):  
            web_content_summary += f"\nSource {i+1}: {data.get('title', 'Untitled')}\n"
            web_content_summary += f"URL: {data.get('source', 'N/A')}\n"
            content_preview = (data.get('content', '')[:500] + "...") if len(data.get('content', '')) > 500 else data.get('content', '')
            web_content_summary += f"Content: {content_preview}\n"

        prompt = f"""
        Based on the following query and web data, generate 3-5 well-formed hypotheses.
        
        QUERY: {query}
        WEB DATA:
        {web_content_summary}
        
        FORMAT RESPONSE AS JSON LIST:
        [
            {{"statement": "Hypothesis A", "confidence": 0.8, "rationale": "Some explanation"}} ,
            {{"statement": "Hypothesis B", "confidence": 0.7, "rationale": "Another explanation"}}
        ]
        
        Important: Format the response as a valid JSON array only, without any additional text, explanation or code block markers.
        """

        # Call Gemini API
        gemini_response = generate_text(prompt)

        # Check for empty response
        if not gemini_response or not gemini_response.strip():
            self.logger.error("Gemini API returned an empty response. Using fallback method.")
            return self._generate_hypotheses_fallback(query, web_data)

        # Clean and sanitize response
        sanitized_response = self._sanitize_json_response(gemini_response)
        
        try:
            parsed_hypotheses = json.loads(sanitized_response)
            
            # Ensure parsed response is a list
            if not isinstance(parsed_hypotheses, list):
                self.logger.error("Gemini response is not a list. Using fallback method.")
                return self._generate_hypotheses_fallback(query, web_data)

            # Convert parsed data into structured hypotheses
            hypotheses = []
            for i, h in enumerate(parsed_hypotheses):
                if not isinstance(h, dict) or "statement" not in h or "confidence" not in h:
                    self.logger.warning(f"Skipping malformed hypothesis: {h}")
                    continue  # Skip bad data
                
                # Track sources - use integer indices for Streamlit frontend compatibility
                sources_indices = []
                sources_urls = []
                
                for j, data in enumerate(web_data):
                    if "content" in data and h["statement"].lower() in data["content"].lower():
                        sources_indices.append(j)
                        if "source" in data:
                            sources_urls.append(data["source"])
                
                hypotheses.append({
                    "id": f"hyp-{int(time.time())}-{i}",
                    "statement": h["statement"],
                    "confidence": float(h.get("confidence", 0.5)),  # Default confidence if missing
                    "rationale": h.get("rationale", ""),
                    "sources": sources_indices[:3],  # Use integer indices for Streamlit
                    "source_urls": sources_urls[:3],  # Keep URLs for reference if needed
                    "created_at": time.time()
                })

            # Ensure hypotheses were generated
            if not hypotheses:
                self.logger.warning("No valid hypotheses generated. Using fallback method.")
                return self._generate_hypotheses_fallback(query, web_data)
            
            return hypotheses
        
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}. Response: {sanitized_response}")
            
            # Try a more robust extraction approach
            try:
                # Extract components from the malformed JSON using regex
                statements = re.findall(r'"statement"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', sanitized_response)
                confidence_matches = re.findall(r'"confidence"\s*:\s*(0\.\d+|1\.0|1)', sanitized_response)
                rationale_matches = re.findall(r'"rationale"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', sanitized_response)
                
                confidences = [float(c) for c in confidence_matches]
                
                # Ensure we have at least statements and confidences
                if statements and confidences and len(statements) == len(confidences):
                    hypotheses = []
                    for i, statement in enumerate(statements):
                        confidence = confidences[i] if i < len(confidences) else 0.5
                        rationale = rationale_matches[i] if i < len(rationale_matches) else "Extracted from malformed JSON"
                        
                        # Use integers for sources to match Streamlit expectations
                        hypotheses.append({
                            "id": f"hyp-{int(time.time())}-{i}",
                            "statement": statement,
                            "confidence": confidence,
                            "rationale": rationale,
                            "sources": [],  # Integer indices
                            "source_urls": [],  # Empty URLs list
                            "created_at": time.time()
                        })
                    
                    if hypotheses:
                        return hypotheses
            except Exception as regex_error:
                self.logger.error(f"Failed to extract hypotheses with regex: {str(regex_error)}")
            
            # If all else fails, use the fallback method
            return self._generate_hypotheses_fallback(query, web_data)
    
    def _generate_hypotheses_fallback(self, query: str, web_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback method to generate hypotheses when Gemini fails.
        
        Args:
            query: The original query
            web_data: List of extracted web data dictionaries
            
        Returns:
            List of hypothesis dictionaries
        """
        hypotheses = []
        
        # Extract key concepts from web data
        concepts = self._extract_key_concepts(web_data)
        
        # Generate 3-5 hypotheses based on extracted concepts
        num_hypotheses = min(len(concepts), random.randint(3, 5))
        
        # Ensure we have at least 3 hypotheses
        if num_hypotheses < 3:
            num_hypotheses = 3
            
        # Default hypotheses if we have no concepts
        if not concepts or len(concepts) == 0:
            timestamp = int(time.time())
            return [
                {
                    "id": f"hyp-{timestamp}-1",
                    "statement": f"AI technologies are revolutionizing forensic investigations by automating evidence analysis",
                    "confidence": 0.8,
                    "rationale": "AI can process large volumes of digital evidence faster than humans",
                    "sources": [0, 1],  # Use integer indices for Streamlit
                    "source_urls": [],  # Empty list for URLs
                    "created_at": time.time()
                },
                {
                    "id": f"hyp-{timestamp}-2",
                    "statement": f"Machine learning algorithms can identify patterns in forensic data that humans might miss",
                    "confidence": 0.75,
                    "rationale": "Pattern recognition is a key strength of modern AI systems",
                    "sources": [2, 3],  # Use integer indices for Streamlit
                    "source_urls": [],  # Empty list for URLs
                    "created_at": time.time()
                },
                {
                    "id": f"hyp-{timestamp}-3",
                    "statement": f"Ethical concerns about AI in forensics include potential bias and privacy implications",
                    "confidence": 0.7,
                    "rationale": "AI systems can inherit biases from training data",
                    "sources": [0, 4],  # Use integer indices for Streamlit
                    "source_urls": [],  # Empty list for URLs
                    "created_at": time.time()
                }
            ]
        
        for i in range(num_hypotheses):
            if i < len(concepts):
                concept = concepts[i]
                source_indices = []
                source_urls = []
                
                # Extract both indices and URLs for sources
                if web_data and isinstance(web_data, list):
                    for j, data in enumerate(web_data):
                        if isinstance(data, dict):
                            content = data.get("content", "")
                            source_matched = False
                            
                            # Check content in different formats
                            if isinstance(content, str) and concept.lower() in content.lower():
                                source_matched = True
                            elif isinstance(content, dict) and "text" in content:
                                if concept.lower() in content["text"].lower():
                                    source_matched = True
                            
                            if source_matched:
                                source_indices.append(j)  # Store index for Streamlit
                                if "source" in data:
                                    source_urls.append(data["source"])  # Store URL for reference
                
                # Provide default sources if none found
                if not source_indices:
                    source_indices = [0, 1] if len(web_data) >= 2 else [0]
                
                hypothesis = {
                    "id": f"hyp-{int(time.time())}-{i}",
                    "statement": f"{concept} is a significant factor in the impact of AI on forensic investigations",
                    "confidence": round(random.uniform(0.5, 0.9), 2),
                    "rationale": f"This concept appears frequently in relevant sources",
                    "sources": source_indices[:3],  # Use integer indices for Streamlit, limit to top 3
                    "source_urls": source_urls[:3],  # Store URLs for reference, limit to top 3
                    "created_at": time.time()
                }
                
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _extract_key_concepts(self, web_data: List[Dict[str, Any]]) -> List[str]:
        """
        Extract key concepts from web data with improved error handling.
        
        Args:
            web_data: List of extracted web data
            
        Returns:
            List of key concepts
        """
        # Simple concept extraction - in a real system, use NLP techniques
        concepts = set()
        
        if not web_data:
            self.logger.warning("No web data provided to extract concepts from")
            return ["AI in forensics", "Digital forensics", "Forensic investigation"]  # Default concepts
        
        for data in web_data:
            if not isinstance(data, dict):
                continue
                
            # Extract concepts from title (simple approach)
            title = data.get("title", "")
            if title and isinstance(title, str):
                for word in title.split():
                    if len(word) > 4:  # Simple filter for meaningful words
                        concepts.add(word.strip('.,!?()[]{}').title())
            
            # Extract concepts from content (simple approach)
            content = data.get("content", {})
            if isinstance(content, dict):
                # Handle nested content structure
                text_content = content.get("text", "")
            elif isinstance(content, str):
                # Direct content string
                text_content = content
            else:
                # Skip if content is neither dict nor string
                continue
                
            if text_content and isinstance(text_content, str):
                content_words = text_content.split()
                for i in range(len(content_words) - 1):
                    if i + 1 < len(content_words):  # Ensure we don't go out of bounds
                        bigram = f"{content_words[i]} {content_words[i+1]}"
                        if len(bigram) > 10:  # Simple filter for meaningful bigrams
                            concepts.add(bigram.strip('.,!?()[]{}').title())
        
        # If no concepts were found, return default fallback concepts
        if not concepts:
            return ["AI in forensics", "Digital forensics", "Forensic investigation"]
            
        return list(concepts)
    
    def generate(self, query: str) -> Dict[str, Any]:
        """
        Generate initial hypotheses based on a query.

        Args:
            query: The input query

        Returns:
            A dictionary containing generated hypotheses
        """
        print(f"[DEBUG] Received query: {query}")

        if not isinstance(query, str) or not query.strip():
            print("[ERROR] Invalid query: Expected a non-empty string.")
            return {"results": []}

        try:
            # Placeholder function for hypothesis generation
            def simple_generation_function(query):
                return [f"Hypothesis {i + 1}: {query} - Variation {i + 1}" for i in range(2, 6)]

            # Generate hypotheses
            result = simple_generation_function(query)
            print(f"[DEBUG] Raw Generation Output: {result}")

            if not isinstance(result, list):
                print("[WARNING] Generation function returned unexpected format. Expected list.")
                return {"results": []}

            context = {"query": query, "hypotheses": result}
            print(f"[DEBUG] Context Before Processing: {context}")

            updated_context = self.process(context)
            print(f"[DEBUG] Context After Processing: {updated_context}")

            # Ensure `updated_context` is a dictionary and extract hypotheses correctly
            if not isinstance(updated_context, dict):
                print("[ERROR] process(context) returned a non-dictionary format. Expected dict.")
                return {"results": []}

            # Check if `updated_context` has `results` and extract hypotheses
            if "results" in updated_context and isinstance(updated_context["results"], dict):
                hypotheses = updated_context["results"].get("hypotheses", [])
            else:
                hypotheses = updated_context.get("hypotheses", [])

            if not isinstance(hypotheses, list):
                print(f"[ERROR] Processed hypotheses is not a list. Received type: {type(hypotheses)}")
                return {"results": []}

            print(f"[DEBUG] Final Processed Hypotheses: {hypotheses}")
            return {"results": hypotheses}

        except Exception as e:
            print(f"[ERROR] Exception in generate method: {e}")
            return {"results": []}