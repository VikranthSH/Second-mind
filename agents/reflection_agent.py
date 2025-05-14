"""
Reflection Agent for The Second Mind
Checks coherence and consistency of hypotheses against web data and knowledge.
"""
import time
from typing import Dict, Any, List
import re
import json

from .base_agent import BaseAgent
from utils.gemini_client import generate_text

class ReflectionAgent(BaseAgent):
    """
    Reflection Agent verifies the coherence and consistency of generated hypotheses
    against extracted web data and existing knowledge using Gemini LLM.
    """
    
    def __init__(self):
        """Initialize the Reflection Agent."""
        super().__init__(agent_id="reflection", name="Reflection Agent")
        self.required_context_keys = ["hypotheses", "web_data", "query"]
    
    def process(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Reflect on and evaluate the coherence of the generated hypotheses.

        Args:
            context: Current context containing hypotheses and web data

        Returns:
            Updated context with reflection results
        """
        start_time = time.time()
        success = False

        try:
        # Validate context
            if not isinstance(context, dict):
                print(f"Warning: ReflectionAgent received {type(context)}, converting to dictionary.")
                context = {"results": context} if isinstance(context, list) else {"results": []}

                hypotheses = context.get("hypotheses", [])
                return self.analyze(hypotheses)

            required_keys = {"hypotheses", "web_data", "query"}
            missing_keys = required_keys - context.keys()
            if missing_keys:
                raise ValueError(f"Missing required context keys: {missing_keys}")

            hypotheses = context["hypotheses"]
            web_data = context["web_data"]
            query = context["query"]

            self.logger.info(f"Reflecting on {len(hypotheses)} hypotheses")

        # Process each hypothesis with Gemini
            reflection_results = []
            for hypothesis in hypotheses:
                try:
                    reflection = self._reflect_with_gemini(hypothesis, web_data, query)
                    reflection_results.append(reflection)
                except Exception as e:
                    self.logger.warning(f"Gemini reflection failed for hypothesis {hypothesis.get('id', 'unknown')}: {str(e)}")
                # Fallback to traditional reflection
                    reflection = self._reflect_on_hypothesis(hypothesis, web_data, query)
                    reflection_results.append(reflection)

        # Update context with reflection results
            context["reflection_results"] = reflection_results
            context["reflection_timestamp"] = time.time()

        # Identify hypotheses needing improvement
            context["hypotheses_to_improve"] = [
                h.get("id", "unknown") for h, r in zip(hypotheses, reflection_results)
                if not r["is_coherent"] or not r["has_supporting_evidence"]
            ]

            success = True
            self.logger.info(f"Completed reflection on {len(hypotheses)} hypotheses")

        except Exception as e:
            self.logger.error(f"Error in Reflection Agent: {str(e)}")

        # Ensure hypotheses exist in the context
            hypotheses = context.get("hypotheses", [])
            if not isinstance(hypotheses, list):
                raise ValueError("Expected 'hypotheses' to be a list but got {}".format(type(hypotheses)))

        # Provide fallback reflection results
            if hypotheses:
                context["reflection_results"] = [
                    {
                        "hypothesis_id": h.get("id", "unknown"),
                        "is_coherent": True,
                        "coherence_score": 0.5,
                        "has_supporting_evidence": False,
                        "contradictions": [],
                        "supporting_facts": [],
                        "comments": ["Failed to complete reflection"]
                    }
                    for h in hypotheses
                ]
            else:
                context["reflection_results"] = []

    # Update metrics
        processing_time = time.time() - start_time
        self.update_metrics(processing_time, success)

        return context

    def _reflect_with_gemini(self, hypothesis: Dict[str, Any], web_data: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Use Gemini LLM to reflect on a hypothesis by checking its coherence and evidence.
        
        Args:
            hypothesis: The hypothesis to evaluate
            web_data: Web data for evidence checking
            query: Original query
            
        Returns:
            Dictionary containing reflection results
        """
        # Extract hypothesis statement
        statement = hypothesis["statement"]
        
        # Prepare web data for the prompt (limiting to prevent token overflow)
        web_content = ""
        for i, data in enumerate(web_data[:3]):  # Limit to 3 sources
            title = data["title"]
            # Truncate content to avoid token limits
            content = data["content"][:500] + "..." if len(data["content"]) > 500 else data["content"]
            web_content += f"\nSOURCE {i+1}: {data['source']}\nTITLE: {title}\nCONTENT: {content}\n"
        
        # Create prompt for Gemini
        prompt = f"""
        As a critical thinking assistant, evaluate the following hypothesis against the provided web data.
        
        QUERY: {query}
        
        HYPOTHESIS: {statement}
        
        WEB DATA:
        {web_content}
        
        Analyze the hypothesis for:
        1. Logical coherence (is it internally consistent and well-formed?)
        2. Supporting evidence from the web data
        3. Contradicting evidence from the web data
        
        FORMAT YOUR RESPONSE AS JSON with these fields:
        - is_coherent: boolean (true/false)
        - coherence_score: number between 0-1
        - has_supporting_evidence: boolean (true/false)
        - supporting_facts: list of objects with "fact" and "source" fields
        - contradictions: list of objects with "contradiction" and "source" fields
        - comments: list of strings with your analysis
        
        EXAMPLE:
        {{
            "is_coherent": true,
            "coherence_score": 0.85,
            "has_supporting_evidence": true,
            "supporting_facts": [
                {{"fact": "Solar panels have shown 20% efficiency in urban environments", "source": "https://example.com/solar-study"}}
            ],
            "contradictions": [
                {{"contradiction": "Solar panels are too expensive for widespread urban use", "source": "https://example.com/cost-analysis"}}
            ],
            "comments": ["The hypothesis is logically sound but faces economic challenges"]
        }}
        """
        
        # Call Gemini for reflection
        gemini_response = generate_text(prompt, temperature=0.2)  # Lower temperature for more factual analysis
        
        if not gemini_response or not gemini_response.strip():
            self.logger.error("Gemini returned an empty response. Check API status or input formatting.")
            return {"error": "Gemini response was empty"}

        try:
            # Parse Gemini response
            # First, try to clean up any JSON formatting issues
            cleaned_response = self._clean_json_response(gemini_response)
            reflection_result = json.loads(cleaned_response)
            
            # Add hypothesis ID to the result
            reflection_result["hypothesis_id"] = hypothesis["id"]
            
            # Validate and ensure all expected fields are present
            required_fields = ["is_coherent", "coherence_score", "has_supporting_evidence", 
                            "supporting_facts", "contradictions", "comments"]
            
            for field in required_fields:
                if field not in reflection_result:
                    if field in ["supporting_facts", "contradictions", "comments"]:
                        reflection_result[field] = []
                    elif field in ["is_coherent", "has_supporting_evidence"]:
                        reflection_result[field] = False
                    elif field == "coherence_score":
                        reflection_result[field] = 0.5
            
            return reflection_result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Gemini response as JSON: {gemini_response[:100]}... Error: {str(e)}")
            # Fall back to extracting useful information from the response
            return self._extract_reflection_from_text(gemini_response, hypothesis["id"])

    def _reflect_on_hypothesis(self, hypothesis: Dict[str, Any], web_data: List[Dict[str, Any]], 
                              query: str) -> Dict[str, Any]:
        """
        Reflect on a single hypothesis by checking its coherence and evidence.
        Fallback method when Gemini fails.
        
        Args:
            hypothesis: The hypothesis to evaluate
            web_data: Web data for evidence checking
            query: Original query
            
        Returns:
            Dictionary containing reflection results
        """
        # Extract hypothesis statement
        statement = hypothesis["statement"]
        
        # Check for internal coherence
        is_coherent = self._check_coherence(statement)
        coherence_score = 0.8 if is_coherent else 0.4
        
        # Find supporting evidence and contradictions
        supporting_facts = []
        contradictions = []
        
        for data in web_data:
            content = data["content"].lower()
            title = data["title"].lower()
            
            # Extract keywords from hypothesis
            keywords = self._extract_keywords(statement)
            
            # Check for supporting evidence
            for keyword in keywords:
                if keyword.lower() in content or keyword.lower() in title:
                    # Find the sentence containing the keyword
                    sentences = re.split(r'[.!?]', content)
                    for sentence in sentences:
                        if keyword.lower() in sentence.lower():
                            fact = sentence.strip()
                            if fact and len(fact) > 10:
                                supporting_facts.append({
                                    "fact": fact,
                                    "source": data["source"]
                                })
                            break
            
            # Check for contradictions - simple approach
            # In a real system, use more sophisticated NLP techniques
            negation_phrases = ["not", "cannot", "doesn't", "isn't", "won't", "never"]
            for phrase in negation_phrases:
                if any(f"{phrase} {keyword.lower()}" in content for keyword in keywords):
                    sentences = re.split(r'[.!?]', content)
                    for sentence in sentences:
                        if any(f"{phrase} {keyword.lower()}" in sentence.lower() for keyword in keywords):
                            contradiction = sentence.strip()
                            if contradiction and len(contradiction) > 10:
                                contradictions.append({
                                    "contradiction": contradiction,
                                    "source": data["source"]
                                })
                            break
        
        # Limit the number of facts and contradictions
        supporting_facts = supporting_facts[:5]  # Top 5 supporting facts
        contradictions = contradictions[:3]  # Top 3 contradictions
        
        # Determine if there's sufficient supporting evidence
        has_supporting_evidence = len(supporting_facts) > 0
        
        # Generate comments
        comments = []
        if not is_coherent:
            comments.append("The hypothesis may not be internally coherent.")
        if not has_supporting_evidence:
            comments.append("The hypothesis lacks supporting evidence from the web data.")
        if contradictions:
            comments.append("The hypothesis has contradicting evidence that should be addressed.")
        if is_coherent and has_supporting_evidence and not contradictions:
            comments.append("The hypothesis is well-supported by the web data.")
        
        return {
            "hypothesis_id": hypothesis["id"],
            "is_coherent": is_coherent,
            "coherence_score": coherence_score,
            "has_supporting_evidence": has_supporting_evidence,
            "contradictions": contradictions,
            "supporting_facts": supporting_facts,
            "comments": comments
        }
    
    def _check_coherence(self, statement: str) -> bool:
        """
        Check if a statement is internally coherent.
        
        Args:
            statement: Hypothesis statement
            
        Returns:
            Boolean indicating coherence
        """
        # Simple coherence check - in a real system, use more sophisticated NLP
        # Check if statement length is reasonable
        if len(statement) < 10 or len(statement) > 500:
            return False
        
        # Check if statement contains contradictory phrases
        contradictory_pairs = [
            ("increase", "decrease"),
            ("always", "never"),
            ("all", "none"),
            ("positive", "negative")
        ]
        
        statement_lower = statement.lower()
        for word1, word2 in contradictory_pairs:
            if word1 in statement_lower and word2 in statement_lower:
                # Check if they're not part of a comparison (X is better than Y)
                if "than" not in statement_lower and "but" not in statement_lower:
                    return False
        
        return True
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text for evidence searching.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - in a real system, use proper NLP techniques
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                      "for", "of", "in", "to", "with", "by", "about", "could"}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Add any two-word phrases (bigrams)
        words = text.lower().split()
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigrams = [b for b in bigrams if not all(w in stop_words for w in b.split())]
        
        return list(set(keywords + bigrams))
    
    def _clean_json_response(self, response: str) -> str:
        """
        Clean and fix common JSON formatting issues in responses.
        
        Args:
            response: The raw response string
        
        Returns:
            Cleaned JSON string
        """
        # Remove any markdown code block indicators
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find the actual JSON content (typically between curly braces)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        
        # Fix common escape sequence issues
        response = response.replace('\\"', '"')  # Replace incorrectly escaped quotes
        response = response.replace('\\', '\\\\')  # Ensure backslashes are properly escaped
        response = response.replace('\\\\"', '\\"')  # Fix double escaped quotes
        
        # Fix unescaped control characters
        response = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', response)
        
        return response

    def _extract_reflection_from_text(self, text: str, hypothesis_id: str) -> Dict[str, Any]:
        """
        Fallback method to extract reflection information from unstructured text.
        
        Args:
            text: Response text that couldn't be parsed as JSON
            hypothesis_id: ID of the hypothesis
        
        Returns:
            Structured reflection dictionary
        """
        result = {
            "hypothesis_id": hypothesis_id,
            "is_coherent": True,  # Default values
            "coherence_score": 0.5,
            "has_supporting_evidence": False,
            "supporting_facts": [],
            "contradictions": [],
            "comments": []
        }
        
        # Look for coherence information
        if re.search(r'not coherent|lacks coherence|incoherent', text, re.IGNORECASE):
            result["is_coherent"] = False
            result["coherence_score"] = 0.3
        
        # Look for evidence information
        if re.search(r'supporting evidence|evidence supports|well supported', text, re.IGNORECASE):
            result["has_supporting_evidence"] = True
        
        # Extract comments - get sentences that look like analysis
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and not sentence.startswith('{') and not sentence.startswith('"'):
                result["comments"].append(sentence)
        
        # Limit comments to just a few
        result["comments"] = result["comments"][:3]
        
        return result