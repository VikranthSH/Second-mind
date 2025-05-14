import time
from datetime import datetime
import logging
import json
from agents.base_agent import BaseAgent
from utils.gemini_client import generate_text
from typing import Dict, Any, List, Tuple

class MetaReviewAgent(BaseAgent):
    """
    Meta Review Agent evaluates the entire process flow and generates feedback
    for system improvement. This agent acts as a quality control mechanism that
    ensures continuous refinement of the overall system.
    """
    
    def __init__(self, context_manager, storage):
        """
        Initialize the Meta Review Agent.
    
        Args:
            context_manager: The context manager for accessing and updating shared context
            storage: The storage system for retrieving historical data
        """
        super().__init__("meta_review", storage)  # Pass only "meta_review" and storage
        self.context_manager = context_manager  # Store context_manager separately
    
        self.metrics = {
            "cycle_duration": [],
            "agent_performance": {},
            "web_data_quality": [],
            "hypothesis_improvement": []
        }
        self.logger = logging.getLogger(__name__)

    
    def process(self, task_data):
        """
        Evaluate the entire research cycle and provide feedback for improvement.
        
        Args:
            task_data: Dictionary containing cycle information, agent outputs, and timings
            
        Returns:
            Dictionary with feedback, recommendations, and process metrics
        """
        self.logger.info("Meta Review Agent processing cycle evaluation")
        start_time = time.time()
        
        # Safely extract cycle data with default values
        cycle_id = task_data.get("cycle_id", 0)
        cycle_data = task_data.get("cycle_data", {})
        query = task_data.get("query", "")
        
        # Safely get previous cycles with error handling
        try:
            previous_cycles = self.storage.get_previous_cycles(query, cycle_id)
        except Exception as e:
            self.logger.error(f"Error retrieving previous cycles: {e}")
            previous_cycles = []
        
        # Analyze cycle performance
        try:
            performance_metrics = self._analyze_cycle_performance(cycle_data)
        except Exception as e:
            self.logger.error(f"Error analyzing cycle performance: {e}")
            performance_metrics = {"total_cycle_time": 0, "agent_times": {}, "bottlenecks": [], 
                                   "successful_agents": [], "failed_agents": []}
        
        # Evaluate web data quality with error handling
        try:
            web_data_quality = self._evaluate_web_data_quality(cycle_data)
        except Exception as e:
            self.logger.error(f"Error evaluating web data quality: {e}")
            web_data_quality = {"sources_count": 0, "data_freshness": 0, "data_relevance": 0,
                                "data_diversity": 0, "successful_extractions": 0, "failed_extractions": 0}
        
        # Measure hypothesis improvement with error handling
        try:
            hypothesis_improvement = self._measure_hypothesis_improvement(cycle_data, previous_cycles)
        except Exception as e:
            self.logger.error(f"Error measuring hypothesis improvement: {e}")
            hypothesis_improvement = {"current_score": 0, "previous_score": 0, "score_delta": 0,
                                      "complexity_increase": 0, "refinement_count": 0, "improvement_percentage": 0}
        
        # Update metrics history
        self._update_metrics(performance_metrics, web_data_quality, hypothesis_improvement)
        
        # Use Gemini LLM to generate insights and recommendations
        try:
            llm_insights = self._generate_llm_insights(cycle_data, performance_metrics, previous_cycles)
        except Exception as e:
            self.logger.error(f"Error generating LLM insights: {e}")
            llm_insights = {"insights": ["LLM processing failed"], 
                            "recommendations": ["Check LLM service connection"], 
                            "bottlenecks": ["LLM processing"]}
        
        # Prepare meta review result
        meta_review_result = {
            "cycle_id": cycle_id,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": performance_metrics,
            "web_data_quality": web_data_quality,
            "hypothesis_improvement": hypothesis_improvement,
            "insights": llm_insights.get("insights", []),
            "recommendations": llm_insights.get("recommendations", []),
            "bottlenecks": llm_insights.get("bottlenecks", []),
            "execution_time": time.time() - start_time
        }
        
        # Update context with meta review results
        try:
            self.context_manager.update_context({
                "meta_review": meta_review_result
            })
        except Exception as e:
            self.logger.error(f"Error updating context: {e}")
        
        # Save meta review to storage with error handling
        try:
            self.storage.save_meta_review(query, cycle_id, meta_review_result)
        except Exception as e:
            self.logger.error(f"Error saving meta review to storage: {e}")
        
        self.logger.info(f"Meta Review completed in {meta_review_result['execution_time']:.2f}s")
        return meta_review_result
    
    def _analyze_cycle_performance(self, cycle_data):
        """
        Analyze the performance of the current cycle.
        
        Args:
            cycle_data: Dictionary containing cycle information and agent outputs
            
        Returns:
            Dictionary with performance metrics
        """
        performance_metrics = {
            "total_cycle_time": 0,
            "agent_times": {},
            "bottlenecks": [],
            "successful_agents": [],
            "failed_agents": []
        }
        
        # Calculate total cycle time and agent execution times
        agent_outputs = cycle_data.get("agent_outputs", {})
        if not isinstance(agent_outputs, dict):
            self.logger.warning(f"Expected dict for agent_outputs, got {type(agent_outputs)}")
            return performance_metrics
            
        for agent_name, agent_data in agent_outputs.items():
            if isinstance(agent_data, dict) and "execution_time" in agent_data:
                try:
                    exec_time = float(agent_data["execution_time"])
                    performance_metrics["agent_times"][agent_name] = exec_time
                    performance_metrics["total_cycle_time"] += exec_time
                    
                    if agent_data.get("status") == "success":
                        performance_metrics["successful_agents"].append(agent_name)
                    else:
                        performance_metrics["failed_agents"].append(agent_name)
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Error processing execution time for {agent_name}: {e}")
        
        # Identify bottlenecks (agents taking more than 25% of total time)
        total_time = performance_metrics["total_cycle_time"]
        if total_time > 0:
            for agent_name, time_taken in performance_metrics["agent_times"].items():
                if time_taken > 0.25 * total_time:
                    performance_metrics["bottlenecks"].append({
                        "agent": agent_name,
                        "time": time_taken,
                        "percentage": (time_taken / total_time) * 100
                    })
        
        return performance_metrics
    
    def _evaluate_web_data_quality(self, cycle_data):
        """
        Evaluate the quality of web data used in the cycle.
        
        Args:
            cycle_data: Dictionary containing cycle information and web data
            
        Returns:
            Dictionary with web data quality metrics
        """
        web_data_quality = {
            "sources_count": 0,
            "data_freshness": 0,
            "data_relevance": 0,
            "data_diversity": 0,
            "successful_extractions": 0,
            "failed_extractions": 0
        }
        
        web_data = cycle_data.get("web_data", {})
        if not isinstance(web_data, dict):
            self.logger.warning(f"Expected dict for web_data, got {type(web_data)}")
            return web_data_quality
            
        sources = web_data.get("sources", [])
        if not isinstance(sources, list):
            self.logger.warning(f"Expected list for sources, got {type(sources)}")
            return web_data_quality
            
        web_data_quality["sources_count"] = len(sources)
        
        # Count successful and failed extractions
        for source in sources:
            if not isinstance(source, dict):
                continue
                
            if source.get("status") == "success":
                web_data_quality["successful_extractions"] += 1
            else:
                web_data_quality["failed_extractions"] += 1
        
        # Calculate data freshness (average age of sources in days)
        current_time = datetime.now()
        total_age = 0
        valid_timestamps = 0
        
        for source in sources:
            if not isinstance(source, dict):
                continue
                
            timestamp = source.get("timestamp")
            if timestamp:
                try:
                    source_time = datetime.fromisoformat(timestamp)
                    age_days = (current_time - source_time).days
                    total_age += age_days
                    valid_timestamps += 1
                except (ValueError, TypeError):
                    pass
        
        if valid_timestamps > 0:
            web_data_quality["data_freshness"] = total_age / valid_timestamps
        
        # Calculate data relevance (based on keyword matching)
        query = cycle_data.get("query", "")
        if query:
            try:
                query_keywords = set(query.lower().split())
                total_relevance = 0
                content_sources = 0
                
                for source in sources:
                    if not isinstance(source, dict):
                        continue
                        
                    content = source.get("content", "")
                    if content:
                        content_keywords = set(content.lower().split())
                        overlap = len(query_keywords.intersection(content_keywords))
                        if len(query_keywords) > 0:
                            relevance = overlap / len(query_keywords)
                            total_relevance += relevance
                            content_sources += 1
                
                if content_sources > 0:
                    web_data_quality["data_relevance"] = total_relevance / content_sources
            except Exception as e:
                self.logger.error(f"Error calculating data relevance: {e}")
        
        # Calculate data diversity (based on source types)
        try:
            source_types = set()
            for source in sources:
                if isinstance(source, dict):
                    source_types.add(source.get("type", "unknown"))
            
            if web_data_quality["sources_count"] > 0:
                web_data_quality["data_diversity"] = len(source_types) / web_data_quality["sources_count"]
        except Exception as e:
            self.logger.error(f"Error calculating data diversity: {e}")
        
        return web_data_quality
    
    def _measure_hypothesis_improvement(self, cycle_data, previous_cycles):
        """
        Measure the improvement in hypothesis quality across cycles.
        
        Args:
            cycle_data: Dictionary containing cycle information and hypothesis
            previous_cycles: List of previous cycle data
            
        Returns:
            Dictionary with hypothesis improvement metrics
        """
        hypothesis_improvement = {
            "current_score": 0,
            "previous_score": 0,
            "score_delta": 0,
            "complexity_increase": 0,
            "refinement_count": 0,
            "improvement_percentage": 0
        }
        
        # Safely extract current hypothesis
        current_hypothesis = cycle_data.get("hypothesis", {})
        if not isinstance(current_hypothesis, dict):
            self.logger.warning(f"Expected dict for hypothesis, got {type(current_hypothesis)}")
            return hypothesis_improvement
            
        # Get current score safely
        try:
            current_score = float(current_hypothesis.get("score", 0))
            hypothesis_improvement["current_score"] = current_score
        except (ValueError, TypeError):
            self.logger.warning("Invalid hypothesis score format")
            current_score = 0
        
        # Get previous hypothesis score
        if previous_cycles and len(previous_cycles) > 0:
            try:
                previous_cycle = previous_cycles[-1]
                if isinstance(previous_cycle, dict):
                    previous_hypothesis = previous_cycle.get("hypothesis", {})
                    if isinstance(previous_hypothesis, dict):
                        previous_score = float(previous_hypothesis.get("score", 0))
                        hypothesis_improvement["previous_score"] = previous_score
                        hypothesis_improvement["score_delta"] = current_score - previous_score
                        
                        if previous_score > 0:
                            hypothesis_improvement["improvement_percentage"] = (
                                (current_score - previous_score) / previous_score
                            ) * 100
                        
                        # Calculate complexity increase based on text length
                        current_text = current_hypothesis.get("text", "")
                        previous_text = previous_hypothesis.get("text", "")
                        
                        if len(previous_text) > 0:
                            hypothesis_improvement["complexity_increase"] = (
                                (len(current_text) - len(previous_text)) / len(previous_text)
                            ) * 100
                        
                        # Count refinements
                        hypothesis_improvement["refinement_count"] = len(previous_cycles) + 1
            except (ValueError, TypeError, IndexError) as e:
                self.logger.error(f"Error measuring hypothesis improvement: {e}")
        
        return hypothesis_improvement
    
    def _update_metrics(self, performance_metrics, web_data_quality, hypothesis_improvement):
        """
        Update the agent's metrics history.
        
        Args:
            performance_metrics: Dictionary with performance metrics
            web_data_quality: Dictionary with web data quality metrics
            hypothesis_improvement: Dictionary with hypothesis improvement metrics
        """
        try:
            self.metrics["cycle_duration"].append(performance_metrics["total_cycle_time"])
            self.metrics["web_data_quality"].append(web_data_quality)
            self.metrics["hypothesis_improvement"].append(hypothesis_improvement)
            
            # Update agent performance metrics
            for agent_name, execution_time in performance_metrics["agent_times"].items():
                if agent_name not in self.metrics["agent_performance"]:
                    self.metrics["agent_performance"][agent_name] = []
                
                self.metrics["agent_performance"][agent_name].append(execution_time)
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def _generate_llm_insights(self, previous_cycles, current_cycle):
        """
        Generate insights from LLM based on previous and current cycle data.
        
        Args:
            previous_cycles (list): List of previous research cycles
            current_cycle (dict): Current research cycle data
            
        Returns:
            dict: Insights, recommendations, and bottlenecks
        """
        try:
            # This is where your LLM processing would happen
            # For now, creating a safe default return structure
            insights = {
                "insights": [],
                "recommendations": [],
                "bottlenecks": []
            }
            
            # If you have previous cycles, maybe extract some patterns
            if previous_cycles and isinstance(previous_cycles, list):
                insights["insights"].append("System has completed previous research cycles")
            
            # Check for bottlenecks in current cycle
            if isinstance(current_cycle, dict) and "performance_metrics" in current_cycle:
                perf = current_cycle["performance_metrics"]
                if "bottlenecks" in perf and perf["bottlenecks"]:
                    for bottleneck in perf["bottlenecks"]:
                        if isinstance(bottleneck, dict) and "agent" in bottleneck and "percentage" in bottleneck:
                            insights["bottlenecks"].append(
                                f"{bottleneck['agent']} is a bottleneck ({bottleneck['percentage']:.1f}%)"
                            )
                            insights["recommendations"].append(
                                f"Optimize {bottleneck['agent']} performance"
                            )
            
            # If no specific insights were generated, provide defaults
            if not insights["insights"]:
                insights["insights"] = ["System is operating normally"]
            if not insights["recommendations"]:
                insights["recommendations"] = ["Continue monitoring system performance"]
            if not insights["bottlenecks"]:
                insights["bottlenecks"] = ["No significant bottlenecks detected"]
            
            return insights
        except Exception as e:
            logger.error(f"Error generating LLM insights: {e}")
            return {
                "insights": ["Error processing insights"],
                "recommendations": ["Check LLM service connection"],
                "bottlenecks": ["LLM processing error"]
            }
    
    def _prepare_llm_prompt(self, cycle_data, performance_metrics, previous_cycles):
        """
        Prepare the prompt for the Gemini LLM.
        
        Args:
            cycle_data: Dictionary containing cycle information
            performance_metrics: Dictionary with performance metrics
            previous_cycles: List of previous cycle data
            
        Returns:
            String prompt for the LLM
        """
        # Safely extract data with defaults
        cycle_id = cycle_data.get("cycle_id", "unknown")
        query = cycle_data.get("query", "unknown")
        
        # Safely extract hypothesis data
        hypothesis = cycle_data.get("hypothesis", {})
        hypothesis_text = "No hypothesis available"
        hypothesis_score = 0
        
        if isinstance(hypothesis, dict):
            hypothesis_text = hypothesis.get("text", "No hypothesis available")
            hypothesis_score = hypothesis.get("score", 0)
        
        # Format performance metrics safely
        total_cycle_time = performance_metrics.get("total_cycle_time", 0)
        successful_agents = ", ".join(performance_metrics.get("successful_agents", ["none"]))
        failed_agents = ", ".join(performance_metrics.get("failed_agents", ["none"]))
        bottlenecks = ", ".join([b.get("agent", "unknown") for b in performance_metrics.get("bottlenecks", [])])
        
        # Safely get web data metrics
        web_data = cycle_data.get("web_data", {})
        sources_count = 0
        successful_extractions = 0
        failed_extractions = 0
        
        if isinstance(web_data, dict):
            sources_count = web_data.get("sources_count", 0)
            successful_extractions = web_data.get("successful_extractions", 0)
            failed_extractions = web_data.get("failed_extractions", 0)
        
        # Create prompt
        prompt = f"""
        You are a Meta Review Agent in "The Second Mind" system that evaluates research cycles and provides feedback for improvement.
        
        CURRENT CYCLE INFORMATION:
        - Cycle ID: {cycle_id}
        - Query: {query}
        - Hypothesis: {hypothesis_text}
        - Hypothesis Score: {hypothesis_score}/10
        
        PERFORMANCE METRICS:
        - Total Cycle Time: {total_cycle_time:.2f} seconds
        - Successful Agents: {successful_agents}
        - Failed Agents: {failed_agents}
        - Bottlenecks: {bottlenecks}
        
        WEB DATA METRICS:
        - Sources Count: {sources_count}
        - Successful Extractions: {successful_extractions}
        - Failed Extractions: {failed_extractions}
        
        PREVIOUS CYCLES: {len(previous_cycles)}
        
        Based on the information above, please provide the following:
        1. Insights: Identify 3-5 key insights about the current cycle.
        2. Recommendations: Suggest 3-5 specific improvements for the next cycle.
        3. Bottlenecks: Identify the top 2-3 bottlenecks in the process.
        
        IMPORTANT: Format your response as strict JSON with the following structure:
        {
            "insights": ["insight1", "insight2", ...],
            "recommendations": ["recommendation1", "recommendation2", ...],
            "bottlenecks": ["bottleneck1", "bottleneck2", ...]
        }
        
        Do not include any commentary or explanations outside of the JSON.
        """
        
        return prompt
    
    def _parse_llm_response(self, llm_response):
        """
        Parse the LLM response into a structured format.
        
        Args:
            llm_response: String response from the LLM
            
        Returns:
            Dictionary with insights, recommendations, and bottlenecks
        """
        default_result = {
            "insights": [
                "No insights available due to LLM processing error"
            ],
            "recommendations": [
                "Review system logs for errors",
                "Check LLM integration"
            ],
            "bottlenecks": [
                "LLM processing"
            ]
        }
        
        if not llm_response:
            self.logger.warning("Empty LLM response received")
            return default_result
        
        try:
            # Extract JSON from the response
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                # Strip any markdown code block indicators
                json_str = llm_response[json_start:json_end]
                json_str = json_str.strip('`')
                
                # Replace any invalid escape sequences
                json_str = json_str.replace('\\', '\\\\')
                
                try:
                    result = json.loads(json_str)
                    
                    # Ensure the result has the expected structure
                    if not all(key in result for key in ["insights", "recommendations", "bottlenecks"]):
                        self.logger.warning("LLM response missing required keys")
                        return default_result
                    
                    return result
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error: {e}")
                    
                    # Try another approach - clean the JSON string
                    clean_json_str = json_str.replace('\n', ' ').replace('\t', ' ')
                    try:
                        result = json.loads(clean_json_str)
                        return result
                    except json.JSONDecodeError:
                        pass
            
            # If JSON parsing failed, try to parse manually
            insights = []
            recommendations = []
            bottlenecks = []
            
            lines = llm_response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if "insights" in line.lower():
                    current_section = "insights"
                elif "recommendations" in line.lower():
                    current_section = "recommendations"
                elif "bottlenecks" in line.lower():
                    current_section = "bottlenecks"
                elif current_section and (line.startswith("-") or line.startswith("*")):
                    item = line[1:].strip()
                    if current_section == "insights":
                        insights.append(item)
                    elif current_section == "recommendations":
                        recommendations.append(item)
                    elif current_section == "bottlenecks":
                        bottlenecks.append(item)
            
            if insights or recommendations or bottlenecks:
                return {
                    "insights": insights or ["No insights available"],
                    "recommendations": recommendations or ["No recommendations available"],
                    "bottlenecks": bottlenecks or ["No bottlenecks identified"]
                }
            
            return default_result
        except Exception as e:
            self.logger.error(f"Error processing LLM response: {e}")
            return default_result
        
    def review(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct a final meta-review of the processed research data.
        
        Args:
            data: Dictionary containing research data
            
        Returns:
            Dictionary with meta-review results
        """
        self.logger.info("Starting meta-review of research data")
        
        # Check input data type
        if not isinstance(data, dict):
            self.logger.warning("[WARNING] MetaReviewAgent received unexpected data format. Using fallback.")
            return {"meta_review_results": []}
        
        # Handle different data formats
        if isinstance(data, tuple) and len(data) > 0 and isinstance(data[0], dict):
            # If data is a tuple containing dictionaries, extract the first element
            data = data[0]
        
        # Extract proximity results safely
        proximity_results = data.get("proximity_results", [])
        
        # Check if the proximity results are valid
        if not isinstance(proximity_results, list):
            self.logger.error(f"Unexpected proximity_results format: {type(proximity_results)}")
            proximity_results = []
        
        # Extract items from proximity results
        meta_review_results = []
        
        for item in proximity_results:
            if isinstance(item, dict):
                # Properly evaluate the item
                meta_review_results.append({
                    "item": item,
                    "review_score": self._evaluate_proximity_score(item)
                })
            else:
                self.logger.warning(f"Unexpected item format in proximity_results: {type(item)}")
                meta_review_results.append({
                    "item": {"item": [], "proximity_score": 0.0},
                    "review_score": 0.0
                })
        
        # If no items were found, add default empty results
        if not meta_review_results:
            meta_review_results = [
                {"item": {"item": [], "proximity_score": 0.0}, "review_score": 0.0},
                {"item": {"item": [], "proximity_score": 0.0}, "review_score": 0.0}
            ]
        
        self.logger.info(f"Meta-review completed with {len(meta_review_results)} results")
        return {"meta_review_results": meta_review_results}

    def _evaluate_proximity_score(self, item: Dict[str, Any]) -> float:
        """
        Evaluate proximity score based on connections.
        
        Args:
            item: Dictionary containing proximity data
            
        Returns:
            Float score representing the proximity evaluation
        """
        # Safely extract proximity score
        if not item:
            return 0.0
        
        # Try different possible keys
        possible_keys = ["proximity_score", "score", "proximity", "connection_score"]
        
        for key in possible_keys:
            if key in item:
                try:
                    score = float(item[key])
                    return score
                except (ValueError, TypeError):
                    pass
        
        # If item contains a nested item with a score
        if "item" in item and isinstance(item["item"], dict):
            return self._evaluate_proximity_score(item["item"])
        
        return 0.0
    
    def update_context_item(self, context_id, data, client=None):
        """
        Update a context item, creating it first if it doesn't exist.
        
        Args:
            context_id (str): The ID of the context to update
            data (dict): The data to store in the context
            client: Optional MCPS client instance
            
        Returns:
            tuple: (success, item_id or error message)
        """
        try:
            if client is None:
                client = self.mcps_client
            
            # First, try to update the existing context
            response = client.update_context_item(context_id, data)
            
            # If 404, create the context first
            if hasattr(response, 'status_code') and response.status_code == 404:
                logger.info(f"Context item {context_id} not found, creating new item")
                
                # Create new context item
                create_response = client.create_context_item(data)
                
                if hasattr(create_response, 'status_code') and create_response.status_code == 200:
                    item_id = create_response.json().get('item_id')
                    logger.info(f"Successfully created context item with ID: {item_id}")
                    return True, item_id
                else:
                    error_msg = create_response.json().get('message') if hasattr(create_response, 'json') else str(create_response)
                    logger.error(f"Failed to create context item: {error_msg}")
                    return False, f"Failed to create context item: {error_msg}"
            
            # If update was successful
            elif hasattr(response, 'status_code') and response.status_code == 200:
                logger.info(f"Successfully updated context item {context_id}")
                return True, context_id
            
            # Other error
            else:
                error_msg = response.json().get('message') if hasattr(response, 'json') else str(response)
                logger.error(f"Failed to update context item: {error_msg}")
                return False, f"Failed to update context item: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error updating context item: {e}")
            return False, f"Error updating context item: {str(e)}"