# inference_with_tools.py
# Full script for inference with a fine-tuned Mistral-7B model, including tool call parsing, real execution, and agent loop.
# Enhanced with LRU caching for tool results and parallel execution.
# Usage: python inference_with_tools.py --model_path ./nomadic-mind-v1/final_merged_model --query "Advise on fitness habits using latest research."

import json
import re
import logging
import argparse
import requests
import ast
import sys
import unittest
import os
import io
import contextlib
import sqlite3
import hashlib
import concurrent.futures
from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from functools import lru_cache

# Set up verbose logging for traceability and debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Security configuration for file and REPL tools
class SecurityConfig:
    """Security settings for file access and REPL execution."""
    ALLOWED_READ_PATH = os.environ.get("AGENT_ALLOWED_READ_PATH", "/data/allowed/read")
    ALLOWED_WRITE_PATH = os.environ.get("AGENT_ALLOWED_WRITE_PATH", "/data/allowed/write")
    MAX_FILE_SIZE = 1_000_000  # 1MB
    REPL_TIMEOUT = 5  # seconds
    REPL_MAX_MEMORY = 100 * 1024 * 1024  # 100MB

    @classmethod
    def validate_file_path(cls, filepath: str, write_mode: bool = False) -> bool:
        """Validate file path against security restrictions."""
        abs_path = os.path.abspath(filepath)
        allowed_path = cls.ALLOWED_WRITE_PATH if write_mode else cls.ALLOWED_READ_PATH
        return abs_path.startswith(os.path.abspath(allowed_path))

    @classmethod
    def validate_file_size(cls, filepath: str) -> bool:
        """Check if file size is within limits."""
        return os.path.exists(filepath) and os.path.getsize(filepath) <= cls.MAX_FILE_SIZE

# Tool registry: Maps tool names to functions. Covers all tools from original tools.py.
TOOL_REGISTRY = {}

def register_tool(name: str):
    """Decorator for DRY tool registration."""
    def decorator(func):
        TOOL_REGISTRY[name] = func
        return func
    return decorator

@register_tool("search_web")
@lru_cache(maxsize=1000)
def search_web(query: str) -> str:
    """
    Real web search using DuckDuckGo (free, no API key). Cached by query.
    
    Args:
        query: Search query string (hashed for caching).
    
    Returns:
        Formatted search results or error message.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Invalid or missing 'query'")
    
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(query)}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        matches = re.findall(r'<a class="result-link" href="(.*?)">(.*?)</a>', response.text, re.DOTALL)
        results = [f"Title: {title.strip()}\nURL: {href.strip()}" for href, title in matches[:5]]
        return "\n".join(results) if results else "No search results found."
    except requests.RequestException as e:
        logger.error(f"Search web error: {e}")
        return f"Error during web search: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in search_web: {e}")
        return f"Unexpected error: {str(e)}"

@register_tool("calc_tool")
@lru_cache(maxsize=1000)
def calc_tool(query: str) -> str:
    """
    Safe calculation using ast.parse and eval. Cached by query.
    
    Args:
        query: Math expression string.
    
    Returns:
        Result as string or error.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Invalid or missing 'query'")
    
    try:
        tree = ast.parse(query, mode='eval')
        if not isinstance(tree.body, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Num)):
            raise ValueError("Unsafe or invalid expression")
        result = eval(compile(tree, filename='<string>', mode='eval'), {"__builtins__": {}}, {"math": __import__("math")})
        return str(result)
    except (SyntaxError, ValueError) as e:
        logger.error(f"Calc tool syntax/value error: {e}")
        return f"Invalid calculation: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in calc_tool: {e}")
        return f"Unexpected error: {str(e)}"

@register_tool("news_tool")
@lru_cache(maxsize=1000)
def news_tool(query: str) -> str:
    """
    Stub for news search (requires NewsAPI key; mocked). Cached by query.
    
    Args:
        query: News search query.
    
    Returns:
        Mocked news results.
    """
    logger.info(f"News tool called with query: {query}")
    return f"Mock news for '{query}': Breaking news on habits - consistency key to transformation."

@register_tool("python_repl")
def python_repl(code: str) -> str:
    """
    Executes Python code safely with restricted environment. Not cached due to dynamic output.
    
    Args:
        code: Python code string.
    
    Returns:
        Execution result or error.
    """
    if not code or not isinstance(code, str):
        raise ValueError("Invalid or missing 'code'")
    
    try:
        safe_globals = {"__builtins__": {}, "math": __import__("math"), "print": print}
        output = []
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, safe_globals, {})
            output.append(f.getvalue())
        return "\n".join(output) if output else "Code executed successfully."
    except Exception as e:
        logger.error(f"Python REPL error: {e}")
        return f"Error executing code: {str(e)}"

@register_tool("read_file")
def read_file(filepath: str) -> str:
    """
    Reads a file from allowed path with size restrictions. Not cached (file content may change).
    
    Args:
        filepath: Path to file.
    
    Returns:
        File contents or error.
    """
    if not filepath or not isinstance(filepath, str):
        raise ValueError("Invalid or missing 'filepath'")
    
    if not SecurityConfig.validate_file_path(filepath):
        raise ValueError(f"Access denied: Filepath {filepath} not in allowed directory")
    if not SecurityConfig.validate_file_size(filepath):
        raise ValueError(f"File {filepath} exceeds size limit or does not exist")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"File read error: {e}")
        return f"Error reading file: {str(e)}"

@register_tool("write_file")
def write_file(filepath: str, content: str) -> str:
    """
    Writes content to a file in allowed path. Not cached (write operation).
    
    Args:
        filepath: Path to file.
        content: Content to write.
    
    Returns:
        Success message or error.
    """
    if not filepath or not content or not isinstance(filepath, str) or not isinstance(content, str):
        raise ValueError("Invalid or missing 'filepath' or 'content'")
    
    if not SecurityConfig.validate_file_path(filepath, write_mode=True):
        raise ValueError(f"Access denied: Filepath {filepath} not in allowed write directory")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        logger.error(f"File write error: {e}")
        return f"Error writing file: {str(e)}"

@register_tool("calendar_tool")
@lru_cache(maxsize=1000)
def calendar_tool(action: str) -> str:
    """
    Stub for calendar operations (requires API; mocked). Cached by action.
    
    Args:
        action: Calendar action (e.g., 'create_event').
    
    Returns:
        Mocked result.
    """
    logger.info(f"Calendar tool called with action: {action}")
    return f"Mock calendar action '{action}' completed."

@register_tool("task_tracker_tool")
def task_tracker_tool(task_details: str) -> str:
    """
    Adds a task to a local SQLite database. Not cached (database state changes).
    
    Args:
        task_details: Task description.
    
    Returns:
        Success message or error.
    """
    if not task_details or not isinstance(task_details, str):
        raise ValueError("Invalid or missing 'task_details'")
    
    db_file = "tasks.db"
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_details TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("INSERT INTO tasks (task_details) VALUES (?)", (task_details,))
        conn.commit()
        task_id = cursor.lastrowid
        conn.close()
        return f"Task {task_id} added to tracker: '{task_details}' (Status: pending)"
    except sqlite3.Error as e:
        logger.error(f"SQLite error adding task: {e}")
        return f"Error adding task: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error adding task: {e}")
        return f"Unexpected error: {str(e)}"

@register_tool("job_search_tool")
@lru_cache(maxsize=1000)
def job_search_tool(query: str) -> str:
    """
    Job search using a free endpoint (mocked; real impl needs SerpAPI). Cached by query.
    
    Args:
        query: Job search query.
    
    Returns:
        Mocked job results.
    """
    if not query or not isinstance(query, str):
        raise ValueError("Invalid or missing 'query'")
    
    logger.info(f"Job search for: {query}")
    return f"Mock job listings for '{query}': Software Engineer at TechCorp, Data Analyst at DataInc."

@register_tool("get_current_weather")
@lru_cache(maxsize=1000)
def get_current_weather(location: str) -> str:
    """
    Stub for weather lookup (requires API; mocked). Cached by location.
    
    Args:
        location: Location for weather.
    
    Returns:
        Mocked weather data.
    """
    logger.info(f"Weather tool called for location: {location}")
    return f"Mock weather for {location}: Sunny, 25°C."

@register_tool("animal_medical_database")
@lru_cache(maxsize=1000)
def animal_medical_database(query: str) -> str:
    """
    Stub for animal medical data lookup (mocked). Cached by query.
    
    Args:
        query: Medical query.
    
    Returns:
        Mocked medical info.
    """
    logger.info(f"Animal medical database called with query: {query}")
    return f"Mock animal medical info for '{query}': Consult a veterinarian for accurate diagnosis."

def extract_tool_calls(model_output: str) -> List[Dict[str, Any]]:
    """
    Extracts tool call JSONs from the model's generated text.
    
    Args:
        model_output: Raw string output from the fine-tuned model.
    
    Returns:
        List of parsed tool call dicts, or empty if none found.
    """
    tool_calls = []
    json_matches = re.findall(r'\{.*?"tool_call".*?\}', model_output, re.DOTALL)
    for match in json_matches:
        try:
            tool_call = json.loads(match)
            if "tool_call" in tool_call and "name" in tool_call["tool_call"] and "arguments" in tool_call["tool_call"]:
                tool_calls.append(tool_call["tool_call"])
            else:
                logger.warning(f"Malformed tool call JSON: {match}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid tool JSON in output: {match} - Error: {e}")
    if tool_calls:
        logger.info(f"Extracted {len(tool_calls)} tool calls from output")
    return tool_calls

def agent_loop(
    user_query: str,
    model_pipeline: pipeline,
    max_iterations: int = 5,
    tool_parallel: bool = True
) -> str:
    """
    Runs an agent loop: Generate response, extract tools, execute (parallel with caching), re-prompt until no calls.
    
    Args:
        user_query: Initial user input.
        model_pipeline: Loaded Hugging Face pipeline for the fine-tuned model.
        max_iterations: Max loops to prevent infinity.
        tool_parallel: Execute tools in parallel using threads.
    
    Returns:
        Final integrated response.
    """
    current_input = user_query
    iteration = 0
    while iteration < max_iterations:
        logger.info(f"Agent loop iteration {iteration + 1}: Generating model response")
        try:
            output = model_pipeline(current_input, max_new_tokens=512, do_sample=False)[0]['generated_text']
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            return f"Error in model inference: {str(e)}"
        
        tool_calls = extract_tool_calls(output)
        if not tool_calls:
            logger.info("No more tool calls detected; returning final response")
            return output
        
        results = []
        if tool_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
                futures = []
                for call in tool_calls:
                    tool_name = call.get("name")
                    args = call.get("arguments", {})
                    if tool_name in TOOL_REGISTRY:
                        logger.info(f"Submitting tool {tool_name} for parallel execution")
                        try:
                            # Extract single arg for cached functions
                            if tool_name in ("search_web", "calc_tool", "news_tool", "calendar_tool", 
                                           "job_search_tool", "get_current_weather", "animal_medical_database"):
                                arg_key = next(iter(args))  # e.g., 'query', 'action', 'location'
                                futures.append((tool_name, executor.submit(TOOL_REGISTRY[tool_name], args[arg_key])))
                            else:
                                futures.append((tool_name, executor.submit(TOOL_REGISTRY[tool_name], **args)))
                        except Exception as e:
                            logger.error(f"Error preparing tool {tool_name}: {e}")
                            results.append(f"Tool {tool_name} error: {str(e)}")
                    else:
                        results.append(f"Unknown tool: {tool_name}")
                
                for tool_name, future in futures:
                    try:
                        result = future.result()
                        results.append(f"Tool {tool_name} result: {result}")
                    except Exception as e:
                        logger.error(f"Tool {tool_name} execution failed: {e}")
                        results.append(f"Tool {tool_name} error: {str(e)}")
        else:
            for call in tool_calls:
                tool_name = call.get("name")
                args = call.get("arguments", {})
                if tool_name in TOOL_REGISTRY:
                    try:
                        logger.info(f"Executing tool: {tool_name} with args: {args}")
                        if tool_name in ("search_web", "calc_tool", "news_tool", "calendar_tool", 
                                       "job_search_tool", "get_current_weather", "animal_medical_database"):
                            arg_key = next(iter(args))
                            result = TOOL_REGISTRY[tool_name](args[arg_key])
                        else:
                            result = TOOL_REGISTRY[tool_name](**args)
                        results.append(f"Tool {tool_name} result: {result}")
                    except ValueError as ve:
                        logger.warning(f"Tool {tool_name} validation error: {ve}")
                        results.append(f"Tool {tool_name} invalid args: {str(ve)}")
                    except Exception as e:
                        logger.error(f"Tool {tool_name} execution failed: {e}")
                        results.append(f"Tool {tool_name} error: {str(e)}")
                else:
                    logger.warning(f"Unknown tool requested: {tool_name}")
                    results.append(f"Unknown tool: {tool_name}")
        
        tool_results_str = "\n".join(results)
        current_input = f"{current_input}\nPrevious output: {output}\nTool results:\n{tool_results_str}\nNow integrate and continue:"
        
        iteration += 1
    
    logger.warning(f"Max iterations ({max_iterations}) reached; returning partial response")
    return output

def load_model_pipeline(model_path: str) -> pipeline:
    """
    Loads the fine-tuned Mistral model and tokenizer into a pipeline.
    
    Args:
        model_path: Path to the merged model directory.
    
    Returns:
        Hugging Face pipeline for text generation.
    """
    logger.info(f"Loading model from {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise

def main():
    """CLI entry point for running inference with tools."""
    parser = argparse.ArgumentParser(description="Run inference with tools on fine-tuned Mistral model.")
    parser.add_argument("--model_path", required=True, help="Path to the merged fine-tuned model")
    parser.add_argument("--query", required=True, help="User query to process")
    
    args = parser.parse_args()
    
    try:
        model_pipe = load_model_pipeline(args.model_path)
        final_response = agent_loop(args.query, model_pipe)
        print(f"Final Response:\n{final_response}")
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"Error: {str(e)}")

class TestToolIntegration(unittest.TestCase):
    """Unit tests for tool parsing, execution, caching, and agent loop behavior."""
    
    def test_extract_tool_calls_valid(self):
        """Test parsing valid tool JSON from output."""
        sample_output = 'Advice: {"tool_call": {"name": "search_web", "arguments": {"query": "test"}}}. More text.'
        calls = extract_tool_calls(sample_output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "search_web")
        self.assertEqual(calls[0]["arguments"]["query"], "test")
    
    def test_extract_tool_calls_invalid(self):
        """Test handling invalid or missing JSON."""
        sample_output = "No tools here. {invalid json}"
        calls = extract_tool_calls(sample_output)
        self.assertEqual(len(calls), 0)
    
    def test_search_web_execution(self):
        """Test real search_web with valid args."""
        result = search_web("python programming")
        self.assertIn("Title", result)
        self.assertNotIn("Error", result)
    
    def test_search_web_invalid_args(self):
        """Test search_web with invalid args."""
        with self.assertRaises(ValueError):
            search_web("")
    
    def test_search_web_cache(self):
        """Test LRU caching for search_web."""
        query = "python programming"
        result1 = search_web(query)
        result2 = search_web(query)
        self.assertEqual(result1, result2)
        self.assertEqual(search_web.cache_info().hits, 1)
    
    def test_calc_tool_valid(self):
        """Test calc_tool with safe expression."""
        result = calc_tool("2 + 3 * 4")
        self.assertEqual(result, "14")
    
    def test_calc_tool_unsafe(self):
        """Test calc_tool rejects unsafe code."""
        result = calc_tool("__import__('os').system('ls')")
        self.assertIn("Invalid", result)
    
    def test_calc_tool_cache(self):
        """Test LRU caching for calc_tool."""
        query = "2 + 3 * 4"
        result1 = calc_tool(query)
        result2 = calc_tool(query)
        self.assertEqual(result1, result2)
        self.assertEqual(calc_tool.cache_info().hits, 1)
    
    def test_python_repl_valid(self):
        """Test python_repl with safe code."""
        result = python_repl("print(2 + 2)")
        self.assertEqual(result, "4")
    
    def test_python_repl_unsafe(self):
        """Test python_repl rejects unsafe code."""
        result = python_repl("import os; os.system('ls')")
        self.assertIn("Error", result)
    
    def test_task_tracker_tool(self):
        """Test task_tracker_tool adds task."""
        result = task_tracker_tool("Test task")
        self.assertIn("Task", result)
        self.assertIn("added to tracker", result)
    
    def test_read_file_invalid_path(self):
        """Test read_file rejects invalid path."""
        with self.assertRaises(ValueError):
            read_file("/unauthorized/file.txt")
    
    def test_agent_loop_termination(self):
        """Test loop terminates without tools."""
        def mock_pipe(x):
            return [{"generated_text": "No tools needed."}]
        response = agent_loop("Test query", mock_pipe)
        self.assertEqual(response, "No tools needed.")
    
    def test_agent_loop_with_tool(self):
        """Test loop with one tool call and execution."""
        def mock_pipe(input):
            if "Tool results" in input:
                return [{"generated_text": "Integrated response."}]
            return [{"generated_text": '{"tool_call": {"name": "calc_tool", "arguments": {"query": "1+1"}}}' }]
        
        response = agent_loop("Calculate something", mock_pipe)
        self.assertEqual(response, "Integrated response.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        unittest.main(argv=sys.argv[:1] + sys.argv[2:])
    else:
        main()
