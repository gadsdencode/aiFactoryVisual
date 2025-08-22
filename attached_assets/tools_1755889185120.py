import logging
from typing import Dict, Callable
from transformers import pipeline

from inference_with_tools import extract_tool_calls  # For inference with fine-tuned model

# Example tool implementations (stubbed; replace with real logic)
TOOL_REGISTRY: Dict[str, Callable] = {
    "search_web": lambda args: f"Mock search results for query: {args.get('query', '')}",  # Real: use requests/SerpAPI
    "calc_tool": lambda args: eval(args.get('query', '0'))  # Real: safer parser like sympy
    # Add more from your tools list
}

def agent_loop(
    user_query: str,
    model_pipeline: pipeline,  # Hugging Face pipeline with your fine-tuned model
    max_iterations: int = 3
) -> str:
    """
    Runs an agent loop: Generate, extract tools, execute, re-prompt until no more calls.
    
    Args:
        user_query: Initial user input.
        model_pipeline: Loaded inference pipeline for the fine-tuned Mistral model.
        max_iterations: Prevent infinite loops.
    
    Returns:
        Final response after tool integrations.
    """
    current_input = user_query
    for iteration in range(max_iterations):
        # Generate response
        output = model_pipeline(current_input)[0]['generated_text']
        
        # Extract and execute tools
        tool_calls = extract_tool_calls(output)
        if not tool_calls:
            return output  # No more tools; return final response
        
        results = []
        for call in tool_calls:
            tool_name = call.get("name")
            if tool_name in TOOL_REGISTRY:
                try:
                    result = TOOL_REGISTRY[tool_name](call.get("arguments", {}))
                    results.append(f"Tool {tool_name} result: {result}")
                except Exception as e:
                    logging.error(f"Tool {tool_name} execution failed: {e}")
                    results.append(f"Tool {tool_name} error: {str(e)}")
            else:
                results.append(f"Unknown tool: {tool_name}")
        
        # Append results to input for next iteration
        current_input = f"{current_input}\n{output}\nTool results:\n" + "\n".join(results)
    
    logging.warning("Max iterations reached; returning partial response")
    return output
