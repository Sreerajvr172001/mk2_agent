import ollama

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_detections",
            "description": "Get current YOLO object detections from robot camera",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "stop_robot",
            "description": "Stop the robot immediately",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "Move robot forward for one second",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

response = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user",
               "content": "Check what the camera sees and decide if the robot should move"}],
    tools=tools
)

print(response["message"]["tool_calls"])