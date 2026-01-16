import json

tools_def = [
    {
        "type": "function",
        "function": {
            "name": "key",
            "description": "Performs key down presses on the arguments passed in order, then performs key releases in reverse order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keys to press in order (e.g., ['ctrl', 'c'])"
                    }
                },
                "required": ["keys"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type",
            "description": "Type a string of text on the keyboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text string to type"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mouse_move",
            "description": "Move the cursor to a described element on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_description": {
                        "type": "string",
                        "description": "The description of the element to move the cursor to"
                    }
                },  
                "required": ["element_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "left_click",
            "description": "Click the left mouse button at a described element on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_description": {
                        "type": "string",
                        "description": "The description of the element to click the left mouse button at   "
                    }
                },
                "required": ["element_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "left_click_drag",
            "description": "Click and drag the cursor to a described element on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_element_description": {
                        "type": "string",
                        "description": "The description of the element to start drag the cursor from"
                    },
                    "end_element_description": {
                        "type": "string",
                        "description": "The description of the element to drag the cursor to"
                    }
                },
                "required": ["end_element_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "right_click",
            "description": "Click the right mouse button at a described element on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_description": {
                        "type": "string",
                        "description": "The description of the element to click the right mouse button at"
                    }
                },
                "required": ["element_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "middle_click",
            "description": "Click the middle mouse button at a described element on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_description": {
                        "type": "string",
                        "description": "The description of the element to click the middle mouse button at"
                    }
                },
                "required": ["element_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "double_click",
            "description": "Double-click the left mouse button at a described element on the screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_description": {
                        "type": "string",
                        "description": "The description of the element to double-click the left mouse button at"
                    }
                },
                "required": ["element_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "triple_click",
            "description": "Triple-click the left mouse button at a described element on the screen (simulated as double-click since it's the closest action).",
            "parameters": {
                "type": "object",
                "properties": {
                    "element_description": {
                        "type": "string",
                        "description": "The description of the element to triple-click the left mouse button at    "
                    }
                },
                "required": ["element_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Performs a scroll of the mouse scroll wheel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pixels": {
                        "type": "number",
                        "description": "The amount of pixels to scroll (positive for down, negative for up)"
                    }
                },
                "required": ["pixels"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hscroll",
            "description": "Performs a horizontal scroll (mapped to regular scroll).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pixels": {
                        "type": "number",
                        "description": "The amount of pixels to scroll horizontally"
                    }
                },
                "required": ["pixels"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait specified seconds for the change to happen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time": {
                        "type": "number",
                        "description": "The number of seconds to wait"
                    }
                },
                "required": ["time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "terminate",
            "description": "Terminate the current task and report its completion status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["success", "failure"],
                        "description": "The status of the task completion"
                    }
                },
                "required": ["status"]
            }
        }
    }
]

SYSTEM_PROMPT_QWEN_3 = """You are a helpful assistant that can understand screenshots in the images and take actions in a computer environment to achieve the task. 
## Tools

You may call the tools defined below to assist with the given task.

Here are some tips for using the tools:
- Use a mouse and keyboard to interact with a computer, and take screenshots.",
- This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
- Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
- If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools_def) + """
</tools>

## Response format

For each step, you must produce exactly two components in the following order:

1. An action description:
   - A single sentence that begins with "Action: "
   - It should briefly and clearly describe what you intend to do in the UI.

2. A <tool_call> block:
   - This block must contain only one JSON object with the structure:
     {"name": "<function-name>", "arguments": <args-json-object>}
   - Wrap this JSON inside <tool_call>...</tool_call>.
   - Do not include any extra text or explanation inside the block.

""" + f"""## Action Rules:
- The system is running on a x86_64 ubuntu system.
- Chrome is the default browser that have been installed for you to use.
- The current working directory is /home/user.
- The password for the user is "password". Use it when you need to authenticate or use sudo commands.
- To invoke tool that requires `element_description` argument, you only need to provide a textual description of the element such as "the top-left corner of the search bar".
- Do not output anything else outside the action and tool call blocks.
- Leave all windows and applications open after completing the task.
- If finishing, call the terminate tool. 
    - Issue the status as success if the task is completed successfully
    - Issue the status as failure if the task is infeasible to complete due to environment constraints. 
"""

SYSTEM_PROMPT_GTA_1 = """"""