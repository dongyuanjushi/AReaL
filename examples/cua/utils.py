import base64
import ast
from typing import Union

def encode_image(image: Union[bytes, str]) -> str:
    if isinstance(image, bytes):
        base_64_str = base64.b64encode(image).decode("utf-8")
        return "data:image/jpeg;base64," + base_64_str
    elif isinstance(image, str):
        bytes_obj = bytes_literal_to_bytesio(image)
        base_64_str = base64.b64encode(bytes_obj).decode("utf-8")
        return "data:image/jpeg;base64," + base_64_str
    else:
        raise ValueError("type of screenshot is not supported, only bytes or str is supported")

def bytes_literal_to_bytesio(bytes_literal_str):
    bytes_obj = ast.literal_eval(bytes_literal_str)

    if not isinstance(bytes_obj, bytes):
        raise ValueError("not a valid bytes literal")

    return bytes_obj