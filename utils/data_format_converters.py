import base64
from typing import Dict,Any
import cv2
import numpy as np


def convert_ndarray_to_serializable( data: Dict[str, Any]) -> Dict[str, Any]:
    processed_data = {}
    for key, value in data.items():
        if hasattr(value, 'shape'):  # It's an ndarray
            try:
                _, buffer = cv2.imencode('.jpg', value)
                processed_data[key] = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                # Log error but don't break the stream
                print(f"Failed to encode {key}: {str(e)}")
                processed_data[key] = None
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            processed_data[key] = convert_ndarray_to_serializable(value)
        elif isinstance(value, list):
            processed_data[key] = [
                convert_ndarray_to_serializable({'item': item})['item']
                if isinstance(item, dict)
                else base64.b64encode(cv2.imencode('.jpg', item)[1]).decode('utf-8')
                if hasattr(item, 'shape')
                else item
                for item in value
            ]
        else:
            processed_data[key] = value
    return processed_data


def encode_frame_to_base64( frame:np.ndarray) -> str:
    """Convert a single frame (ndarray) to base64 string."""
    if frame is None:
        return ""

    if isinstance(frame, str):  # Already encoded
        return frame

    if hasattr(frame, 'shape'):  # It's an ndarray
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Failed to encode frame: {str(e)}")
            return ""
    return ""
