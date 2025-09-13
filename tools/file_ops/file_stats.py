"""
Example file statistics tool
Shows how to create tools in the new system
"""

from pathlib import Path
from typing import Dict, Any

def get_tool_metadata() -> Dict[str, Any]:
    """Return tool metadata"""
    return {
        'name': 'file_stats',
        'description': 'Get statistics about a file or directory',
        'category': 'file_ops',
        'parameters': {
            'path': {
                'required': True,
                'type': 'str',
                'description': 'Path to file or directory'
            }
        },
        'examples': ['file_stats(path="config.json")'],
        'permissions': ['file_read']
    }

def execute(path: str) -> Dict[str, Any]:
    """Execute the tool"""
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    stats = file_path.stat()
    
    result = {
        'path': str(file_path),
        'size_bytes': stats.st_size,
        'modified': stats.st_mtime,
        'is_directory': file_path.is_dir(),
        'is_file': file_path.is_file()
    }
    
    if file_path.is_dir():
        items = list(file_path.iterdir())
        result['item_count'] = len(items)
        result['subdirectories'] = len([i for i in items if i.is_dir()])
        result['files'] = len([i for i in items if i.is_file()])
    
    return result
