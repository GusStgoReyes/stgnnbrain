import json
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self, file_path: str = '../../config.json', **kwargs):
        with open(file_path, 'r') as file:
            data = json.load(file)
        for key, value in data.items():
            setattr(self, key, value)

        # add any additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)