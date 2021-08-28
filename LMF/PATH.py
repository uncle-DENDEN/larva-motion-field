from pathlib import Path
import os

cwd = os.getcwd()
imgpath = os.path.join(cwd, 'sample')
filepath = os.path.join(cwd, 'output')

Path(imgpath).mkdir(parents=True, exist_ok=True)
Path(filepath).mkdir(parents=True, exist_ok=True)
