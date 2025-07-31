import subprocess
from model import TuneModelNew

subprocess.run(["cargo", "run", "--", "tune"])
TuneModelNew()