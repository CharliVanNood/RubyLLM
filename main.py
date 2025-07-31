import subprocess
from model import TrainModelNew

subprocess.run(["cargo", "run", "--", "train"])
TrainModelNew()