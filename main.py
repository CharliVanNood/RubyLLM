import subprocess
from model import ContinueModelNew, TrainModelNew

subprocess.run(["cargo", "run", "--", "train"])
TrainModelNew()
#ContinueModelNew()
