import subprocess
from model import TrainModelNew

result = subprocess.run(["cargo", "run"])
print("Process finished with return code:", result.returncode)
TrainModelNew()