from model import GetResponse, GetModel

modelAndTokenizer = GetModel()
GetResponse(modelAndTokenizer[0], modelAndTokenizer[1], "spreek je ook nederlands of dat niet?", 512)
