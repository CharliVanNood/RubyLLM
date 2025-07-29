from model import GetResponse, GetModel

modelAndTokenizer = GetModel()
GetResponse(modelAndTokenizer[0], modelAndTokenizer[1], "this is a query, hi hi how are you?", 512)
