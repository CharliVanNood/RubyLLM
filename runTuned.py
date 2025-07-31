from model import GetResponse, GetModelTuned

modelAndTokenizer = GetModelTuned()
GetResponse(modelAndTokenizer[0], modelAndTokenizer[1], "this is a query, hi hi how are you?", 512)
