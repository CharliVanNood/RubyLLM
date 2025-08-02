from model import GetResponse, GetModelTuned
import time

modelAndTokenizer = GetModelTuned()
GetResponse(modelAndTokenizer[0], modelAndTokenizer[1], "this is a query, hi hi how are you?[STA]", 512)
time.sleep(5)
GetResponse(modelAndTokenizer[0], modelAndTokenizer[1], "hi how are you?[STA]", 512)
time.sleep(5)
GetResponse(modelAndTokenizer[0], modelAndTokenizer[1], "good morning[STA]", 512)
time.sleep(5)
GetResponse(modelAndTokenizer[0], modelAndTokenizer[1], "what day is it?[STA]", 512)
time.sleep(5)
GetResponse(modelAndTokenizer[0], modelAndTokenizer[1], "how are you doing?[STA]", 512)
time.sleep(5)
