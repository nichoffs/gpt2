from model import GPT2

model = GPT2.load_checkpoint("Jul071039PM/10000")
print(model.generate("Once upon a time, Owen was ", num_return_sequences=1, max_length=50)[0])
