
name = "100_人"

sent = "MjnzmdknWGVlRqnEhpcPYINnMKYUankcbNfXeoCxaOviKQilLhNHMziTJaLYikXPZFnPgUjXPHHaPONYneEqtlYQFRRKWdjYTypu"

received = ""

with open(f"results/{name}.txt", "r", encoding='utf-8') as f:
    received = f.read()

rate = 0
for i in range(len(sent)):
    if sent[i] == received[i]:
        rate += 1

rate /= len(sent)

rate *= 100

print(f"Rate: {rate:.2f} %")