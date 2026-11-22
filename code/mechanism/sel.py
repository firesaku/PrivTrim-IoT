import io, random, string
p=r"D:\CodexData\tmp\privtrim-review-20261015\conclusions.txt"
lines=[l for l in io.open(p,encoding="utf-8").read().split("\n") if l.strip()]
random.seed(7)
for l in lines:
    if "CONC " in l or "METHOD_MEANS" in l:
        pre="".join(random.choice(string.ascii_letters+string.digits) for _ in range(28))
        print(pre+" ||| "+l)
