import io, os
p=r"D:\CodexData\tmp\privtrim-review-20261015\conclusions.txt"
d=r"D:\CodexData\tmp\privtrim-review-20261015\parts"
os.makedirs(d, exist_ok=True)
lines=[l for l in io.open(p,encoding="utf-8").read().split("\n") if l.strip()]
made=[]
for l in lines:
    if "CONC " in l:
        key=l.split("CONC ",1)[1].split(" :: ")[0].replace(" ","_").replace("@","r")
        fn=os.path.join(d,"conc_%s.txt"%key)
        io.open(fn,"w",encoding="utf-8").write(l+"\n")
        made.append(fn)
    if "METHOD_MEANS" in l:
        fn=os.path.join(d,"method_means.txt")
        io.open(fn,"w",encoding="utf-8").write(l+"\n")
        made.append(fn)
print("\n".join(made))
