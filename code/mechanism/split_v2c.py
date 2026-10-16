import io, os, re
p=r"D:\CodexData\tmp\privtrim-review-20261015\v2c_remote\v2c_report.txt"
d=r"D:\CodexData\tmp\privtrim-review-20261015\v2c_parts"
os.makedirs(d, exist_ok=True)
lines=io.open(p,encoding="utf-8",errors="replace").read().split("\n")
n=0
for l in lines:
    if l.startswith("ratio=") or ("== ALL ==" in l and l.strip()):
        n+=1
        io.open(os.path.join(d,"%02d.txt"%n),"w",encoding="utf-8").write(l.strip()+"\n")
print("wrote",n,"parts")
for i in range(1,n+1):
    print(os.path.join(d,"%02d.txt"%i))
