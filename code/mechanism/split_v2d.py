import io, os
p=r"D:\CodexData\tmp\privtrim-review-20261015\v2d_remote\v2d_report.txt"
d=r"D:\CodexData\tmp\privtrim-review-20261015\v2d_parts"
os.makedirs(d, exist_ok=True)
lines=[l for l in io.open(p,encoding="utf-8",errors="replace").read().split("\n") if l.strip()]
n=0
for l in lines:
    s=l.strip()
    if s.startswith("ratio=") or (s.split() and s.split()[0] in ("sim_emb","sim_char3","cons_emb","random_adj","random_far")):
        n+=1
        io.open(os.path.join(d,"%02d.txt"%n),"w",encoding="utf-8").write(s+"\n")
print("wrote",n)
