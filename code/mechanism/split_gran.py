import io, os
p=r"D:\CodexData\tmp\privtrim-review-20261015\v3_remote\v3_gran_report.txt"
d=r"D:\CodexData\tmp\privtrim-review-20261015\gran_parts"
os.makedirs(d, exist_ok=True)
n=0
for l in io.open(p,encoding="utf-8",errors="replace").read().split("\n"):
    s=l.strip()
    if s.split() and s.split()[0] in ("sim_emb","imp_emb","cons_imp_emb","random_adj"):
        n+=1
        io.open(os.path.join(d,"%02d.txt"%n),"w",encoding="utf-8").write(s+"\n")
print("wrote",n)
