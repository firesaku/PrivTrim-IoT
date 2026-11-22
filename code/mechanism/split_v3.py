import io, os
d=r"D:\CodexData\tmp\privtrim-review-20261015\v3_remote"
out=r"D:\CodexData\tmp\privtrim-review-20261015\v3_parts"
os.makedirs(out, exist_ok=True)
n=0
for c in ("scatter","guard","gran","bound"):
    p=os.path.join(d,"v3_%s_report.txt"%c)
    if not os.path.exists(p): continue
    for l in io.open(p,encoding="utf-8",errors="replace").read().split("\n"):
        s=l.strip()
        if not s or s.startswith("V3 CONFIG") or s.startswith("mode") or s.startswith("读法"): continue
        if s.split() and s.split()[0] in ("sim_emb","cons_emb","imp_emb","cons_imp_emb","random_adj","random_far","bound_emb","rand4_emb"):
            n+=1
            io.open(os.path.join(out,"%s_%02d.txt"%(c,n)),"w",encoding="utf-8").write(s+"\n")
print("wrote",n)
