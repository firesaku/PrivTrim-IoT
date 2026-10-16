import io, os
p=r"D:\CodexData\tmp\privtrim-review-20261015\v3e_remote\e2e3_report.txt"
d=r"D:\CodexData\tmp\privtrim-review-20261015\e3_parts"
os.makedirs(d, exist_ok=True)
n=0
for l in io.open(p,encoding="utf-8",errors="replace").read().split("\n"):
    s=l.strip()
    if not s: continue
    if s.startswith("E2E v3") or s.startswith("cases=") or s.startswith("dataset") or s.split()[0] in ("squad","hotpotqa","===","full","qtx","qtx_cluster","qtx_random"):
        n+=1
        io.open(os.path.join(d,"%02d.txt"%n),"w",encoding="utf-8").write(s+"\n")
print("wrote",n)
