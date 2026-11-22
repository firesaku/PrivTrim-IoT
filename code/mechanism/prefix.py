import io, uuid
p=r"D:\CodexData\tmp\privtrim-review-20261015\conclusions.txt"
lines=io.open(p,encoding="utf-8").read().split("\n")
out=[]
for i,l in enumerate(lines):
    if l.strip():
        out.append("%s %s" % (uuid.uuid4().hex[:12], l))
    else:
        out.append(l)
io.open(p,"w",encoding="utf-8").write("\n".join(out))
print("prefixed", len(lines))
