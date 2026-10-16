#!/bin/bash
B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
echo "--- models dir ---"
ls -1 "$B/models" 2>&1
echo "--- Qwen snapshot ---"
ls -ld "$B"/models/models--Qwen--Qwen2.5-7B-Instruct 2>&1
ls -1 "$B"/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/ 2>&1
echo "--- config.json ---"
for d in "$B"/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ ; do
  echo "TRY=$d"; ls -l "$d" 2>&1 | head -6
done
echo "--- emb snapshot ---"
ls -1 "$B"/models/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/ 2>&1
echo "--- cases ---"
ls -l "$B"/run/qwen25_7b/cases.jsonl.gz 2>&1
echo "--- scorer ---"
ls -l "$B"/project/external/squad_v11/evaluate-v2.0.py 2>&1
echo "--- gpu partition detail ---"
sinfo -p gpu -o "%N %t %C %G %m" 2>&1
scontrol show partition gpu 2>&1 | grep -Ei 'AllowAccounts|AllowGroups|MaxTime|DefaultTime|QoS|TRES' | head -10
echo "--- 我的 assoc ---"
sacctmgr -nP show assoc where user=$(id -un) format=Account,Partition,QOS 2>&1
echo "ALMESBAR_DETAIL_DONE"
