#!/bin/bash

# 检查是否输入了提交信息
if [ -z "$1" ]; then
    echo "用法：./gitpush.sh '你的提交信息'"
    exit 1
fi

COMMIT_MSG="$1"

# 执行 git add
echo "=== 执行 git add . ==="
git add . || { echo "❌ git add 失败"; exit 1; }

# 执行 git commit
echo "=== 执行 git commit -m '$COMMIT_MSG' ==="
git commit -m "$COMMIT_MSG" || { echo "❌ git commit 失败"; exit 1; }

# 执行 git push
echo "=== 执行 git push origin Atol-Server ==="
git push origin Atol-Server || { echo "❌ git push 失败"; exit 1; }

echo "✅ 推送成功！提交信息：$COMMIT_MSG"