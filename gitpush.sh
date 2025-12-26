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

# 获取当前分支名
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# 推送当前分支到远程
echo "=== 执行 git push origin $CURRENT_BRANCH ==="
git push origin "$CURRENT_BRANCH" || { echo "❌ git push 失败"; exit 1; }
echo "✅ 推送成功！提交信息：$COMMIT_MSG"
