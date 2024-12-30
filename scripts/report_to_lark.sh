MESSAGE=$1

curl -X POST -H "Content-Type: application/json" \
    -d "{\"msg_type\":\"text\",\"content\":{\"text\":\"$MESSAGE\"}}" \
    https://open.feishu.cn/open-apis/bot/v2/hook/b412d81e-93df-4898-8657-688e00a0bdc4
