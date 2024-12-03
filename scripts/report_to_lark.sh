MESSAGE=$1

curl -X POST -H "Content-Type: application/json" \
    -d "{\"msg_type\":\"text\",\"content\":{\"text\":\"$MESSAGE\"}}" \
    https://open.feishu.cn/open-apis/bot/v2/hook/7c4537ae-638c-45f7-af86-6709ff65179e
