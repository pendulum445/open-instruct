MESSAGE=$1

curl -X POST -H "Content-Type: application/json" \
    -d "{\"msg_type\":\"text\",\"content\":{\"text\":\"$MESSAGE\"}}" \
    https://open.feishu.cn/open-apis/bot/v2/hook/d960daf0-aba6-4552-8634-ab5f1bff41ca
