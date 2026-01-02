# 1. Get the ID of your first running instance
ID=$(vastai show instances --raw | jq -r '.[0].id')

vastai start instance $ID

echo "waiting 30s for the instance to start"
sleep 30

# 2. Use that ID to connect (with the Mac-friendly xargs -o)
vastai ssh-url "$ID" | sed 's|ssh://root@||; s|:| |' | xargs -o -r bash -c 'ssh -tt -p $1 root@$0 -L 8080:localhost:8080 "tmux a || tmux"'
