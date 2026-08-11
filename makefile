run:
	uv run python3 src/main.py

models:
	curl -X GET "http://localhost:9001/models"

post1:
	curl -X POST "http://localhost:9000/v1/chat/completions" -H "Content-Type: application/json" -d '{ "model": "gemma", "messages": [{"role": "user", "content": "日本語で自己紹介して"}], "max_tokens": 1024, "temperature": 0.2 }'

post2:
	curl -X POST "http://localhost:9000/v1/chat/completions" -H "Content-Type: application/json" -d '{ "model": "llama", "messages": [{"role": "user", "content": "バナナの実と種について解説して"}] }'

post3:
	curl -X POST "http://localhost:9001/v1/chat/completions" -H "Content-Type: application/json" -d '{ "model": "qwen", "messages": [{"role": "user", "content": "アキレスはカメに追いつける？"}], "max_tokens": 200, "temperature": 0.2 }'
