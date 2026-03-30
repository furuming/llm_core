run:
	uv run python3 src/main.py

post1:
	curl -X POST "http://localhost:9001/generate" -H "Content-Type: application/json" -d '{ "prompt": "日本語で自己紹介して","max_new_tokens": 1024,"temperature": 0.2 }'

post2:
	curl -X POST "http://localhost:9001/generate" -H "Content-Type: application/json" -d '{ "prompt": "バナナの実と種について解説して" }'

post3:
	curl -X POST "http://localhost:9001/generate" -H "Content-Type: application/json" -d '{ "prompt": "アキレスはカメに追いつける？","max_new_tokens": 200,"temperature": 0.2 }'
