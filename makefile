run:
	uv run python3 src/main.py

models:
	curl -X GET "http://localhost:9001/models"

post1:
	curl -X POST "http://localhost:9000/generate" -H "Content-Type: application/json" -d '{ "prompt": "日本語で自己紹介して","model_family":"gemma","max_new_tokens": 1024,"temperature": 0.2 }'

post2:
	curl -X POST "http://localhost:9000/generate" -H "Content-Type: application/json" -d '{ "prompt": "バナナの実と種について解説して","model_family":"llama" }'

post3:
	curl -X POST "http://localhost:9001/generate" -H "Content-Type: application/json" -d '{ "prompt": "アキレスはカメに追いつける？","model_family":"qwen","max_new_tokens": 200,"temperature": 0.2 }'
