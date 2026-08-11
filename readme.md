# llm_core

## WSL2 + Docker Compose で起動

Docker Desktop の WSL integration を有効にし、リポジトリを WSL2 の Linux
ファイルシステム（例: `~/src/llm_core`）に置いて実行します。NVIDIA GPU を使う
ため、ホスト側には WSL2 対応 NVIDIA ドライバーも必要です。

```bash
cp .env.example .env
docker compose build
docker compose run --rm --service-ports llm-core python src/main.py
```

Compose はプロジェクト全体を `/app` に bind mount します。そのため、WSL2 側で
編集したソースはコンテナを再ビルドせずに反映されます。依存関係を変更した場合は
`docker compose build` を再実行してください。コンテナ内のプロセスは WSL2 の
標準ユーザーとファイルを共有しやすい `app` ユーザー（UID/GID `1000`）で実行されます。

Hugging Face がダウンロードしたモデルはホストの `./models`（コンテナ内の
`/models`）に保存されます。コンテナを作り直しても再利用され、ホストからも直接
確認できます。モデルを削除する場合は `./models` 内の対象ファイルを削除してください。

Dockerfile には開発用の起動コマンドを固定していません。Compose は TTY を有効に
しているため、必要なコマンドを `docker compose run` で指定できます。

GPU を使わず CPU のみで確認する場合は、`compose.yml` の `gpus: all` を削除して
起動してください。

## レイヤーごとの責務

### Domain
業務ルールを持つビジネスロジックの中心
- 比較対象モデル
- メッセージ
- 生成パラメータ

### Application
ユースケース層
- 同じプロンプトを複数モデルに送る
- 結果を整形する
- 

## Infrastructure
外部依存を実装する層です。
- vLLM OpenAI 互換 API 呼び出し
- DB保存
- ログ出力
- 設定読み込み

## API/Presentation
HTTPリクエストの入出力の変換

### `/v1/chat/completions`

`POST /v1/chat/completions` は OpenAI Chat Completions と同じ基本的なメッセージ形式を受け取ります。

```json
{
  "model": "gemma",
  "messages": [
    {"role": "system", "content": "簡潔に回答してください。"},
    {"role": "user", "content": "富士山の高さは？"}
  ],
  "max_tokens": 256,
  "temperature": 0.2,
  "top_p": 1.0
}
```

レスポンスも `chat.completion` の形式で、`choices` とトークンの `usage` を返します。
`model` には `/models` の `family` または `model_name` を指定できます。



## 全体像
```mermaid
flowchart TB

%% =========================
%% Client / Entry
%% =========================
Client[Client / Browser / CLI]

%% =========================
%% Presentation Layer
%% =========================
subgraph PRESENTATION["Presentation / API Layer"]
    FastAPI[FastAPI Router\n/compare]
    DI[Dependency Injection\npresentation/dependencies.py]
    ReqRes[Request / Response Schemas]
end

%% =========================
%% Application Layer
%% =========================
subgraph APPLICATION["Application Layer"]
    UseCase[CompareModelsUseCase]
end

%% =========================
%% Domain Layer
%% =========================
subgraph DOMAIN["Domain Layer"]
    Ports[Ports\nChatGateway\nResultRepository]
    Entities[Entities / Value Objects\nMessage\nGenerationConfig\nModelTarget\nInferenceResult]
end

%% =========================
%% Infrastructure Layer
%% =========================
subgraph INFRASTRUCTURE["Infrastructure Layer"]
    VllmGateway[VllmOpenAIGateway]
    ResultRepo[InMemoryResultRepository\nor DB Repository]
    Settings[Settings / Config]
end

%% =========================
%% External Systems
%% =========================
subgraph EXTERNAL["External Systems"]
    VLLM[vLLM OpenAI-Compatible Server]
    DB[(SQLite / Postgres / S3\noptional)]
end

%% =========================
%% Main request flow
%% =========================
Client --> FastAPI
FastAPI --> ReqRes
FastAPI --> DI
DI --> Settings
DI --> UseCase
DI --> VllmGateway
DI --> ResultRepo

UseCase --> Entities
UseCase --> Ports

VllmGateway --> Ports
ResultRepo --> Ports

VllmGateway --> VLLM
ResultRepo --> DB

%% =========================
%% Notes on dependency direction
%% =========================
classDef presentation fill:#E3F2FD,stroke:#1565C0,color:#0D47A1;
classDef application fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20;
classDef domain fill:#FFF8E1,stroke:#F9A825,color:#6D4C41;
classDef infra fill:#F3E5F5,stroke:#6A1B9A,color:#4A148C;
classDef external fill:#ECEFF1,stroke:#546E7A,color:#263238;

class Client external;
class FastAPI,DI,ReqRes presentation;
class UseCase application;
class Ports,Entities domain;
class VllmGateway,ResultRepo,Settings infra;
class VLLM,DB external;
```

## シーケンスのイメージ
```mermaid
sequenceDiagram

participant Client
participant API
participant UseCase
participant Gateway as ChatGateway
participant vLLM

Client->>API: POST /compare
API->>UseCase: execute()

loop モデルごと
    UseCase->>Gateway: generate()
    Gateway->>vLLM: /v1/chat/completions
    vLLM-->>Gateway: response
    Gateway-->>UseCase: InferenceResult
end

UseCase-->>API: results
API-->>Client: JSON
```
