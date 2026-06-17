# Pitch Analyzer deploy

## 推奨構成

公開する場合は Docker 対応のホスティングに `pitch_analyzer/` をデプロイします。
Render、Railway、Fly.io などで動かせます。HTTPS はホスティング側が提供するものを使います。

## 必要な設定

- Root directory: `pitch_analyzer`
- Build: Dockerfile
- Port: 環境変数 `PORT` を使う設定
- Health check path: `/`

## 注意

- スマホカメラは HTTPS が必要です。公開URLは必ず `https://...` で開いてください。
- 動画解析はCPUを使います。無料枠だと解析に時間がかかることがあります。
- `SAVE_CSV=1` を設定した場合だけ、解析CSVをサーバー側に保存します。公開環境では通常オフ推奨です。

## ローカルDocker確認

```bash
cd /home/hs15156/pitch_analyzer
docker build -t pitch-analyzer .
docker run --rm -p 8000:8000 pitch-analyzer
```

その後、PCで `http://127.0.0.1:8000` を開きます。
