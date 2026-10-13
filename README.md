# AX RAG 스타터 (문서 Q&A)

- 포털이 넣어 주는 전용 PostgreSQL(pgvector) `DATABASE_URL`에 문서를 저장합니다.
- 임베딩은 `ax-embed`(bge-m3, 1024차원), 답변은 `ax-local`을 씁니다.
- 로컬 개발: `docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16` 후
  `.env` 작성 → `pip install -r requirements.txt && uvicorn app:app --reload`
