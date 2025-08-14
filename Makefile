.PHONY: install run ui docker-cli docker-ui

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && python -m src.cli --inp raw --out out --workers 8

ui:
	. .venv/bin/activate && streamlit run app.py

docker-cli:
	docker build -t id-mvp .
	docker run --rm -v "$$PWD/raw:/in" -v "$$PWD/out:/out" id-mvp --inp /in --out /out --workers 8

docker-ui:
	docker build -t id-mvp .
	docker run --rm -p 8501:8501 id-mvp streamlit run /app/app.py --server.address=0.0.0.0 --server.port=8501
