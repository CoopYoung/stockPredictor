FROM continuumio/miniconda3

WORKDIR /app

RUN apt-get update && apt-get install -y curl

COPY environment.yml .
COPY assets/ ./assets/
COPY . .

RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "stock-predictor", "/bin/bash", "-c"]

EXPOSE 8050

ENV PYTHONUNBUFFERED=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8050/ || exit 1

#ENTRYPOINT ["conda", "run", "-n", "stock-predictor", "python", "app.py"]
ENTRYPOINT conda run -n stock-predictor python -u app.py 2>&1 | tee app.log