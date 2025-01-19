FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
COPY . .

RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "stock-predictor", "/bin/bash", "-c"]

EXPOSE 8050

ENTRYPOINT ["conda", "run", "-n", "stock-predictor", "python", "app.py"]