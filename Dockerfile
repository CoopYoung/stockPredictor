# Use an official miniconda image
FROM continuumio/miniconda3:latest
# Set the working directory in the container


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install any needed packages specified in environment.yml


RUN conda install -c conda-forge mamba -y

COPY environment.yml .

RUN mamba env create -f environment.yml

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
# Copy the requirements file into the container
#COPY requirements.txt .
WORKDIR /app
# Install any needed packages specified in requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 5000 available to the world outside this container
# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
RUN mamba clean --all --yes

EXPOSE 5000

CMD ["conda", "run", "-n", "myenv", "python", "app.py"]
