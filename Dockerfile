# Airflow image with the pipeline's Python dependencies baked in
FROM apache/airflow:3.0.2
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
# pinning apache-airflow stops pip from upgrading Airflow while installing the extras
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements-airflow.txt
