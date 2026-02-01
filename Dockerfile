FROM apache/airflow:2.8.0

# Install system dependencies for LightGBM and other packages
USER root
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy and install Python requirements
COPY requirements-airflow.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

# Copy DAGs, source code and scripts
COPY --chown=airflow:airflow dags /opt/airflow/dags
COPY --chown=airflow:airflow src /opt/airflow/src
COPY --chown=airflow:airflow scripts /opt/airflow/scripts
COPY --chown=airflow:airflow config.yaml /opt/airflow/config.yaml