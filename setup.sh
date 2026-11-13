#!/usr/bin/env bash
# Prepares folders and the .env file before the first docker compose up.
# Usage: ./setup.sh            first-time setup, keeps an existing .env
#        ./setup.sh --reset    also removes containers, the database, logs, and artifacts
set -euo pipefail

if [[ "${1:-}" == "--reset" ]]; then
  docker compose down -v --remove-orphans
  rm -rf ./logs ./artifacts
fi

mkdir -p ./logs ./artifacts

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

# the containers run as this user so files they create stay editable on the host
if grep -q '^AIRFLOW_UID=' .env; then
  sed -i.bak "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env && rm -f .env.bak
else
  echo "AIRFLOW_UID=$(id -u)" >> .env
fi

# every Airflow container must share these, or task tokens fail signature verification
for key in AIRFLOW_JWT_SECRET AIRFLOW_SECRET_KEY; do
  if ! grep -q "^${key}=." .env; then
    value=$(openssl rand -hex 32)
    if grep -q "^${key}=" .env; then
      sed -i.bak "s/^${key}=.*/${key}=${value}/" .env && rm -f .env.bak
    else
      echo "${key}=${value}" >> .env
    fi
  fi
done

echo "Setup complete. Next: docker compose up -d --build"
