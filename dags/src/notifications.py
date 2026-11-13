"""
Run notifications: email through SMTP when configured, and always a line in a notification log.
"""

import logging
import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from src import config

logger = logging.getLogger(__name__)

NOTIFICATION_LOG = "notifications.log"
SMTP_TIMEOUT_SECONDS = 30
SMTP_ENV_KEYS = ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "NOTIFY_EMAIL_TO")


def smtp_settings():
    """Return SMTP settings from environment variables, or None if email is not configured."""
    if not all(os.environ.get(key) for key in SMTP_ENV_KEYS):
        return None
    return {
        "host": os.environ["SMTP_HOST"],
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ["SMTP_USER"],
        "password": os.environ["SMTP_PASSWORD"],
        "sender": os.environ.get("SMTP_FROM", os.environ["SMTP_USER"]),
        "recipient": os.environ["NOTIFY_EMAIL_TO"],
    }


def send_notification(subject: str, body: str, artifact_dir=None) -> str:
    """
    Record a notification and email it when SMTP is configured.

    An email failure is logged rather than raised, so a mail problem never fails the pipeline.

    Returns
    -------
    channel : str
        "email" if the email was sent, otherwise "log".
    """
    artifact_dir = artifact_dir or config.ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(artifact_dir / NOTIFICATION_LOG, "a") as f:
        f.write(f"{timestamp} | {subject} | {body}\n")

    settings = smtp_settings()
    if settings is None:
        logger.info("notification (email not configured): %s", subject)
        return "log"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings["sender"]
    message["To"] = settings["recipient"]
    message.set_content(body)
    try:
        with smtplib.SMTP(settings["host"], settings["port"],
                          timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.starttls()
            server.login(settings["user"], settings["password"])
            server.send_message(message)
    except (smtplib.SMTPException, OSError) as err:
        logger.warning("email notification failed: %s", err)
        return "log"
    return "email"


def describe_context(context: dict) -> str:
    """Summarize an Airflow callback context in one line."""
    ti = context.get("ti") or context.get("task_instance")
    dag_run = context.get("dag_run")
    part_list = [
        f"dag={getattr(dag_run, 'dag_id', None) or getattr(ti, 'dag_id', 'unknown')}",
        f"run={getattr(dag_run, 'run_id', None) or getattr(ti, 'run_id', 'unknown')}",
    ]
    if ti is not None and getattr(ti, "task_id", None):
        part_list.append(f"task={ti.task_id} try={getattr(ti, 'try_number', '?')}")
    if context.get("exception"):
        part_list.append(f"error={context['exception']}")
    return ", ".join(part_list)


def task_failure_callback(context: dict) -> None:
    """Notify when a task fails after exhausting its retries."""
    send_notification("Airflow task failed", describe_context(context))


def dag_success_callback(context: dict) -> None:
    """Notify when a pipeline run succeeds."""
    send_notification("Airflow pipeline succeeded", describe_context(context))


def dag_failure_callback(context: dict) -> None:
    """Notify when a pipeline run fails."""
    send_notification("Airflow pipeline failed", describe_context(context))
