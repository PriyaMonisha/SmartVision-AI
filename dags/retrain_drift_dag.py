"""
dags/retrain_drift_dag.py
─────────────────────────
SmartVision AI — Drift Detection and Retraining Trigger DAG

Schedule : Daily at 02:00 UTC (low-traffic window)
Trigger  : Manual trigger also supported via Airflow UI (catchup=False)

Pipeline
────────
check_drift_alert → log_drift_classes → trigger_retrain → notify_complete
       │
       └─ AirflowSkipException if no drift (downstream tasks skipped cleanly)
          notify_complete always runs (trigger_rule="all_done")

Production replacements
───────────────────────
trigger_retrain BashOperator → KubernetesPodOperator / GoogleCloudRunJobOperator
                               / webhook to Colab / Vertex AI custom job
notify_complete print         → SlackWebhookOperator / EmailOperator / PagerDuty
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowSkipException  # module-level import (not inside fn)
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# ── Constants ──────────────────────────────────────────────────────────────────
FASTAPI_URL = "http://fastapi:8000"
DRIFT_STATUS_ENDPOINT = f"{FASTAPI_URL}/drift/status"
REQUEST_TIMEOUT_SECONDS = 10

# ── Default arguments ──────────────────────────────────────────────────────────
default_args = {
    "owner": "smartvision",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,   # no SMTP configured in local dev
    "email_on_retry": False,
}


# ── Task callables ─────────────────────────────────────────────────────────────

def check_drift_alert(**context) -> dict:
    """
    Poll GET /drift/status. Raises AirflowSkipException when no class has alert=1
    so the retraining pipeline is bypassed cleanly. Pushes the full status
    payload to XCom for downstream tasks.

    Expected API response shape:
        {"classes": [{"class_name": "cat", "alert": 1, "ks_statistic": 0.3, ...}]}
    Adjust key names to match your actual /drift/status schema if different.
    """
    # Import inside function — 'requests' is not an Airflow dep.
    # Module-level import would slow DAG parsing (scheduler re-parses every 30s).
    import requests  # noqa: PLC0415

    log = logging.getLogger(__name__)

    try:
        response = requests.get(DRIFT_STATUS_ENDPOINT, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to FastAPI at {DRIFT_STATUS_ENDPOINT}. "
            "Verify the fastapi service is running and joined to the 'smartvision' Docker network."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Request to {DRIFT_STATUS_ENDPOINT} timed out after {REQUEST_TIMEOUT_SECONDS}s."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Drift status endpoint returned HTTP {exc.response.status_code}: {exc.response.text}"
        ) from exc

    payload = response.json()
    log.info("Drift status payload: %s", json.dumps(payload, indent=2))

    # XCom push — payload is a plain dict (JSON-serialisable) ✓
    context["ti"].xcom_push(key="drift_payload", value=payload)

    classes = payload.get("classes", [])
    alerted = [c for c in classes if c.get("alert", 0) == 1]

    if not alerted:
        log.info(
            "No drift detected across %d monitored class(es). Skipping retraining.",
            len(classes),
        )
        raise AirflowSkipException("No drift detected — retraining pipeline skipped.")

    log.warning(
        "Drift detected in %d of %d class(es): %s",
        len(alerted),
        len(classes),
        [c.get("class_name", "<unknown>") for c in alerted],
    )
    return payload


def log_drift_classes(**context) -> None:
    """Pull drift payload from XCom and emit a structured drift report to the task log."""
    log = logging.getLogger(__name__)

    payload = context["ti"].xcom_pull(task_ids="check_drift_alert", key="drift_payload")
    if not payload:
        log.warning("No drift payload in XCom — check_drift_alert may have been skipped.")
        return

    classes = payload.get("classes", [])
    drifted = [c for c in classes if c.get("alert", 0) == 1]
    healthy = [c for c in classes if c.get("alert", 0) == 0]

    separator = "=" * 60
    log.info(separator)
    log.info("DRIFT REPORT  %s UTC", datetime.utcnow().strftime("%Y-%m-%d %H:%M"))
    log.info(separator)
    log.info("Classes monitored : %d", len(classes))
    log.info("Classes drifted   : %d", len(drifted))
    log.info("Classes healthy   : %d", len(healthy))

    if drifted:
        log.warning("DRIFTED CLASSES:")
        for cls in drifted:
            log.warning(
                "  %-20s  KS=%-8s  p=%-10s",
                cls.get("class_name", "<unknown>"),
                cls.get("ks_statistic", "N/A"),
                cls.get("p_value", "N/A"),
            )

    log.info(separator)


def notify_complete(**context) -> None:
    """
    Emit a pipeline completion summary. Always runs (trigger_rule="all_done").
    Handles upstream skip and failure gracefully via the payload check.

    Production: replace with SlackWebhookOperator, EmailOperator, or PagerDuty.
    """
    log = logging.getLogger(__name__)

    dag_run = context["dag_run"]
    payload = context["ti"].xcom_pull(task_ids="check_drift_alert", key="drift_payload")
    classes = payload.get("classes", []) if payload else []
    drifted = [c.get("class_name", "?") for c in classes if c.get("alert", 0) == 1]

    summary = {
        "dag_id": dag_run.dag_id,
        "run_id": dag_run.run_id,
        "execution_date": context["execution_date"].isoformat(),
        "drifted_classes": drifted,
        "retrain_triggered": len(drifted) > 0,
        "pipeline_status": "COMPLETE",
    }

    log.info("Pipeline summary: %s", json.dumps(summary, indent=2))

    # TODO: Replace with real notification
    # Slack example (requires apache-airflow-providers-slack):
    #   from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
    #   SlackWebhookOperator(
    #       task_id='slack_notify', slack_webhook_conn_id='slack_default',
    #       message=json.dumps(summary)
    #   ).execute(context)
    print(f"[SmartVision] Retraining pipeline complete. Summary: {json.dumps(summary)}")


# ── DAG definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id="smartvision_retrain_drift",
    description="Daily drift check with conditional retraining trigger for SmartVision AI",
    doc_md=__doc__,           # renders module docstring in Airflow UI
    default_args=default_args,
    schedule="0 2 * * *",    # 02:00 UTC daily; set to None for manual-trigger-only
    start_date=datetime(2024, 1, 1),
    catchup=False,            # no backfill — only run on next scheduled time
    max_active_runs=1,        # prevent concurrent retrain jobs
    tags=["smartvision", "drift", "retraining", "ml"],
) as dag:

    check_drift = PythonOperator(
        task_id="check_drift_alert",
        python_callable=check_drift_alert,
    )

    log_drift = PythonOperator(
        task_id="log_drift_classes",
        python_callable=log_drift_classes,
    )

    trigger_retrain = BashOperator(
        task_id="trigger_retrain",
        # Jinja renders {{ }} before bash executes — xcom_pull is a valid Jinja call in BashOperator
        bash_command=(
            "echo 'SmartVision: retraining triggered' && "
            "echo 'Drifted classes from XCom: "
            "{{ ti.xcom_pull(task_ids=\"check_drift_alert\", key=\"drift_payload\") }}' && "
            "echo 'TODO: replace this with a real trigger:' && "
            "echo '  gcloud ai custom-jobs create --display-name smartvision-retrain ...' && "
            "echo '  OR: curl -X POST $COLAB_WEBHOOK_URL' && "
            "echo '  OR: kubectl apply -f jobs/retrain-job.yaml'"
        ),
    )

    notify = PythonOperator(
        task_id="notify_complete",
        python_callable=notify_complete,
        trigger_rule="all_done",   # runs even when upstream tasks are skipped or failed
    )

    # Linear pipeline: skip propagates from check_drift → log_drift → trigger_retrain
    # notify always runs regardless (all_done)
    check_drift >> log_drift >> trigger_retrain >> notify
