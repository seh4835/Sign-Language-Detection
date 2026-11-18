FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-api.txt .
RUN python -m pip install -r requirements-api.txt \
    && addgroup --system app \
    && adduser --system --ingroup app app

COPY src/api/app.py src/api/schemas.py ./src/api/
COPY src/models/__init__.py src/models/temporal_baseline.py \
    src/models/yolo_pose_hands_lstm_v2.py ./src/models/
COPY models/yolo_pose_hands_lstm_v2/webcam_baseline_artifacts/ \
    ./models/yolo_pose_hands_lstm_v2/webcam_baseline_artifacts/

RUN chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
