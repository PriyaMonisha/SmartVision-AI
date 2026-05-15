# Rule: FAST_MODE

## In training notebooks (04_train_classifier.py, 05_yolo_training.py, etc.)
FAST_MODE is a LOCAL variable at the top of the notebook.
It is passed EXPLICITLY as a parameter to training functions.
It does NOT change config.FAST_MODE.

```python
# ================================================================
FAST_MODE = True   # LOCAL variable — flip to False for production
# ================================================================
train_model(MODEL, fast_mode=FAST_MODE)  # pass explicitly
```

## In config.py / Docker
config.FAST_MODE reads from the FAST_MODE environment variable.
For Docker: set `FAST_MODE=false` in docker-compose.yml environment.

## Never
- Never `import config; config.FAST_MODE = False` from notebook scope
- Never rely on config.FAST_MODE being mutable from notebooks
