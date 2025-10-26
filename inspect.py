# inspect_model.py
import joblib, json, os
from pprint import pprint

MODEL_PATH = "lang_detector_pipeline_v2_papluca_all20.joblib"  # or ./lang_detector_pipeline_v2_papluca_all20.joblib

def safe_print(x):
    try:
        pprint(x)
    except Exception:
        print(repr(x)[:1000])

if not os.path.exists(MODEL_PATH):
    print("Model file not found at", MODEL_PATH)
    raise SystemExit(1)

print("Loading model (this may take a few seconds)...")
model = joblib.load(MODEL_PATH)
print("Loaded model type:", type(model))
print()

print("=== repr(model) (first 1000 chars) ===")
print(repr(model)[:1000])
print()

# Common places to look for classes and pipeline steps
print("=== Checking for attributes ===")
attrs = dir(model)
candidates = [a for a in attrs if ("classes" in a or "steps" in a or "predict" in a or "transform" in a)]
safe_print(candidates)
print()

# model.classes_
if hasattr(model, "classes_"):
    print("model.classes_ found:")
    safe_print(model.classes_)

# If pipeline-like
if hasattr(model, "steps"):
    print("Pipeline steps present. Steps and types:")
    for name, est in model.steps:
        print(" -", name, ":", type(est), "-", est.__class__.__name__)
        if hasattr(est, "classes_"):
            print("    classes_ at step", name, "->", getattr(est, "classes_"))
        # check for predict_proba on each step (usually final)
        print("    has predict_proba?:", hasattr(est, "predict_proba"))
    final = model.steps[-1][1]
    print()
    print("Final estimator type:", type(final), "-", final.__class__.__name__)
    print("Final estimator has predict_proba?:", hasattr(final, "predict_proba"))
    if hasattr(final, "classes_"):
        print("Final estimator classes_:", getattr(final, "classes_"))

# If the pipeline stores named_steps:
if hasattr(model, "named_steps"):
    print("named_steps keys:", list(model.named_steps.keys()))

# get_params sample
try:
    params = model.get_params()
    print("Sample parameter keys from model.get_params():")
    safe_print(list(params.keys())[:40])
except Exception as e:
    print("Could not get params:", e)

# Try sample prediction (uses a short sample text)
test_text = ["This is a sample to test prediction."]
print()
print("=== Attempting a sample predict() on one sentence ===")
try:
    pred = model.predict(test_text)
    print("predict output:", pred)
except Exception as e:
    print("predict failed:", e)

print()
print("=== Attempting predict_proba (if available) ===")
try:
    if hasattr(model, "predict_proba"):
        print("model.predict_proba exists. Trying it...")
        proba = model.predict_proba(test_text)
        print("predict_proba shape:", None if proba is None else (len(proba), len(proba[0])))
    else:
        # try to access final estimator
        if hasattr(model, "steps"):
            final = model.steps[-1][1]
            print("Checking final estimator:", type(final))
            print("final.predict_proba?:", hasattr(final, "predict_proba"))
            if hasattr(final, "predict_proba"):
                proba = final.predict_proba(model.named_steps.get(list(model.named_steps.keys())[0]).transform(test_text) if hasattr(model.named_steps.get(list(model.named_steps.keys())[0]), 'transform') else test_text)
                print("predict_proba succeeded (shape info above).")
except Exception as e:
    print("predict_proba attempt failed:", e)

print()
print("=== Finished inspection ===")
