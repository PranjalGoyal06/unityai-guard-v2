# unityai-guard-v2
UnityAI-Guard (IMPROVED) by Team LexicoGraphers (CS 613 NLP 2025-26, IITGN)

Setup and run
-----------

Install dependencies and run the labeling script:

```bash
python3 -m pip install -r requirements.txt
python3 bengali_labelling.py
```

The script will produce `bengali_labelled.csv` with four columns: `raw text`, `source`, `harmful/safe`, `harmful categories`.
