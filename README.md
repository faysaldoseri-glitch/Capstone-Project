Bahrain Budget App 💰
A personal budget tracking app I built for my data science project. It reads bank SMS messages, scans receipts, and uses ML to predict how much I'll spend by the end of the month.
Built specifically for Bahrain (supports BHD, Arabic + English, local banks).
What it does

SMS Parsing — paste your bank SMS alerts and it extracts the amount, merchant, date, and category automatically
Receipt Scanning — take a photo of a receipt and it reads it using PaddleOCR (works offline)
Voice Input — say "I spent 5 dinars at Starbucks" and it logs it
ML Predictions — trained on 12 months of real spending data, predicts end-of-month spending
Telegram Alerts — sends you a message when you're close to your budget limit
Editable Table — fix wrong categories or merchant names directly in the app

The ML Part
I trained a Ridge Regression + Random Forest ensemble on my own spending data (923 transactions over 12 months).

X (features): day of month, cumulative spend, daily rate, remaining days, previous month total, 3-month rolling average, etc. (9 features total)
Y (target): remaining spend — how much more I'll spend until month end
R² = 0.87 with ~7% average prediction error

The model also has a sanity check — if the ML prediction is way off from the simple daily pace calculation, it blends both together instead of showing a wrong number.
How to run
pip install streamlit pandas scikit-learn joblib paddlepaddle paddleocr
Then:
cd budget
python train_model.py       # train the model (only need to do this once)
python import_history.py    # load your SMS history into the database
streamlit run budget.py     # start the app
Files
FileWhat it doesbudget.pyMain Streamlit appparser.pyExtracts transactions from SMS, receipts, voicepredictions.pyMakes end-of-month predictions using the trained modeltrain_model.pyTrains the Ridge + RF ensembleconfig.pyAll settings, regex patterns, category rulesdb.pySQLite database layerservices.pyTelegram, OCR, voice transcriptionimport_history.pyLoads CSV spending history into the database
Tech stack

Python, Streamlit
scikit-learn (Ridge, Random Forest, ensemble)
PaddleOCR for receipt scanning
SQLite for storage
HuggingFace Whisper for voice
Telegram Bot API for alerts
