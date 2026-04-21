"""
app.py – Bahrain Budget App v4.0
Run:  streamlit run app.py
"""

from __future__ import annotations
import tempfile
from datetime import datetime, date

import streamlit as st
import pandas as pd

from db       import init_db, save_transactions, load_transactions, clear_transactions, load_transactions_with_id, update_transaction, delete_transaction
from parser   import parse_sms_block, parse_receipt, parse_voice_text, normalize_arabic
from services import (
    send_telegram, format_summary, maybe_send_budget_alert,
    ocr_image, transcribe_audio,
)
from config   import SAMPLE_SMS, BOT_TOKEN, CHAT_ID, CATEGORY_RULES
from predictions import predict_eom_spend, generate_insights, detect_category_drift

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
init_db()
st.set_page_config(page_title="Budget App · Bahrain", page_icon="💰", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0ea5e9 100%);
    border-radius: 18px; padding: 2rem 2.5rem; margin-bottom: 1.2rem;
    color: white; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60%; right: -15%;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(14,165,233,0.3) 0%, transparent 70%);
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.25rem 0; }
.hero p  { opacity: 0.75; font-size: 0.9rem; margin: 0; }
.mc {
    background: linear-gradient(145deg, #f8fafc, #f1f5f9);
    border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 1.1rem 1.3rem; text-align: center;
    transition: transform 0.15s, box-shadow 0.15s;
}
.mc:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.06); }
.mc .lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 0.2rem; }
.mc .val { font-size: 1.45rem; font-weight: 700; color: #0f172a; }
.mc .val.blue   { color: #0ea5e9; }
.mc .val.green  { color: #10b981; }
.mc .val.red    { color: #ef4444; }
.mc .val.purple { color: #8b5cf6; }
.sh {
    font-size: 1.1rem; font-weight: 700; color: #1e293b;
    border-left: 4px solid #0ea5e9; padding-left: 0.75rem;
    margin: 1.8rem 0 0.7rem 0;
}
.stProgress > div > div > div > div { background: linear-gradient(90deg, #0ea5e9, #06b6d4); }
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #94a3b8 !important; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────
for k, v in {"selected_month": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ──────────────────────────────────────────────────────────
def mc(label, value, css=""):
    c = f"val {css}" if css else "val"
    st.markdown(f'<div class="mc"><div class="lbl">{label}</div><div class="{c}">{value}</div></div>', unsafe_allow_html=True)

def sh(title):
    st.markdown(f'<div class="sh">{title}</div>', unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df_existing = load_transactions()
available_months: list[str] = []
if not df_existing.empty:
    df_existing["date"] = pd.to_datetime(df_existing["date"])
    available_months = sorted(df_existing["date"].dt.to_period("M").astype(str).unique().tolist())

default_month = datetime.today().strftime("%Y-%m")
if available_months:
    default_month = available_months[-1]
if st.session_state.selected_month is None:
    st.session_state.selected_month = default_month

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    monthly_budget = st.number_input("Monthly budget (BHD)", min_value=0.0, value=300.0, step=10.0)
    st.session_state.selected_month = st.selectbox(
        "Active month",
        options=available_months or [default_month],
        index=(available_months.index(st.session_state.selected_month)
               if st.session_state.selected_month in available_months else 0),
    )

    st.divider()
    if st.button("🗑 Reset all data", use_container_width=True):
        clear_transactions()
        st.session_state.selected_month = datetime.today().strftime("%Y-%m")
        st.rerun()

    st.divider()
    st.markdown("##### 📨 Telegram")
    if st.button("Send Summary", use_container_width=True, key="tg_btn"):
        msg = format_summary(load_transactions(), monthly_budget, st.session_state.selected_month)
        code, _ = send_telegram(msg)
        st.success("Sent ✅") if code == 200 else st.error(f"Failed ({code})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HERO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    '<div class="hero"><h1>💰 Bahrain Budget App</h1>'
    '<p>Your personal spending tracker — SMS, receipts & voice in one place</p></div>',
    unsafe_allow_html=True,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADD TRANSACTIONS (on main page, compact, each auto-saves)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sh("➕  Add Transaction")

tab_quick, tab_voice, tab_ocr, tab_sms = st.tabs(["✏️ Quick Add", "🎤 Voice", "📷 Receipt", "📩 Bulk SMS"])

# ── Quick Add: manual entry with date picker ─────────────────────────
with tab_quick:
    qc1, qc2, qc3 = st.columns([1, 1, 1.5])
    with qc1:
        q_date = st.date_input("Date", value=date.today(), key="q_date")
    with qc2:
        q_amount = st.number_input("Amount (BHD)", min_value=0.0, step=0.5, key="q_amount")
    with qc3:
        q_merchant = st.text_input("Merchant / Description", placeholder="e.g. Starbucks", key="q_merchant")

    if st.button("💾 Save Entry", type="primary", use_container_width=True, key="quick_save"):
        if q_amount > 0 and q_merchant.strip():
            from parser import categorize, extract_item_name
            row = pd.DataFrame([{
                "date": q_date,
                "amount_bhd": round(q_amount, 3),
                "merchant": q_merchant.strip(),
                "category": categorize(q_merchant.strip(), q_merchant.strip()),
                "raw_sms": f"Manual: {q_amount} BHD at {q_merchant}",
                "item_name": extract_item_name(q_merchant.strip(), q_merchant.strip()),
                "qty": 1,
                "source_type": "manual",
                "store_name": q_merchant.strip(),
                "currency": "BHD",
            }])
            save_transactions(row)
            st.success(f"Saved: {q_amount:.3f} BHD at {q_merchant} ✅")
            st.rerun()
        else:
            st.warning("Enter an amount and merchant name.")

# ── Voice: record → transcribe → auto-save ───────────────────────────
with tab_voice:
    vc1, vc2 = st.columns([2, 1])
    with vc1:
        recorded_audio = st.audio_input("🎙️ Record or upload audio")
    with vc2:
        v_lang = st.selectbox("Language", ["English", "Arabic"], key="v_lang")
        v_date = st.date_input("Date", value=date.today(), key="v_date")

    uploaded_audio = st.file_uploader("Or upload a file", type=["mp3", "wav", "webm", "ogg"], key="voice_upload")

    if st.button("🎤 Transcribe & Save", type="primary", use_container_width=True, key="voice_save"):
        try:
            audio_path = None
            if recorded_audio is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(recorded_audio.read()); audio_path = tmp.name
            elif uploaded_audio is not None:
                suffix = "." + uploaded_audio.name.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_audio.read()); audio_path = tmp.name
            else:
                st.warning("Record or upload audio first.")
                st.stop()

            lang_code = "ar" if v_lang == "Arabic" else "en"
            with st.spinner("Transcribing..."):
                text = transcribe_audio(audio_path, language=lang_code)

            st.info(f"Heard: **{text}**")

            # Auto-parse and save
            voice_df = parse_voice_text(text, date_override=v_date)
            if not voice_df.empty:
                n = save_transactions(voice_df)
                st.success(f"Saved {n} transaction(s) ✅")
                st.rerun()
            else:
                st.warning(f"Could not parse a transaction from: '{text}'. Try the Quick Add tab instead.")
        except Exception as e:
            st.error(f"Error: {e}")

# ── OCR: upload → extract → auto-save ────────────────────────────────
with tab_ocr:
    uploaded_image = st.file_uploader("Upload receipt photo", type=["png", "jpg", "jpeg"], key="ocr_upload")
    if uploaded_image:
        img_bytes = uploaded_image.read()
        st.image(img_bytes, use_container_width=True)
        if st.button("📷 Extract & Save", type="primary", use_container_width=True, key="ocr_save"):
            try:
                with st.spinner("Reading receipt..."):
                    extracted = ocr_image(img_bytes)
                if extracted.strip():
                    receipt_df, confidence = parse_receipt(extracted)
                    if not receipt_df.empty:
                        n = save_transactions(receipt_df)
                        st.success(f"Saved {n} item(s) ✅")
                        if confidence == "medium":
                            st.warning("Prices may be approximate — total is correct.")
                        preview = [c for c in ["item_name", "amount_bhd", "category"] if c in receipt_df.columns]
                        st.dataframe(receipt_df[preview], hide_index=True)
                        st.rerun()
                    else:
                        st.warning("No items found in receipt.")
                else:
                    st.warning("OCR returned empty text.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Bulk SMS: paste multiple lines ────────────────────────────────────
with tab_sms:
    st.caption("Paste bank SMS messages — one per line.")

    # Load sample BEFORE the text_area is instantiated
    if st.button("Load sample", use_container_width=True, key="load_sample"):
        st.session_state["sms_area"] = SAMPLE_SMS
        st.rerun()

    sms_text = st.text_area("SMS text", height=180, placeholder="Paste your bank SMS alerts here...", key="sms_area")

    if st.button("💾 Parse & Save", type="primary", use_container_width=True, key="sms_save"):
        if sms_text.strip():
            df_new = parse_sms_block(sms_text)
            if df_new.empty:
                st.warning("No transactions found.")
            else:
                n = save_transactions(df_new)
                st.success(f"Saved {n} transactions ✅")
                st.rerun()
        else:
            st.warning("Paste some SMS text first.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DASHBOARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
df = load_transactions()

if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    valid_months = sorted(df["date"].dt.to_period("M").astype(str).unique().tolist())
    if valid_months and st.session_state.selected_month not in valid_months:
        st.session_state.selected_month = valid_months[-1]

month_pick = st.session_state.selected_month

alert = maybe_send_budget_alert(df, monthly_budget, month_pick)
if alert:
    st.info(alert)

if df.empty:
    st.info("👋 No transactions yet — use the tabs above to add entries.")
    st.stop()

# ── Filters ──────────────────────────────────────────────────────────
sh("🔍  Filter & Explore")

all_categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
all_merchants  = ["All"] + sorted(df["merchant"].dropna().unique().tolist())

fc1, fc2, fc3 = st.columns(3)
with fc1:
    view_mode = st.selectbox("View by", ["Month", "Year", "All Time", "Custom Range"], key="vm")
with fc2:
    cat_filter = st.selectbox("Category", all_categories, key="cf")
with fc3:
    merch_filter = st.selectbox("Merchant", all_merchants, key="mf")

dff = df.copy()
dff["month"] = dff["date"].dt.to_period("M").astype(str)
dff["year"]  = dff["date"].dt.year.astype(str)

if view_mode == "Month":
    dff = dff[dff["month"] == month_pick]
    period_label = month_pick
elif view_mode == "Year":
    sel_year = st.selectbox("Year", sorted(dff["year"].unique().tolist(), reverse=True), key="ys")
    dff = dff[dff["year"] == sel_year]
    period_label = sel_year
elif view_mode == "Custom Range":
    dc1, dc2 = st.columns(2)
    with dc1:
        d_start = st.date_input("From", value=dff["date"].min().date(), key="ds")
    with dc2:
        d_end = st.date_input("To", value=dff["date"].max().date(), key="de")
    dff = dff[(dff["date"].dt.date >= d_start) & (dff["date"].dt.date <= d_end)]
    period_label = f"{d_start} → {d_end}"
else:
    period_label = "All Time"

if cat_filter != "All":
    dff = dff[dff["category"] == cat_filter]
if merch_filter != "All":
    dff = dff[dff["merchant"] == merch_filter]

# ── KPIs ─────────────────────────────────────────────────────────────
sh("📊  Overview")

total     = float(dff["amount_bhd"].sum()) if not dff.empty else 0.0
txn_count = len(dff)
avg_txn   = total / txn_count if txn_count > 0 else 0.0
remaining = monthly_budget - total if view_mode == "Month" else None

k1, k2, k3, k4 = st.columns(4)
with k1: mc("Total Spent", f"{total:.3f} BHD", "blue")
with k2: mc("Transactions", str(txn_count), "purple")
with k3: mc("Avg / Transaction", f"{avg_txn:.3f} BHD")
with k4:
    if remaining is not None:
        mc("Remaining", f"{remaining:.3f} BHD", "green" if remaining >= 0 else "red")
    else:
        mc("Period", period_label)

if view_mode == "Month" and monthly_budget > 0:
    pct = min(max(total / monthly_budget, 0), 1)
    st.progress(pct)
    if remaining is not None and remaining < 0:
        st.error("🚨 Over budget this month!")
    elif pct >= 0.8:
        st.warning("⚠️ Approaching your budget limit.")

# ── Predictions ──────────────────────────────────────────────────────
if view_mode == "Month":
    sh("🔮  Insights & Predictions")
    prediction = predict_eom_spend(df, month_pick, monthly_budget)

    if prediction:
        p1, p2, p3 = st.columns(3)
        with p1: mc("Predicted End of Month", f"{prediction['predicted_eom']:.0f} BHD", "purple")
        with p2:
            if prediction["days_until_budget_out"] is not None:
                d = prediction["days_until_budget_out"]
                mc("Days Until Budget Runs Out", f"{d:.0f} days", "red" if d < 7 else ("blue" if d < 15 else "green"))
            else:
                mc("Days Until Budget Runs Out", "N/A")
        with p3: mc("Daily Pace", f"{prediction['daily_rate']:.1f} BHD/day")

        st.caption(f"Model: {prediction['model_used']} · Confidence: {prediction['confidence']} · {prediction['pct_elapsed']:.0f}% of month elapsed")

    for txt in generate_insights(df, month_pick, monthly_budget):
        st.markdown(f"&nbsp;&nbsp;{txt}")

# ── Charts ───────────────────────────────────────────────────────────
if not dff.empty:
    sh("📈  Spending Breakdown")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**By Category**")
        st.bar_chart(dff.groupby("category")["amount_bhd"].sum().sort_values(ascending=True))
    with ch2:
        st.markdown("**Spending Trend**")
        if view_mode in ("Month", "Custom Range"):
            st.line_chart(dff.groupby(dff["date"].dt.date)["amount_bhd"].sum().sort_index())
        else:
            st.bar_chart(dff.groupby(dff["month"])["amount_bhd"].sum().sort_index())

    sh("🏆  Top Spenders")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**By Category**")
        cat_sum = dff.groupby("category")["amount_bhd"].sum()
        cat_cnt = dff.groupby("category")["amount_bhd"].count()
        cat_tbl = pd.DataFrame({"total_bhd": cat_sum, "Transactions": cat_cnt}).reset_index()
        cat_tbl = cat_tbl.sort_values("total_bhd", ascending=False)
        cat_tbl["% share"] = (cat_tbl["total_bhd"] / cat_tbl["total_bhd"].sum() * 100).round(1)
        st.dataframe(cat_tbl, use_container_width=True, hide_index=True)
    with t2:
        st.markdown("**By Merchant**")
        m_sum = dff.groupby("merchant")["amount_bhd"].sum()
        m_cnt = dff.groupby("merchant")["amount_bhd"].count()
        m_tbl = pd.DataFrame({"total_bhd": m_sum, "Transactions": m_cnt}).reset_index()
        m_tbl = m_tbl.sort_values("total_bhd", ascending=False).head(15)
        m_tbl["% share"] = (m_tbl["total_bhd"] / m_tbl["total_bhd"].sum() * 100).round(1)
        st.dataframe(m_tbl, use_container_width=True, hide_index=True)

    if view_mode in ("Year", "All Time"):
        sh("📅  Monthly Comparison")
        mt = dff.groupby("month", as_index=False)["amount_bhd"].sum().sort_values("month")
        mt.columns = ["Month", "Spent (BHD)"]
        st.bar_chart(mt.set_index("Month"))
        if monthly_budget > 0:
            mt["Budget"] = monthly_budget
            mt["Status"] = mt["Spent (BHD)"].apply(lambda x: "✅ Under" if x <= monthly_budget else "🚨 Over")
            st.dataframe(mt, use_container_width=True, hide_index=True)

    sh("📋  Transaction Log")

    # Load transactions with IDs for editing
    edit_df = load_transactions_with_id()
    if not edit_df.empty:
        edit_df["date"] = pd.to_datetime(edit_df["date"])
        edit_df["edit_month"] = edit_df["date"].dt.to_period("M").astype(str)

        # Apply same filters as dashboard
        if view_mode == "Month":
            edit_df = edit_df[edit_df["edit_month"] == month_pick]
        elif view_mode == "Year":
            edit_df = edit_df[edit_df["date"].dt.year.astype(str) == sel_year] if 'sel_year' in dir() else edit_df
        elif view_mode == "Custom Range":
            edit_df = edit_df[(edit_df["date"].dt.date >= d_start) & (edit_df["date"].dt.date <= d_end)] if 'd_start' in dir() else edit_df
        if cat_filter != "All":
            edit_df = edit_df[edit_df["category"] == cat_filter]
        if merch_filter != "All":
            edit_df = edit_df[edit_df["merchant"] == merch_filter]

    if edit_df.empty:
        st.info("No transactions to display.")
    else:
        all_app_categories = sorted(CATEGORY_RULES.keys())

        # Prepare display dataframe with a select checkbox
        table_df = edit_df[["id", "date", "amount_bhd", "merchant", "item_name", "category", "source_type"]].copy()
        table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")
        table_df.insert(0, "Select", False)

        edited = st.data_editor(
            table_df,
            use_container_width=True,
            height=400,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Select": st.column_config.CheckboxColumn("✓", help="Select rows to delete", default=False),
                "id": None,  # hidden
                "date": st.column_config.TextColumn("Date", disabled=True),
                "amount_bhd": st.column_config.NumberColumn("Amount (BHD)", disabled=True, format="%.3f"),
                "merchant": st.column_config.TextColumn("Merchant", help="Edit the merchant name"),
                "item_name": st.column_config.TextColumn("Item Name", help="Edit the item name"),
                "category": st.column_config.SelectboxColumn(
                    "Category",
                    options=all_app_categories,
                    help="Pick the correct category",
                ),
                "source_type": st.column_config.TextColumn("Source", disabled=True),
            },
            key="txn_editor",
        )

        # Action buttons
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("💾 Save Changes", type="primary", use_container_width=True, key="save_edits"):
                changes = 0
                for idx in edited.index:
                    if idx not in table_df.index:
                        continue
                    old_row = table_df.loc[idx]
                    new_row = edited.loc[idx]
                    row_id = int(old_row["id"])

                    old_merchant = str(old_row["merchant"])
                    old_item = str(old_row["item_name"])
                    old_cat = str(old_row["category"])
                    new_merchant = str(new_row["merchant"])
                    new_item = str(new_row["item_name"])
                    new_cat = str(new_row["category"])

                    if old_merchant != new_merchant or old_cat != new_cat or old_item != new_item:
                        update_transaction(row_id, new_merchant, new_cat, new_item)
                        changes += 1

                if changes > 0:
                    st.success(f"Updated {changes} transaction(s) ✅")
                    st.rerun()
                else:
                    st.info("No changes detected.")

        with btn2:
            selected_ids = []
            for idx in edited.index:
                if idx in table_df.index and edited.loc[idx, "Select"]:
                    selected_ids.append(int(table_df.loc[idx, "id"]))

            if st.button(
                f"🗑 Delete Selected ({len(selected_ids)})" if selected_ids else "🗑 Delete Selected",
                use_container_width=True,
                key="delete_selected",
                disabled=len(selected_ids) == 0,
            ):
                for rid in selected_ids:
                    delete_transaction(rid)
                st.success(f"Deleted {len(selected_ids)} transaction(s) ✅")
                st.rerun()

        # Quick add row
        st.markdown("")
        with st.expander("➕ Add a Transaction"):
            ac1, ac2, ac3, ac4 = st.columns([1.2, 1, 2, 2])
            with ac1:
                add_date = st.date_input("Date", value=date.today(), key="add_date")
            with ac2:
                add_amount = st.number_input("Amount (BHD)", min_value=0.0, step=0.5, key="add_amount")
            with ac3:
                add_merchant = st.text_input("Merchant", placeholder="e.g. Starbucks", key="add_merchant")
            with ac4:
                add_category = st.selectbox("Category", options=all_app_categories, key="add_category")

            add_item = st.text_input("Item Name (optional)", placeholder="e.g. Latte", key="add_item")

            if st.button("💾 Add Transaction", type="primary", use_container_width=True, key="add_txn_btn"):
                if add_amount > 0 and add_merchant.strip():
                    row = pd.DataFrame([{
                        "date": add_date,
                        "amount_bhd": round(add_amount, 3),
                        "merchant": add_merchant.strip(),
                        "category": add_category,
                        "raw_sms": f"Manual: {add_amount} BHD at {add_merchant}",
                        "item_name": add_item.strip() if add_item else add_merchant.strip(),
                        "qty": 1,
                        "source_type": "manual",
                        "store_name": add_merchant.strip(),
                        "currency": "BHD",
                    }])
                    save_transactions(row)
                    st.success(f"Added: {add_amount:.3f} BHD at {add_merchant} ✅")
                    st.rerun()
                else:
                    st.warning("Enter an amount and merchant name.")

st.markdown("---")
st.caption("Built with ❤️ in Bahrain")