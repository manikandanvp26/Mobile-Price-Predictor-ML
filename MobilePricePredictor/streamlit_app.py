"""Streamlit version of the Mobile Price Predictor with all original features."""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mobile_price_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
DATA_PATH = BASE_DIR / "data" / "mobile_phones.csv"
TEMPLATE_PATH = BASE_DIR / "static" / "phone_template.csv"

CATEGORICAL = ["brand", "model"]
NUMERIC = ["rating", "ram_gb", "storage_gb", "screen_inch",
           "main_camera_mp", "front_camera_mp", "battery_mah"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price_inr"
LABELS = {
    "rating": "User rating (1-5)",
    "ram_gb": "RAM (GB)",
    "storage_gb": "Storage (GB)",
    "screen_inch": "Screen size (inches)",
    "main_camera_mp": "Rear camera (MP)",
    "front_camera_mp": "Selfie camera (MP)",
    "battery_mah": "Battery (mAh)",
}
RANGES = {
    "rating": (1.0, 5.0),
    "ram_gb": (1.0, 32.0),
    "storage_gb": (1.0, 2048.0),
    "screen_inch": (3.0, 10.0),
    "main_camera_mp": (1.0, 300.0),
    "front_camera_mp": (1.0, 100.0),
    "battery_mah": (500.0, 20000.0),
}
PAGES = [
    "🎯 Predict", "📱 Compare", "🧪 What-if", "💡 Recommend", "🔮 Future price",
    "📈 Analytics", "📊 Model evaluation", "🧠 Explain", "❤️ Wishlist & alerts",
    "📂 Batch upload", "〰 Price curve", "ⓘ About",
]


def price_category(price):
    if price < 10000:
        return "Budget"
    if price < 25000:
        return "Mid-range"
    if price < 50000:
        return "Premium"
    return "Flagship"


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH)
    return raw.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


DATA = load_data()
_SCALE = DATA[NUMERIC].std().replace(0, 1).to_numpy(dtype=float)
_NUM_ARRAY = DATA[NUMERIC].to_numpy(dtype=float) / _SCALE


def predict_df(df, bundle):
    """Predicted prices in INR, never negative."""
    return np.maximum(0, bundle["model"].predict(df[FEATURES]))


def estimate(brand, model_name, specs, bundle):
    row = pd.DataFrame([{"brand": brand, "model": model_name, **specs}])
    return float(predict_df(row, bundle)[0])


def phone_defaults(brand, model_name):
    """Median specs for this brand/model (or brand, or whole dataset)."""
    subset = DATA[(DATA.brand == brand) & (DATA.model == model_name)]
    if subset.empty:
        subset = DATA[DATA.brand == brand]
    if subset.empty:
        subset = DATA
    return {c: float(subset[c].median()) for c in NUMERIC}


def fairness_score(specs, price):
    """Compare the price with the median of the 5 most similar dataset listings."""
    target = np.array([specs[c] for c in NUMERIC], dtype=float) / _SCALE
    dist = np.sqrt(((_NUM_ARRAY - target) ** 2).sum(axis=1))
    nearest = np.argsort(dist)[:5]
    median = float(DATA[TARGET].iloc[nearest].median())
    gap = abs(price - median) / max(median, 1.0) * 100
    return max(0.0, 100.0 - gap), median


def brand_model_select(key):
    brands = sorted(DATA["brand"].unique())
    brand = st.selectbox("Brand", brands, key=f"{key}_brand")
    models = sorted(DATA.loc[DATA.brand == brand, "model"].unique())
    model_name = st.selectbox("Mobile model", models, key=f"{key}_model")
    return brand, model_name


def spec_inputs(key, defaults):
    values = {}
    cols = st.columns(3)
    for i, col in enumerate(NUMERIC):
        lo, hi = RANGES[col]
        with cols[i % 3]:
            values[col] = st.number_input(
                LABELS[col], min_value=lo, max_value=hi,
                value=float(min(max(defaults[col], lo), hi)), key=f"{key}_{col}",
            )
    return {c: float(v) for c, v in values.items()}


def add_to_list(key, item):
    st.session_state.setdefault(key, []).append(item)


def remove_from_list(key, index):
    items = st.session_state.get(key, [])
    if 0 <= index < len(items):
        items.pop(index)


def build_pdf(result):
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    y = 800
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Mobile Price Report")
    y -= 30
    pdf.setFont("Helvetica", 11)
    lines = [f"Brand: {result['brand']}", f"Model: {result['model']}"]
    lines += [f"{LABELS[k]}: {v:g}" for k, v in result["specs"].items()]
    lines += [
        f"Estimated price: INR {result['price']:,.0f}",
        f"Likely range: INR {result['low']:,.0f} to INR {result['high']:,.0f}",
        f"Category: {result['category']}",
        f"Fairness score: {result['fairness']:.0f}/100",
        "Note: estimate based on dataset listings. Not a regulated valuation.",
    ]
    for line in lines:
        pdf.drawString(50, y, line)
        y -= 20
    pdf.save()
    return buf.getvalue()



APP_CSS = """
.stApp {background:#ECEDE4;}
.block-container {padding-top:1.2rem; max-width:1280px;}
.header {background:#1E3A5F; color:#fff; border-radius:16px; padding:18px 24px; margin-bottom:12px;}
.brand-title {font-size:1.45rem; font-weight:700;}
.brand-sub {opacity:.85; font-size:.9rem;}
.meta {display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; color:#4a5568; margin:6px 0 10px;}
.badge {background:#E3E8F0; border-radius:999px; padding:4px 12px; font-weight:600; font-size:.8rem; color:#1E3A5F;}
.page-title {font-size:1.6rem; font-weight:700; color:#1E3A5F;}
.page-sub {color:#4a5568; margin-bottom:10px;}
.card-title {font-weight:700; font-size:1.1rem; color:#1E3A5F; margin-bottom:6px;}
.section-title {font-weight:700; font-size:1.1rem; color:#1E3A5F; margin:18px 0 8px;}
div[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:16px; box-shadow:0 2px 8px rgba(30,58,95,.08);}
.muted {color:#4a5568;}
.small {font-size:.85rem;}
.center {text-align:center;}
.big-price {font-size:2.6rem; font-weight:800; color:#1E3A5F; text-align:center; margin-top:6px;}
.pill-cat {background:#E3E8F0; border-radius:999px; padding:6px; text-align:center; font-weight:600; color:#1E3A5F; margin:12px 0;}
.fair-row {display:flex; justify-content:space-between; align-items:center; font-weight:600; color:#1E3A5F; margin-top:10px;}
.fair-score {font-size:1.4rem; font-weight:800;}
.tile-grid {display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px;}
.spec-tile {background:#fff; border:1px solid #DDE1E8; border-radius:12px; padding:12px;}
.spec-value {font-weight:700; font-size:1.15rem; color:#1E3A5F; word-break:break-word;}
.spec-label {color:#6b7280; font-size:.8rem;}
@media (max-width: 640px) {.big-price {font-size:2rem;} .page-title {font-size:1.3rem;}}
"""


def spec_tile(label, value):
    return f'<div class="spec-tile"><div class="spec-value">{value}</div><div class="spec-label">{label}</div></div>'


# ---------- Page: Predict ----------
def page_predict(bundle):
    uncertainty = float(bundle["uncertainty_abs"])
    st.markdown('<div class="page-title">Predict Mobile Price</div>'
                '<div class="page-sub">Choose a real brand and model from the dataset. '
                'Results update as you change the specs.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        with st.container(border=True):
            st.markdown('<div class="card-title">Choose a phone</div>', unsafe_allow_html=True)
            brand, model_name = brand_model_select("pred")
            specs = spec_inputs("pred", phone_defaults(brand, model_name))

    price = estimate(brand, model_name, specs, bundle)
    fair, median = fairness_score(specs, price)
    gap = 100 - fair
    low = max(0.0, price - uncertainty)
    high = price + uncertainty
    category = price_category(price)
    result = {
        "brand": brand, "model": model_name, "specs": specs, "price": price,
        "low": low, "high": high, "category": category, "fairness": fair, "median": median,
    }

    with right:
        with st.container(border=True):
            st.markdown('<div class="center muted">Estimated price</div>'
                        f'<div class="big-price">₹{price:,.0f}</div>'
                        f'<div class="center muted small">Estimated range: ₹{low:,.0f} – ₹{high:,.0f} '
                        '(90% test residual range)</div>'
                        f'<div class="pill-cat">{category}</div>', unsafe_allow_html=True)
            st.markdown('<div class="fair-row"><span>Price fairness score</span>'
                        f'<span class="fair-score">{fair:.0f}/100</span></div>', unsafe_allow_html=True)
            st.progress(int(round(fair)))
            verdict = "Fair" if gap <= 10 else "Outside typical range"
            st.caption(f"{verdict} · {gap:.1f}% vs comparable median (₹{median:,.0f}). "
                       "A heuristic, not a valuation.")
            st.download_button("Download PDF report", build_pdf(result),
                               file_name="phone_report.pdf", mime="application/pdf")
            st.button("♡ Add to wishlist", on_click=add_to_list, args=("wishlist", dict(result)))
            target = st.number_input("Alert target price (₹)", min_value=0.0,
                                     value=float(round(price)), step=1000.0, key="alert_target")
            st.button("＋ Set price alert", on_click=add_to_list,
                      args=("alerts", {**result, "target": target}))

    st.markdown('<div class="section-title">Selected phone details</div>', unsafe_allow_html=True)
    tiles = [spec_tile("Brand", brand), spec_tile("Mobile model", model_name)]
    tiles += [spec_tile(LABELS[c], f"{specs[c]:g}") for c in NUMERIC]
    st.markdown(f'<div class="tile-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)


# ---------- Page: Compare ----------
def page_compare(bundle):
    st.subheader("Mobile comparison")
    options = {f"{b} {m}": (b, m) for b, m in
               DATA[["brand", "model"]].drop_duplicates().itertuples(index=False)}
    labels = sorted(options)
    chosen = st.multiselect("Choose up to four phones", labels, default=labels[:2],
                            max_selections=4, key="cmp_choice")
    if not chosen:
        st.info("Choose at least one phone.")
        return
    uncertainty = float(bundle["uncertainty_abs"])
    rows = []
    for label in chosen:
        b, m = options[label]
        d = phone_defaults(b, m)
        price = estimate(b, m, d, bundle)
        rows.append({
            "Brand": b, "Model": m, "Predicted (₹)": round(price),
            "Low (₹)": round(max(0, price - uncertainty)),
            "High (₹)": round(price + uncertainty),
            "Category": price_category(price),
            **{LABELS[c]: round(d[c], 2) for c in NUMERIC},
        })
    st.dataframe(pd.DataFrame(rows))
    st.caption("Specs are the median values for each phone in the dataset.")


# ---------- Page: What-if ----------
def page_whatif(bundle):
    st.subheader("What-if price simulator")
    st.write("See how changing a phone's specs changes its estimated price.")
    brand, model_name = brand_model_select("wi")
    d = phone_defaults(brand, model_name)
    base = estimate(brand, model_name, d, bundle)
    st.metric("Base estimated price", f"₹{base:,.0f}")

    rows = []
    for col in NUMERIC:
        for pct in (-10, 10):
            new = float(np.clip(d[col] * (1 + pct / 100), *RANGES[col]))
            p = estimate(brand, model_name, {**d, col: new}, bundle)
            rows.append({
                "Change": f"{LABELS[col]} {pct:+d}%",
                "New value": round(new, 2),
                "Estimated price (₹)": round(p),
                "Difference (₹)": round(p - base),
            })
    st.markdown("**Standard scenarios (±10% per spec)**")
    st.dataframe(pd.DataFrame(rows))

    st.markdown("**Custom change**")
    col = st.selectbox("Spec to change", NUMERIC, format_func=lambda c: LABELS[c],
                       key="wi_feature")
    lo, hi = RANGES[col]
    new_val = st.number_input("New value", min_value=lo, max_value=hi,
                              value=float(min(max(d[col], lo), hi)),
                              key=f"wi_val_{col}")
    p = estimate(brand, model_name, {**d, col: float(new_val)}, bundle)
    st.metric("Estimated price with this change", f"₹{p:,.0f}",
              delta=f"₹{p - base:,.0f}")


# ---------- Page: Recommend ----------
def page_recommend(bundle):
    st.subheader("Smart recommendations")
    st.write("Finds phones under your budget and ranks them by value: "
             "estimated price minus listing price (higher is a better deal).")
    budget = st.number_input("Maximum budget (₹)", min_value=1000.0,
                             value=30000.0, step=1000.0, key="rec_budget")
    brand_opt = st.selectbox("Brand", ["All"] + sorted(DATA.brand.unique()), key="rec_brand")
    min_ram = st.selectbox("Minimum RAM (GB)", [2, 4, 6, 8, 12], index=1, key="rec_ram")
    min_storage = st.selectbox("Minimum storage (GB)", [32, 64, 128, 256], index=1,
                               key="rec_storage")
    if st.button("Find best phones", key="rec_go"):
        f = DATA[(DATA[TARGET] <= budget) & (DATA.ram_gb >= min_ram)
                 & (DATA.storage_gb >= min_storage)]
        if brand_opt != "All":
            f = f[f.brand == brand_opt]
        if f.empty:
            st.warning("No phones match these filters. Try a higher budget or fewer requirements.")
            return
        f = f.copy()
        f["predicted"] = predict_df(f, bundle).round(0)
        f["value_gap"] = (f["predicted"] - f[TARGET]).round(0)
        top = f.sort_values(["value_gap", "rating"], ascending=[False, False]).head(10)
        st.dataframe(top[["brand", "model", TARGET, "predicted", "value_gap",
                          "rating", "ram_gb", "storage_gb"]].rename(columns={
            "brand": "Brand", "model": "Model", TARGET: "Listing price (₹)",
            "predicted": "Estimated price (₹)", "value_gap": "Value gap (₹)",
            "rating": "Rating", "ram_gb": "RAM (GB)", "storage_gb": "Storage (GB)"}))


# ---------- Page: Future price ----------
def page_future(bundle):
    st.subheader("Future price scenario")
    st.warning("This dataset has no dated price history, so this is an assumption-based "
               "projection, not a learned market trend.")
    brand, model_name = brand_model_select("future")
    current = estimate(brand, model_name, phone_defaults(brand, model_name), bundle)
    depreciation = st.slider("Annual depreciation (%)", 0, 50, 15, key="future_dep")
    periods = {"6 months": 0.5, "1 year": 1, "2 years": 2, "3 years": 3, "5 years": 5}
    choice = st.selectbox("Forecast period", list(periods), index=1, key="future_period")
    years = periods[choice]
    projected = current * (1 - depreciation / 100) ** years
    c1, c2 = st.columns(2)
    c1.metric("Current estimated price", f"₹{current:,.0f}")
    c2.metric(f"Estimated price in {choice}", f"₹{projected:,.0f}")
    table = pd.DataFrame([{"Period": k, "Estimated price (₹)": round(current * (1 - depreciation / 100) ** y)}
                          for k, y in periods.items()])
    st.dataframe(table)


# ---------- Page: Analytics ----------
def page_analytics():
    st.subheader("Analytics dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", f"{len(DATA):,}")
    c2.metric("Brands", DATA.brand.nunique())
    c3.metric("Median price", f"₹{DATA[TARGET].median():,.0f}")
    c4.metric("Price range", f"₹{DATA[TARGET].min():,.0f} – ₹{DATA[TARGET].max():,.0f}")

    order = ["Budget", "Mid-range", "Premium", "Flagship"]
    bands = DATA[TARGET].map(price_category).value_counts().reindex(order, fill_value=0)
    st.markdown("**Price bands**")
    st.bar_chart(bands)
    st.markdown("**Top brands by records**")
    st.bar_chart(DATA.brand.value_counts().head(10))


# ---------- Page: Model evaluation ----------
def page_evaluation(metrics):
    st.subheader("Model evaluation")
    st.write("Models are evaluated on a held-out test split. Cross-validation picked the deployed model.")
    models = metrics["models"]
    keys = list(models)
    label = lambda k: models[k]["label"]
    mode = st.radio("Mode", ["Single model", "Compare two models"], horizontal=True, key="ev_mode")
    if mode == "Single model":
        chosen = [st.selectbox("Model", keys, format_func=label,
                               index=keys.index(metrics["best_model"]), key="ev_a")]
    else:
        a = st.selectbox("Model 1", keys, format_func=label, index=0, key="ev_a2")
        b = st.selectbox("Model 2", keys, format_func=label, index=1, key="ev_b2")
        if a == b:
            st.warning("Choose two different models to compare.")
            return
        chosen = [a, b]
    table = pd.DataFrame([{
        "Model": models[k]["label"], "CV R²": models[k]["cv_r2_mean"],
        "Test R²": models[k]["test_r2"], "Test MAE (₹)": models[k]["test_mae_inr"],
        "RMSE (₹)": models[k]["test_rmse_inr"], "MAPE (%)": models[k]["test_mape_pct"],
        "Within 10% (%)": models[k]["within_10pct"],
    } for k in chosen]).set_index("Model")
    st.dataframe(table)
    st.bar_chart(table[["Test R²"]])
    st.caption(f"Deployed model (best by CV R²): {label(metrics['best_model'])}")
    st.markdown("**Feature importance (deployed model)**")
    imp = pd.DataFrame(metrics["feature_importance"]).set_index("label")[["importance"]]
    st.bar_chart(imp)


# ---------- Page: Explain ----------
def page_explain(bundle):
    st.subheader("Explainable AI")
    st.write("Shows how each numeric spec moves the estimate: the change in price when that spec "
             "is replaced by its dataset median. This is an occlusion method, not SHAP values.")
    brand, model_name = brand_model_select("exp")
    d = phone_defaults(brand, model_name)
    specs = spec_inputs("exp", d)
    base = estimate(brand, model_name, specs, bundle)
    effects = {}
    for col in NUMERIC:
        without = {**specs, col: float(DATA[col].median())}
        effects[LABELS[col]] = round(base - estimate(brand, model_name, without, bundle))
    eff = pd.Series(effects).sort_values(key=lambda s: s.abs(), ascending=False)
    st.metric("Estimated price", f"₹{base:,.0f}")
    st.bar_chart(eff)
    st.dataframe(eff.rename("Effect on price (₹)").to_frame())


# ---------- Page: Wishlist & alerts ----------
def page_wishlist(bundle):
    st.subheader("Wishlist & price alerts")
    st.write("Alerts trigger when the model's estimate reaches your target. "
             "They do not monitor live retailer prices.")
    wishlist = st.session_state.get("wishlist", [])
    st.markdown("**Saved phones**")
    if not wishlist:
        st.info("No saved phones yet. Add one from the Predict page.")
    for i, item in enumerate(wishlist):
        c1, c2 = st.columns([4, 1])
        c1.write(f"**{item['brand']} {item['model']}**: ₹{item['price']:,.0f} "
                 f"({item['category']})")
        c2.button("Remove", key=f"rm_w_{i}", on_click=remove_from_list,
                  args=("wishlist", i))

    alerts = st.session_state.get("alerts", [])
    st.markdown("**Price alerts**")
    if not alerts:
        st.info("No alerts yet. Set one from the Predict page.")
    for i, a in enumerate(alerts):
        now = estimate(a["brand"], a["model"], a["specs"], bundle)
        status = "🔔 Target reached" if now <= a["target"] else "Waiting"
        c1, c2 = st.columns([4, 1])
        c1.write(f"**{a['brand']} {a['model']}**: target ₹{a['target']:,.0f}, "
                 f"current estimate ₹{now:,.0f}. {status}")
        c2.button("Remove", key=f"rm_a_{i}", on_click=remove_from_list,
                  args=("alerts", i))
    st.caption("Saved items last only for this browser session. They reset when the app restarts.")


# ---------- Page: Batch upload ----------
REQUIRED_BATCH = ["brand", "ram_gb", "storage_gb", "screen_inch",
                  "main_camera_mp", "front_camera_mp", "battery_mah"]


def prepare_batch(raw):
    """Clean an uploaded table so the model can use it. Returns (df, notes, error)."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)
    missing = [c for c in REQUIRED_BATCH if c not in df.columns]
    if missing:
        return None, [], f"Missing required columns: {', '.join(missing)}"

    notes = []
    df["brand"] = df["brand"].fillna("Unknown").astype(str).str.strip()
    if "model" not in df.columns:
        df["model"] = "Unknown"
        notes.append("No 'model' column, so 'Unknown' was used. Brand and specs drive the estimate.")
    else:
        df["model"] = df["model"].fillna("Unknown").astype(str).str.strip()
    if "rating" not in df.columns:
        median_rating = float(DATA["rating"].median())
        df["rating"] = median_rating
        notes.append(f"No 'rating' column, so the dataset median ({median_rating:.1f}) was used.")

    filled = 0
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        blanks = int(df[c].isna().sum())
        if blanks:
            df[c] = df[c].fillna(float(DATA[c].median()))
            filled += blanks
    if filled:
        notes.append(f"{filled} blank or invalid spec values were filled with dataset medians.")
    return df, notes, None


def page_batch(bundle):
    st.markdown('<div class="page-title">Batch upload</div>'
                '<div class="page-sub">Upload a CSV or Excel file of phones. Model and rating are optional; '
                'blank specs are filled with dataset medians.</div>', unsafe_allow_html=True)
    if TEMPLATE_PATH.exists():
        st.download_button("Download CSV template", TEMPLATE_PATH.read_bytes(),
                           file_name="phone_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], key="batch_file")
    if uploaded is None:
        return
    try:
        if uploaded.name.lower().endswith(".xlsx"):
            raw = pd.read_excel(uploaded)
        else:
            raw = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        return

    clean, notes, error = prepare_batch(raw)
    if error:
        st.error(error)
        return
    for note in notes:
        st.info(note)
    try:
        out = clean.copy()
        out["predicted_price_inr"] = predict_df(clean, bundle).round(0)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return
    if TARGET in out.columns:
        actual = pd.to_numeric(out[TARGET], errors="coerce")
        out["difference_inr"] = (out["predicted_price_inr"] - actual).round(0)

    st.metric("Phones predicted", len(out))
    st.dataframe(out)
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False)
    c2.download_button("Download Excel", xbuf.getvalue(), file_name="predictions.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------- Page: Price curve ----------
def page_curve(bundle):
    st.subheader("Price curve")
    st.write("See how one spec changes the estimated price, with the others held at the phone's median values.")
    brand, model_name = brand_model_select("curve")
    d = phone_defaults(brand, model_name)
    options = ["ram_gb", "storage_gb", "battery_mah", "main_camera_mp"]
    feature = st.selectbox("Feature", options, format_func=lambda c: LABELS[c], key="curve_feature")
    values = np.linspace(DATA[feature].min(), DATA[feature].max(), 25)
    rows = pd.DataFrame([{"brand": brand, "model": model_name, **d} for _ in values])
    rows[feature] = values
    prices = np.maximum(0, bundle["model"].predict(rows[FEATURES]))
    chart = pd.DataFrame({LABELS[feature]: np.round(values, 1),
                          "Estimated price (₹)": np.round(prices)}).set_index(LABELS[feature])
    st.line_chart(chart)


# ---------- Page: About ----------
def page_about():
    st.subheader("About this project")
    st.write("This app trains machine-learning regressors on the 836-row mobile phone dataset. "
             "Brand and model choices come directly from the dataset.")
    st.warning("Forecast limitation: the source data has no dated price history. Future prices "
               "are scenario projections using the annual depreciation you choose.")
    st.info("Alert limitation: alerts compare the model's estimate with your target. They are "
            "not live retailer price tracking.")


# ---------- Main ----------
st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

bundle = load_bundle()
metrics = load_metrics()
best_label = metrics["models"][metrics["best_model"]]["label"]
st.markdown('<div class="header"><div class="brand-title">Mobile Price Predictor</div>'
            '<div class="brand-sub">Smart AI-powered mobile analytics</div></div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="meta"><span>Dataset-driven mobile intelligence · '
            f'<b>{len(DATA):,} phone listings</b></span>'
            f'<span class="badge">{best_label} · dataset trained</span></div>',
            unsafe_allow_html=True)
page = st.pills("Page", PAGES, default=PAGES[0], key="page",
                label_visibility="collapsed") or PAGES[0]

if page == "🎯 Predict":
    page_predict(bundle)
elif page == "📱 Compare":
    page_compare(bundle)
elif page == "🧪 What-if":
    page_whatif(bundle)
elif page == "💡 Recommend":
    page_recommend(bundle)
elif page == "🔮 Future price":
    page_future(bundle)
elif page == "📈 Analytics":
    page_analytics()
elif page == "📊 Model evaluation":
    page_evaluation(metrics)
elif page == "🧠 Explain":
    page_explain(bundle)
elif page == "❤️ Wishlist & alerts":
    page_wishlist(bundle)
elif page == "📂 Batch upload":
    page_batch(bundle)
elif page == "〰 Price curve":
    page_curve(bundle)
else:
    page_about()
