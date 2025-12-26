import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config

st.set_page_config(
    page_title="Skincare Recommender",
    page_icon="🧴",
    layout="centered"
)

st.title("🧴 Skincare Recommendation System")
st.caption("Skin-type based recommendations using ingredient intelligence ✨")
st.caption("💰 All prices are shown in INR (₹)")


# Load dataset

df = pd.read_csv("skincare_products_clean.csv")


# Currency conversion (SYMBOL → INR)

EUR_TO_INR = 90.0
USD_TO_INR = 83.0
GBP_TO_INR = 105.0

def detect_currency(price_str):
    if "€" in price_str:
        return "EUR"
    elif "$" in price_str:
        return "USD"
    elif "£" in price_str:
        return "GBP"
    else:
        return "EUR"  # fallback

def convert_to_inr(price_str):
    currency = detect_currency(price_str)
    value = float(re.sub(r"[^\d.]", "", price_str))

    if currency == "EUR":
        return value * EUR_TO_INR
    elif currency == "USD":
        return value * USD_TO_INR
    elif currency == "GBP":
        return value * GBP_TO_INR
    else:
        return value

df["price_inr"] = df["price"].astype(str).apply(convert_to_inr)


# Clean ingredients

def clean_ingredients(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z, ]", "", text)
    text = text.replace(",", " ")
    return text


df["clean_ingredients"] = df["clean_ingreds"].apply(clean_ingredients)


# Skin type rules

SKIN_TYPE_RULES = {
    "Oily": {
        "good": ["niacinamide", "salicylic", "zinc", "tea tree"],
        "avoid": ["coconut oil", "shea butter", "lanolin"]
    },
    "Dry": {
        "good": ["glycerin", "hyaluronic", "ceramide", "squalane"],
        "avoid": ["alcohol"]
    },
    "Sensitive": {
        "good": ["centella", "aloe", "panthenol"],
        "avoid": ["fragrance", "essential oil", "alcohol"]
    },
    "Combination": {
        "good": ["niacinamide", "green tea"],
        "avoid": []
    }
}


# Skin type scoring

def skin_type_score(ingredients, skin_type):
    score = 0
    rules = SKIN_TYPE_RULES[skin_type]

    for good in rules["good"]:
        if good in ingredients:
            score += 1

    for bad in rules["avoid"]:
        if bad in ingredients:
            score -= 1

    return score

for skin in SKIN_TYPE_RULES.keys():
    df[f"{skin}_score"] = df["clean_ingredients"].apply(
        lambda x: skin_type_score(x, skin)
    )


# TF-IDF similarity

tfidf = TfidfVectorizer(stop_words="english")
ingredient_matrix = tfidf.fit_transform(df["clean_ingredients"])
similarity_matrix = cosine_similarity(ingredient_matrix)


# User inputs

skin_type = st.selectbox(
    "Select your skin type 👇",
    ["Oily", "Dry", "Sensitive", "Combination"]
)

product_type = st.selectbox(
    "What are you looking for? 🧴",
    sorted(df["product_type"].dropna().unique())
)

min_price = int(df["price_inr"].min())
max_price = int(df["price_inr"].max())

price_range = st.slider(
    "Select your budget range (₹)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

top_n = st.slider("Number of recommendations", 3, 10, 5)


# Recommendation logic

def recommend_products(skin_type, product_type, price_range, top_n):
    min_p, max_p = price_range

    filtered = df[df["product_type"] == product_type]
    filtered = filtered[
        (filtered["price_inr"] >= min_p) &
        (filtered["price_inr"] <= max_p)
    ]

    if filtered.empty:
        return pd.DataFrame()

    filtered = filtered.sort_values(
        by=f"{skin_type}_score",
        ascending=False
    )

    top_candidates = filtered.head(30)
    idx = top_candidates.index[0]

    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_indices = [
        i[0] for i in similarity_scores
        if i[0] in top_candidates.index
    ][:top_n]

    return df.loc[top_indices]


# Display results

if st.button("✨ Get Recommendations"):
    results = recommend_products(
        skin_type,
        product_type,
        price_range,
        top_n
    )

    if results.empty:
        st.warning("No products found in this price range 😕")
    else:
        st.subheader("Recommended Products 💖")

        for _, row in results.iterrows():
            st.markdown(f"### {row['product_name']}")
            st.write(f"🧴 **Type:** {row['product_type']}")
            st.write(f"💰 **Price:** ₹{row['price_inr']:.0f}")
            st.markdown(f"[🔗 View Product]({row['product_url']})")
            st.markdown("---")
