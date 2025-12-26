 🧴 Skincare Recommendation System

A skin-type based skincare product recommendation system built using ingredient analysis and deployed with Streamlit.


 📌 Problem Statement
Most publicly available skincare datasets do not include explicit skin type labels. This project addresses that limitation by inferring skin type compatibility from ingredient composition and recommending suitable products accordingly.

 🧠 Approach
- Cleaned and normalized ingredient text data  
- Inferred skin type suitability using rule-based ingredient heuristics  
- Applied TF-IDF vectorization on ingredients  
- Computed cosine similarity between products  
- Filtered recommendations based on:
  - Skin type  
  - Product category (moisturizer, serum, cleanser, etc.)  
  - User-defined budget range  
- Converted mixed currency prices (€,$,£) to INR for consistent filtering  


🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  


📊 Key Features
- Skin-type based personalization (Oily, Dry, Sensitive, Combination)  
- Product category selection  
- Budget-aware recommendations (INR)  
- Ingredient-based similarity matching  
- Explainable and rule-based logic  


 🌐 Streamlit Application:
The project includes an interactive Streamlit web app where users can:
- Select their skin type  
- Choose the product category  
- Set a budget range  
- View personalized product recommendations with prices and links  


 ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py

```
👩‍💻 Author
Kavya Srivastava

