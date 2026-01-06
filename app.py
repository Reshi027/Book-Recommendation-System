import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>📚 Book Recommendation System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Machine Learning (Truncated SVD) based Recommendation System</p>",
    unsafe_allow_html=True
)

books = pd.read_csv("books.csv", low_memory=False)
ratings = pd.read_csv("ratings.csv")

active_users = ratings["User-ID"].value_counts()
active_users = active_users[active_users >= 50].index
ratings = ratings[ratings["User-ID"].isin(active_users)]

popular_books = ratings["ISBN"].value_counts()
popular_books = popular_books[popular_books >= 50].index
ratings = ratings[ratings["ISBN"].isin(popular_books)]

user_book = ratings.pivot_table(
    index="User-ID",
    columns="ISBN",
    values="Book-Rating"
).fillna(0)

svd = TruncatedSVD(n_components=50, random_state=42)
user_features = svd.fit_transform(user_book)

user_sim = cosine_similarity(user_features)

def recommend(user_id):
    if user_id not in user_book.index:
        return None

    idx = user_book.index.get_loc(user_id)

    scores = list(enumerate(user_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    similar_users = [user_book.index[i[0]] for i in scores]

    user_books = ratings[ratings["User-ID"] == user_id]["ISBN"]

    recs = ratings[ratings["User-ID"].isin(similar_users)]
    recs = recs[~recs["ISBN"].isin(user_books)]

    result = recs.merge(
        books, on="ISBN"
    )[["Book-Title", "Book-Author"]].drop_duplicates().head(5)

    return result

st.sidebar.header("🔍 User Selection")

user_id = st.sidebar.selectbox(
    "Select User ID",
    user_book.index.tolist()
)

if st.sidebar.button("Recommend Books"):
    with st.spinner("Finding books you may like..."):
        result = recommend(user_id)

        st.subheader(" Recommended Books")

        if result is None or result.empty:
            st.write("No recommendations found.")
        else:
            st.dataframe(
                result.reset_index(drop=True),
                use_container_width=True
            )
