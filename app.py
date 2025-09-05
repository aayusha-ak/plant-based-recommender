#Aayusha Kandel

import streamlit as st
from recommender import loadData, recommendRecipes

st.title("Plant Based Meal Recommender")
#st.write("Hello World!")

# Load the data and process it
veganRecipe, newIngredient, vectorizer, tfidf_matrix = loadData()

# User input for ingredients
user_ingredients = st.multiselect(
    "Enter ingredients you would like to see in your plant-based meals:",
    options=newIngredient,
    
)     

# If user has selected ingredients, recommend recipes
if st.button("Get Recommendations") and user_ingredients:
    recommendations = recommendRecipes(user_ingredients, vectorizer, tfidf_matrix, veganRecipe)
    if recommendations:
        st.write("Top 5 recommended recipes based on your ingredients:")
        for recipe in recommendations:
            st.subheader(recipe['name'])
            st.write(recipe['ingredients'])
            st.write(recipe['preparation'])
            st.markdown(f"[View Recipe]({recipe['link']})")
            st.divider()
            
    else:
        st.write("No recommendations found. Please try different ingredients.")
else:
    st.write("Please select ingredients to get recipe recommendations.")
