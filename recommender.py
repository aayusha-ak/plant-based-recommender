import pandas as pd
import re
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Fucntion to clean ingredients list from csv file
def cleanIngredient(text):
    # Removes leading and trailing whitespaces
    # Each ingredient is on a new line
    lines = text.strip().split('\n')

    cleaned = []

    for line in lines:
        line = line.lower()
        
        # Taking out measurements
        line = re.sub(r'\d+[\d\s\/]*[a-z]*', '', line)
        # Taking out () and anything inside it
        line = re.sub(r'\(.*?\)', '', line)
        # Taking out any character that is not a word
        line = re.sub(r'[^\w\s]', '', line)
        # Taking out specific words
        line = re.sub(r'\b(chopped|usm|units|scale|package|ingredients|diced|grated|minced|can|bag|medium|large|small|tbsp|tsp|g|ml|of|skin|bowl|tooth|suggestions|squeeze|3/4|home|drink|night|mountain|selection|silicone|accuracy|example|everyday|Original|air|)\b', '', line)
        # Taking out one or more whitespaces, but replaces with one
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            cleaned.append(line)
    return cleaned

# Function to load data and get ready for recommendations
def loadData():
    veganRecipe = pd.read_csv("Data/vegan_recipes.csv")

    # Get ingredients from dataset
    ingredientColumn = veganRecipe['ingredients']

    # Apply ingredient cleaning function to ingredients
    cleanedIngredient = ingredientColumn.apply(cleanIngredient)

    # Loop through cleanedIngredient list and add each ingredient to IngredientWords list
    IngredientWords = []
    for iL in cleanedIngredient:
        for iLL in iL:
            IngredientWords.append(iLL)
    
    # Join all elements in list to one string value
    string_cleanedIngredient = ' '.join(IngredientWords)

    # Load English model
    nlp = spacy.load('en_core_web_sm')

    # Putting ingredients through nlp pipeline
    IngredientObject = nlp(string_cleanedIngredient)

    newIngredient = []

    #looping through ingredianteObjecct to find nouns and adding them to newIngredient
    for token in IngredientObject:
        if token.pos_ == 'NOUN':
            newIngredient.append(token.text)

    #remove duplicates
    newIngredient = list(set(newIngredient))

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(veganRecipe['ingredients'])

    
    return veganRecipe, newIngredient, vectorizer, tfidf_matrix


# Fucntion to recommend recipes based on user inputs
def recommendRecipes(user_ingredients, vectorizer, tfidf_matrix, veganRecipe):

    # Convert user ingredients to a string for vectorization
    userIngredientList = list(user_ingredients)
    userIngredientString = ' '.join(userIngredientList)

    # Transform the user ingredients into the same vector space as the recipes
    userVector = vectorizer.transform([userIngredientString])

    # Calculate cosine similarities between user vector and recipe vectors
    #flatten the result to get a 1D array of similarities to find top matches
    cosine_similarities = cosine_similarity(userVector, tfidf_matrix).flatten()

    #Order the indices in descending order of similarity and get the 1st 5 indices
    topIndices = cosine_similarities.argsort()[::-1][:5]

    recommendations = []
    for i in topIndices:
        recipeName = veganRecipe.iloc[i]['title']
        recipeIngredients = veganRecipe.iloc[i]['ingredients']
        recipeLink = veganRecipe.iloc[i]['href']
        recipePreparation = veganRecipe.iloc[i]['preparation']
        recommendations.append(
            {
                'name': recipeName,
                'ingredients': recipeIngredients,
                'link': recipeLink,
                'preparation': recipePreparation
            }
        )
    return recommendations
        
    

