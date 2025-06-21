# Recipe Rating Forecast 🍽️

---

## Introduction

Cooking delicious meals for oneself, family, and friends can be an extremely rewarding experience. Seeing others enjoy something that you've created with smiles of joy, astonishment, and awe brings an unmatched satisfaction. But what really makes a recipe stand out from some others? There must be underlying factors that play a role in determining highly rated food in addition to personal preferences. This project aims to incorportate various machine learning techniques to perform an analysis on the dataset "Recipes and Ratings", which conatin recipes and ratings from [food.com](https://www.food.com/) since 2008, to perform an analysis on a chosen dataset, construct a predictive model, create visualizations, and report the findings. Ultimately answering the question: 

> **What factors contribute to a highly rated recipe?**

1. `RAW_recipes.csv`: 83782 rows where each row is a unique recipe. 

| Column             | Description                                                    |
| ------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `'name'`           | Recipe name                                                                                                                                                                                                                                    |
| `'id'`             | Recipe ID                                                                                                                                                                                                                                      |
| `'minutes'`        | Minutes to prepare recipe                                                                                                                                                                                                                      |
| `'contributor_id'` | User ID who submitted this recipe                                                                                                                                                                                                              |
| `'tags'`           | [Food.com](food.com) tags for recipe                                                                                                                                                                                                           |
| `'nutrition'`      | Nutrition information in the form Nutrition information in the form <br> \[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), <br>saturated fat (PDV), carbohydrates (PDV)\]; PDV stands for “percentage of daily value” |
| `'n_steps'`        | Number of steps in recipe                                                                                                                                                                                                                      |
| `'steps'`          | Text for recipe steps, in order                                                                                                                                                                                                                |
| `'description'`    | User-provided description                                                                                                                                                                                                                      |
| `'ingredients'`    | Text for ingredients used in recipe                                                                                                                                                                                                            |
| `'n_ingredients'`  | Number of ingredients                                                                                                                                                                                                                          |

2. `RAW_interactions.csv`: 731927 rows of reviews, containing non-unique instances

| Column        | Description         |
| ------------- | ------------------- |
| `'user_id'`   | User ID             |
| `'recipe_id'` | Recipe ID           |
| `'date'`      | Date of interaction |
| `'rating'`    | Rating given        |
| `'review'`    | Review text         |

The columns that are relevant to answering the question are `'name'`, `'id'`, `'minutes'`, `'nutrition'`, `'n_steps'`, `'n_ingredients'`, and `'rating'`. 

---

## Data Cleaning and Exploratory Data Analysis

---

### Data Cleaning

1. Cleaning `'rating'` in `RAW_interactions.csv`: All the invalid rating values of `0` were replaced with `np.NaN` then dropped, because it represents a missing entry. By doing so, it is made sure that all recipes have a valid rating, while keeping the features that are shared across one recipe (e.g. `'minutes'`, `'nutrition'`, `'n_steps'`, `'n_ingredients'`, ...) are intact when merged in the later merge. 

2. Create average rating per recipe: Group the interactions by `'recipe_id'` and aggregate on the `'rating'` column to calculate `'avg_rating'` 

3. Merging the datasets: To bring the features of the recipe and the average rating together, merge `recipes` and `interactions_grouped` on on the intersection of ids (`'id'` and `'recipe_id'`). 

4. Clean `'nutrition'`: The values in the `nutrition` column is a string resembling a list. The individual information are split into separate columns–`'calories'`, `'total_fat'`, `'sugar'`, `'sodium'`, `'protein'`, `'saturated_fat'`, `'carb'`.

5. Unrealistic nutrient values: Significant number of recipes had nutrient values that were unrealistic for a standard single serving (e.g. 45609.0, 36188.8, ...), which suggest input errors or recipe listing for more than a single serving. To account for these outliers, any recipes above 3,000 calories–the upper bound of the required calorie per day according to DGA–were dropped. 

6. Final filtering: The irrelevant columns, `'id'`, `'contributor_id'`, `'recipe_id'`, `'submitted'`, `'name'`, `'description'`, `'steps'`, `'tags'`, `'nutrition'`, `'ingredients'`, were dropped. Then, any rows with missing values were dropped since these are essential features required for building the model. 

The first 5 rows of the cleaned dataset is as follows: 

|     | minutes | n_steps | n_ingredients | avg_rating | calories | total_fat | sugar | sodium | protein | saturated_fat | carbohydrates |
| --- | ------- | ------- | ------------- | ---------- | -------- | --------- | ----- | ------ | ------- | ------------- | ------------- |
| 0   | 40      | 10      | 9             | 4.0        | 138.4    | 10.0      | 50.0  | 3.0    | 3.0     | 19.0          | 6.0           |
| 1   | 45      | 12      | 11            | 5.0        | 595.1    | 46.0      | 211.0 | 22.0   | 13.0    | 51.0          | 26.0          |
| 2   | 40      | 6       | 9             | 5.0        | 194.8    | 20.0      | 6.0   | 32.0   | 22.0    | 36.0          | 3.0           |
| 3   | 120     | 7       | 7             | 5.0        | 878.3    | 63.0      | 326.0 | 13.0   | 20.0    | 123.0         | 39.0          |
| 4   | 90      | 17      | 13            | 5.0        | 267.0    | 30.0      | 12.0  | 12.0   | 29.0    | 48.0          | 2.0           |

---

### Exploratory Data Analysis

#### Univariate Analysis

- **Distribution of Average Recipe Ratings**: This shows that the ratings are right-skewed, meaning that people typically gave positive ratings. This may be due to the fact that platforms as such generally tend to put together reciepes that are widely favored. 

 <iframe
 src="assets/fig1.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

- **Distribution of Number of Ingredients**: Most recipes require 8 to 9 recipes, meaning that the majority of the people prefer a reasonably complex recipe over overly simple or complicated ones. 

 <iframe
 src="assets/fig2.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

#### Bivariate Analysis

- **Average Rating vs. Preparation Time**: Some recipes in the dataset take an extremely long time to prepare, some reaching up to 288000 or 259205 minutes (roughly 180 days). Logarithmic transformation was applied on the `'minutes'` to shrink the scale and better visualize the preparation time. While there is no clear linear relationship between the two features, the majority of recipes are clustered around 3 to 5 log minutes of preparation time. This suggests a similarity with the obsevation made with the distribution of number of ingredients–people usually prefer a reasonably complex and time consuming recipe.

 <iframe
 src="assets/fig3.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

- **Average Rating vs. Calories**: The majority of the recipes are clustered around 0 to 800 calories range, while recipes with higher calories tend to appear less and more scattered out overall. As with the previous plot (Average Rating vs. Preparation Time), many of the recipes are concentrated at the whole number rating values (1, 2, 3, 4, 5). This is likely due to the users typical behavior of rating in 0.5 increments or whole number. Again, there is no evident linear relationship visible. 

 <iframe
 src="assets/fig4.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

#### Interesting Aggregates

**Average Nutrient Value by Rating Bins**:
- `'calories'` tend to fluctuate around 1.25 to 3 average rating, then stabilizes around 3.25 average rating.
- All other average nutrient values tend do now show prominent shifts like `'calories'`, but there are some interesting observations to point out.
    - `'sodium'`, marked by red shows a subtle rise at 1.25 average rating, suggesting that overly salty recipes are not favored. This is interesting because the umami taste or characteristic in a dish is mainly correlated with saltiness.
    - As with the `'calories'`, `'sugar'` value slightly fluctuates on the lower half of the average rating, then flattens out, which suggests that sweeter recipes are not always favored.
    - Not as noticeable as the others, but most of the features seems to reach an ideal point as the average rating increases.  

| Rating Bin  | Nutrient      | Mean Nutrient |
| ----------- | ------------- | ------------- |
| (1.25, 1.5] | calories      | 509.23        |
| (1.5, 1.75] | calories      | 211.80        |
| (1.75, 2.0] | calories      | 406.70        |
| (2.0, 2.25] | calories      | 233.48        |
| (2.25, 2.5] | calories      | 336.89        |
| ...         | ...           | ...           |
| (4.0, 4.25] | carbohydrates | 12.43         |
| (4.25, 4.5] | carbohydrates | 12.29         |
| (4.5, 4.75] | carbohydrates | 11.63         |
| (4.75, 5.0] | carbohydrates | 12.17         |
| NaN         | carbohydrates | 13.64         |

 <iframe
 src="assets/fig5.html"
 width="800"
 height="600"
 frameborder="0"
 ></iframe>

---

## Framing a Prediction Problem