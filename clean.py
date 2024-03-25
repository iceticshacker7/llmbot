import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/hp/Documents/Airline_review.csv")

# Filter to remove rows where Verified column is "FALSE"
df_filtered = df[df["Verified"] == True]

# Save the filtered data
df_filtered.to_csv("airline_reviews_verified.csv", index=False)

print("Filtered data saved to airline_reviews_verified.csv")
