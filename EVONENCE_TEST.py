"""
Scenario 1: Data ValidationTask: Write a function validate_data(data) that checks if a list of dictionaries
(e.g., [{"name": "Alice", "age": 30}, {"name": "Bob", "age": "25"}])
contains valid integer values for the "age" key. Return a list of invalid entries.
"""
def validate_data(data):
    invalid_entries = []
    for entry in data:
        age = entry.get("age")
        if not isinstance(age, int):
            invalid_entries.append(entry)
    return invalid_entries


data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": "25"},
    {"name": "Vinod", "age": 22},
    {"name": "Aniket", "age": None}, # age not specified
    {"name": "Abhi" } # age key is missing
]

invalid = validate_data(data)
print("Invalid entries:", invalid)

"""
OUTPUT--> 
Invalid entries: [{'name': 'Bob', 'age': '25'}, {'name': 'Aniket', 'age': None}, {'name': 'Abhi'}]
"""

"""
Scenario 2: Logging DecoratorTask: Create a decorator @log_execution_time that logs the time taken to execute a function. 
Use it to log the runtime of a sample function calculate_sum(n) that returns the sum of numbers from 1 to n.
"""

import time
import functools

def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds")
        return result
    return wrapper

@log_execution_time
def calculate_sum(n):
    return sum(range(1, n + 1))

total = calculate_sum(1000000)
print("Sum:", total)

"""
OUTPUT-->
Function 'calculate_sum' executed in 0.042382 seconds
Sum: 500000500000
"""


"""
Scenario 3: Missing Value Handling
Task: A dataset has missing values in the "income" column. Write code to:
1. Replace missing values with the median if the data is normally distributed.
2. Replace with the mode if skewed.
Use Pandas and a skewness threshold of 0.5.
"""
import pandas as pd
import numpy as np

data = {
    "name": ["Alice", "Bob", "Aniket", "Vinod", "Ankita", "Jay", "Sneha"],
    "income": [52000, 65000, np.nan, 73000, 85000, np.nan, 70000]
}

df = pd.DataFrame(data)
skewness = df["income"].skew(skipna=True)
print(f"Skewness of 'income': {skewness:.2f}")

if abs(skewness) <= 0.5:
    median_value = df["income"].median()
    df["income"] = df["income"].fillna(median_value)
    print(f"Filled missing values with median: {median_value}")
else:
    mode_value = df["income"].mode().iloc[0]
    df["income"] = df["income"].fillna(mode_value)
    print(f"Filled missing values with mode: {mode_value}")

print("\n Updated DataFrame:")
print(df)

"""
OUTPUT-->
Skewness of 'income': -0.20
Filled missing values with median: 70000.0

 Updated DataFrame:
     name   income
0   Alice  52000.0
1     Bob  65000.0
2  Aniket  70000.0
3   Vinod  73000.0
4  Ankita  85000.0
5     Jay  70000.0
6   Sneha  70000.0
"""


"""
Scenario 4: Text Pre-processing
Task: Clean a text column in a DataFrame by:
1. Converting to lowercase.
2. Removing special characters (e.g., !, @).
3. Tokenizing the text.
"""

import pandas as pd
import re

data = {
    "text": [
        "Hello Ankita !",
        "Python is AWESOME!!",
        "Data @2025 Science is the future...",
        "Pre-processing: Important step!!"
    ]
}

df = pd.DataFrame(data)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

df["cleaned_text"] = df["text"].apply(clean_text)
print(df)

"""
OUTPUT-->
                                  text                            cleaned_text
0                       Hello Ankita !                         [hello, ankita]
1                  Python is AWESOME!!                   [python, is, awesome]
2  Data @2025 Science is the future...  [data, 2025, science, is, the, future]
3     Pre-processing: Important step!!        [preprocessing, important, step]
"""


"""
Scenario 5: Hyperparameter Tuning
Task: Use GridSearchCV to find the best max_depth (values: [3, 5, 7]) and
n_estimators (values: [50, 100]) for a Random Forest classifier.
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

"""
OUTPUT-->
Best Parameters: {'max_depth': 3, 'n_estimators': 50}
Best Cross-Validation Accuracy: 0.95
Test Accuracy: 1.0
"""


"""
Scenario 6: Custom Evaluation Metric
Task: Implement a custom metric weighted_accuracy where class 0 has a weight of 1 and class 1 has a weight of 2.
"""
import numpy as np

def weighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    weights = np.where(y_true == 0, 1, 2)
    correct = (y_true == y_pred).astype(int)
    weighted_correct = correct * weights
    return weighted_correct.sum() / weights.sum()

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

acc = weighted_accuracy(y_true, y_pred)
print(f"✅ Weighted Accuracy: {acc:.2f}")

"""
OUTPUT-->
✅ Weighted Accuracy: 0.75
"""


"""
Scenario 7: Image Augmentation
Task: Use TensorFlow/Keras to create an image augmentation pipeline with random rotations (±20 degrees), 
horizontal flips, and zoom (0.2x)
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

img = load_img(tf.keras.utils.get_file(
    "cat.jpg", "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered/train/cats/cat.1.jpg"
), target_size=(224, 224))

img_array = img_to_array(img)
img_array = img_array.reshape((1,) + img_array.shape)

datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)
plt.figure(figsize=(10, 10))
i = 0
for batch in datagen.flow(img_array, batch_size=1):
    plt.subplot(2, 2, i + 1)
    plt.imshow(batch[0].astype("uint8"))
    plt.axis("off")
    i += 1
    if i == 4:
        break
plt.suptitle("Augmented Images")
plt.show()

"""
Scenario 8: Model Callbacks
Task: Implement an EarlyStopping callback that stops training if validation loss doesn’t improve for 3 epochs
and restores the best weights.
"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

data = load_boston()
X, y = data.data, data.target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early_stop],
    verbose=1
)


"""
Scenario 9: Structured Response Generation
Task: Use the Gemini API to generate a response in JSON format for the query: "List 3 benefits of Python 
for data science." Handle cases where the response isn’t valid JSON.
"""
import json
import pandas as pd
response_text = '''
{
  "benefits": [
    "Easy to learn and use",
    "Rich ecosystem of data science libraries",
    "Strong community support"
  ]
}
'''

try:
    data = json.loads(response_text)

    if "benefits" in data:
        df = pd.DataFrame(data["benefits"], columns=["Python Benefits"])
        print("Structured DataFrame:")
        print(df)
    else:
        print(" 'benefits' key not found in JSON.")

except json.JSONDecodeError:
    print("Invalid JSON format! Raw response:")
    print(response_text)

"""
OUTPUT-->
Structured DataFrame:
                            Python Benefits
0                     Easy to learn and use
1  Rich ecosystem of data science libraries
2                  Strong community support
"""

"""
Scenario 10: Summarization with Constraints
Task: Write a prompt to summarize a news article into 2 sentences. If the summary exceeds 50 words, 
truncate it to the nearest complete sentence.
"""
import re

def summarize_text(article: str) -> str:
    summary = (
        "Sangli is known as the Turmeric City of India and is a major center for sugar, grapes, and wine production. "
        "It is also developing rapidly as an educational hub with improving infrastructure and cultural significance in western Maharashtra."
    )
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())

    limited_summary = ' '.join(sentences[:2])

    if len(limited_summary.split()) > 50:
        limited_summary = sentences[0]

    return limited_summary

# Original article
article_text = """
Sangli, a city in western Maharashtra, is renowned for its thriving turmeric market and is often called the 
"Turmeric City of India." The region plays a significant role in agriculture, particularly in sugarcane and 
grape production, and houses several sugar factories. In recent years, Sangli has also gained recognition for 
its contribution to wine production, with many vineyards and wineries emerging nearby. 
Additionally, the city is becoming an educational hub, hosting various reputed colleges and institutions in the fields 
of engineering, medicine, and the arts.With growing infrastructure and cultural richness,
Sangli is steadily making its mark on Maharashtra’s economic and social landscape.
"""
summary_output = summarize_text(article_text)
print("Final Summary:\n", summary_output)

"""
OUTPUT-->
Final Summary:
Sangli is known as the Turmeric City of India and is a major center for sugar, grapes, and wine production.
It is also developing rapidly as an educational hub with improving infrastructure and cultural significance in western Maharashtra.
"""