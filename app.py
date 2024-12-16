import streamlit as st
import os
from dotenv import load_dotenv
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from fuzzywuzzy import fuzz
import datetime
import pandas as pd
import re
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor


load_dotenv()

st.title("Village Restaurant Price Predictor")

#API_KEYS
YELP_API_KEY = os.environ.get('YELP_API_KEY')
WEATHER_API_KEY = os.environ.get('WEATHER_API_KEY')

#Header
HEADERS = {'Authorization': f'Bearer {YELP_API_KEY}'}

#Selenium setup
chrome_options = Options()
chrome_options.add_argument("--lang=en")
prefs = {
    "intl.accept_languages": "en, en_US",
    "gl": "US",
    "hl": "en",
}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument("--headless")  # Enable headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU for headless
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resources
driver = webdriver.Chrome(options=chrome_options)



## Part 1
business_id = 'village-the-soul-of-india-hicksville'
url = f"https://api.yelp.com/v3/businesses/{business_id}"

response = requests.get(url, headers=HEADERS)
data = response.json()
name = data.get('name')
address = ', '.join(data['location']['display_address'])
hours = data.get('hours', [{}])[0].get('open', [])
schedule = [f"{day['start']} - {day['end']}" for day in hours]

def convert_time_format(time_str):
    if time_str:  # Check if time_str is not empty
        time_obj = datetime.datetime.strptime(time_str, '%H%M')
        return time_obj.strftime('%I:%M %p')
    else:
        return "Invalid Time"

# Convert the time ranges
formatted_times = []
for time_range in schedule:
    start, end = time_range.split(' - ')
    start_time = convert_time_format(start)
    end_time = convert_time_format(end)
    formatted_times.append(f'{start_time} - {end_time}')

menu_url = data['attributes']['menu_url']
categories = data['categories']

village_categories = []
for category in categories:
    village_categories.append(category['alias'])

village_categories = ','.join(village_categories)

latitude = data['coordinates']['latitude']
longitude = data['coordinates']['longitude']

st.subheader("Village Restaurant Details")
st.write(f"**Name :** {name}")
st.write(f"**Address :** {address}")
st.write(f"**Open Time:** {', '.join(formatted_times)}")

driver.get(menu_url)
menus = driver.find_elements(By.CSS_SELECTOR, "div.card-body h5 b")
village_menu = []
for menu in menus:
    village_menu.append(menu.text)

prices = driver.find_elements(By.CSS_SELECTOR, "div.text-muted")

village_prices = []
for price in prices:
    village_prices.append(price.text)

joined_prices = ' '.join(village_prices)

prices = re.findall(r'\$\s?\d+\.\d{2}', joined_prices)

# Clean the prices (optional, remove any leading whitespace after the $ symbol)
prices = [price.replace(" ", "") for price in prices]
prices = [float(price.replace('$', '')) for price in prices]

village_menu_price = list(zip(village_menu, prices))
village_df = pd.DataFrame(village_menu_price, columns=['Food', 'Price'])

st.subheader("Village Restaurant Menu")
st.dataframe(village_df)

##Finding Nearby Restaurants
url = "https://api.yelp.com/v3/businesses/search"
params = {
    'term': 'restaurant',  # Search for restaurants
    'categories': village_categories,  # Similar categories
    'latitude': latitude,  # Latitude of the current restaurant
    'longitude': longitude,  # Longitude of the current restaurant
    'radius': 2000,  # Radius in meters (2 km)
    'sort_by': 'rating',  # Sort by rating
    'limit': 5  # Limit to top 5 results
}

response = requests.get(url, headers=HEADERS, params=params)

data = response.json()
# Extract details for top 5 restaurants
top_restaurants = []
for business in data['businesses']:
    top_restaurants.append({
        'name': business['name'],
        'rating': business['rating'],
        'address': ', '.join(business['location']['display_address']),
        'menu_url': business['attributes']['menu_url'],
        'distance': round(business['distance'] / 1000, 2)  # Convert meters to km
    })

# Print the results
st.subheader("Nearby Restaurants Details")
for idx, restaurant in enumerate(top_restaurants, start=1):
    st.write(f"{idx}. {restaurant['name']} - {restaurant['rating']} stars")
    print(f"   Address: {restaurant['address']}")
    print(f"   Distance: {restaurant['distance']} km")

#Fetching Menuu
# Kunga Kitchen
kunga_url = 'https://www.yelp.com/menu/kunga-kitchen-hicksville'
driver.get(kunga_url)
menus = driver.find_elements(By.CSS_SELECTOR, 'div.arrange_unit.arrange_unit--fill.menu-item-details h4')

prices = driver.find_elements(By.CSS_SELECTOR, 'li.menu-item-price-amount')

kunga_menu = []
for menu in menus:
    kunga_menu.append(menu.text)

kunga_price = []
for price in prices:
    kunga_price.append(price.text.strip())

kunga_price = [float(p.replace('$', '')) for p in kunga_price]

kunga_menu_price = list(zip(kunga_menu, kunga_price))

kunga_df = pd.DataFrame(kunga_menu_price, columns=['Food', 'Price'])

st.write(f"**Kunga Kitchen Menu**")
st.dataframe(kunga_df)

#tasteOfMumbai Menu
tom_url = 'https://www.yelp.com/menu/taste-of-mumbai-hicksville-3'
driver.get(tom_url)
menus = WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.arrange div h4 a')))

tom_menus = []

for menu in menus:
    tom_menus.append(menu.text)

prices = driver.find_elements(By.CSS_SELECTOR, 'li.menu-item-price-amount')
tom_prices = []
for price in prices:
    tom_prices.append(price.text)

tom_prices = [float(p.replace('$', '')) for p in tom_prices]

tom_menu_prices = list(zip(tom_menus, tom_prices))
tom_df = pd.DataFrame(tom_menu_prices, columns=['Food', 'Price'])

st.write(f"**Taste of Mumbai Menu**")
st.dataframe(tom_df)

# kathis&kababs
kk_url = 'https://www.yelp.com/menu/kathis-and-kababs-hicksville'
driver.get(kk_url)
menus = driver.find_elements(By.CSS_SELECTOR, 'div.arrange div h4')
kk_menus = []
for menu in menus:
    kk_menus.append(menu.text.lstrip("1234567890. "))

kk_prices = []
prices = driver.find_elements(By.CSS_SELECTOR, 'li.menu-item-price-amount')
for price in prices:
    kk_prices.append(price.text)

kk_prices = [float(p.replace('$', '')) for p in kk_prices]
kk_menu_prices = list(zip(kk_menus, kk_prices))
kk_df = pd.DataFrame(kk_menu_prices, columns=['Food', 'Price'])
st.write(f"**Kathis And Kababs Menu**")
st.dataframe(kk_df)

# tasteOfChennai

toc_url = 'https://www.yelp.com/menu/taste-of-chennai-hicksville'
driver.get(toc_url)
menus_elements = driver.find_elements(By.CSS_SELECTOR, 'div.arrange div h4 a')
toc_menus = []

for menu in menus_elements:
    toc_menus.append(menu.text)

toc_prices = []
prices_elements = driver.find_elements(By.CSS_SELECTOR, 'li.menu-item-price-amount')
for price in prices_elements:
    toc_prices.append(price.text)

toc_prices = [float(p.replace('$', '')) for p in toc_prices]
toc_menu_prices = list(zip(toc_menus, toc_prices))

toc_df = pd.DataFrame(toc_menu_prices, columns=['Food', 'Price'])

st.write(f"**Taste of Chennai Menu**")
st.dataframe(toc_df)

# Converting menu dfs into dictionaries
village_menu = village_df.set_index('Food')['Price'].to_dict()
kunga_menu = kunga_df.set_index('Food')['Price'].to_dict()
tom_menu = tom_df.set_index('Food')['Price'].to_dict()
kk_menu = kk_df.set_index('Food')['Price'].to_dict()
toc_menu = toc_df.set_index('Food')['Price'].to_dict()

#Finding the lowest price
menus = [village_menu, kunga_menu, tom_menu, kk_menu, toc_menu]

# Initialize a dictionary to store the lowest price for each menu item
lowest_prices = {}

# Function to find the best match for an item from all menus
def get_best_match(item_name, all_menus):
    best_match = None
    highest_score = 0
    for menu in all_menus:
        for menu_item in menu:
            score = fuzz.ratio(item_name.lower(), menu_item.lower())  # Calculate the similarity score
            if score > highest_score:
                highest_score = score
                best_match = menu_item
    return best_match

# Iterate through each menu and find the lowest price for each matched item
for menu in menus:
    for item, price in menu.items():
        best_match = get_best_match(item, menus)

        if best_match not in lowest_prices:
            lowest_prices[best_match] = price
        else:
            lowest_prices[best_match] = min(lowest_prices[best_match], price)

# Print the lowest prices for matched items
lowest_price = {}
for item, price in lowest_prices.items():
    # print(f"{item}: ${price:.2f}")
    lowest_price[item] = price

lowest_price_df = pd.DataFrame(list(lowest_price.items()), columns=['Food', 'Price'])
st.subheader("Lowest Price Among All Restaurants")
st.dataframe(lowest_price_df)

st.subheader("Busy Time And Weather")
## Getting Busy time
map_url = 'https://www.google.com/maps/place/Village+-+The+Soul+of+India/@40.7665603,-73.523538,17z/data=!4m14!1m7!3m6!1s0x89c281752d83843d:0x1f2a365d2207b71c!2sVillage+-+The+Soul+of+India!8m2!3d40.7665603!4d-73.523538!16s%2Fg%2F11thj448_5!3m5!1s0x89c281752d83843d:0x1f2a365d2207b71c!8m2!3d40.7665603!4d-73.523538!16s%2Fg%2F11thj448_5?entry=ttu&g_ep=EgoyMDI0MTIxMS4wIKXMDSoASAFQAw%3D%3D?hl=en&gl=US'

driver.get(map_url)

busy_elements = WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div[role="img"][aria-label]')))

# Extract the aria-label values
busy_times = [element.get_attribute('aria-label') for element in busy_elements]

# Define a threshold for filtering (e.g., discard below 20%)
threshold = 20

times = []
# Print the filtered busy times
for time in busy_times:
    times.append(time)

percentage_pattern = r'(\d+)%'  # Matches the number before a percentage
time_pattern = r'(\d{1,2}\u202f?[APap][Mm])'  # Matches the time (e.g., 6 AM, 6 PM)

# Extract percentages and times
percentages = re.findall(percentage_pattern, " ".join(times))
times = re.findall(time_pattern, " ".join(times))
percentages = [float(p) for p in percentages]
times = [time.replace("\u202f", "") for time in times]
# Print the extracted values

busy_time = []
for i, per in enumerate(percentages):
    if per > 30:
        busy_time.append(times[i])

busy_time = sorted(set(busy_time))
busy_time = ",".join(busy_time)
st.write(f"**Busy Time :** {busy_time}")

#Current Temperature and Rain
weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={WEATHER_API_KEY}'
response = requests.get(weather_url)
data = response.json()

temp_kelvin = data['main']['temp']
temp_fahrenheit = (temp_kelvin - 273.15) * 9 / 5 + 32
temp_fahrenheit = round(temp_fahrenheit, 2)
rain = data.get('rain', {}).get('1h', 0)

st.write(f"**Temperature in Fahrenheit :** {temp_fahrenheit} F")
st.write(f"**Rain in last one hour :** {rain} mm ")

from datetime import datetime

# Determining whether the current time is in busy time or not
# Get the current time
now = datetime.now()

# Convert the current time to 12-hour format with AM/PM
current_time = now.strftime("%-I%p")

if current_time in busy_time:
    is_busy = True
else:
    is_busy = False

# Determining whether there is bad weather or not setting some thresholds
temp_threshold = 45
rain_threshold = 2.5

if temp_fahrenheit < temp_threshold:
    is_badWeather = True
elif rain >= rain_threshold:
    is_badWeather = True
else:
    is_badWeather = False

#Price Adjustment And Model Data Collection
model_data = []
busy_time_multiplier = 1.10  # 10% price increase during busy times
weather_multiplier = 1.15  # 15% price increase for bad weather

def adjust_price(village_menu, lowest_price_df, is_busy, is_badWeather):
    adjusted_prices = {}

    for item, village_price in village_menu.items():
        # Compare the item with the lowest price menu
        lowest_competitor_price = lowest_price_df[lowest_price_df['Food'] == item]['Price']

        # If the item is not found in the competitor list, continue
        if lowest_competitor_price.empty:
            lowest_competitor_price = village_price
        else:
            lowest_competitor_price = lowest_competitor_price.iloc[0]  # Get the price value

        # Adjust price based on conditions
        if is_badWeather or is_busy:
            # Apply the price increase (e.g., 10% increase in busy times, 15% for bad weather)
            adjusted_price = village_price * busy_time_multiplier if is_busy else village_price * weather_multiplier
            # Ensure the price is higher than the lowest competitive price
            adjusted_price = max(adjusted_price, lowest_competitor_price * 1.05)  # 5% more than lowest competitor
        else:
            # If no conditions are met, use the lowest competitor price
            adjusted_price = lowest_competitor_price

        adjusted_prices[item] = adjusted_price

        model_data.append({
            'Food': item,
            'Village_Price': village_price,
            'Competitor_Price': lowest_competitor_price,
            'Temperature_F': temp_fahrenheit,
            'Is_Bad_Weather': is_badWeather,
            'Is_Busy_Time': is_busy,
            'Target_Price': adjusted_price
        })
    return adjusted_prices

adjusted_prices = adjust_price(village_menu, lowest_price_df, is_busy, is_badWeather)

adjusted_prices_df = pd.DataFrame(list(adjusted_prices.items()), columns=['Food', 'Adjusted_Price'])

st.subheader("Adjusted Price")
st.dataframe(adjusted_prices_df)

## Model Creation

model_df = pd.DataFrame(model_data)

#Prepare the dataset
X = model_df[['Village_Price', 'Competitor_Price', 'Temperature_F', 'Is_Bad_Weather', 'Is_Busy_Time']]
y = model_df['Target_Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model (RandomForestRegressor in this case)
model = RandomForestRegressor(random_state=42)

# Perform Cross-Validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')  # 5-fold cross-validation

# Calculate the training accuracy (R²) on the training data
model.fit(X_train, y_train)
train_r2 = r2_score(y_train, model.predict(X_train))

# Calculate the testing accuracy (R²) on the testing data
test_r2 = r2_score(y_test, model.predict(X_test))

# Print the results
st.write(f"**Accuracies of the model**")
st.write(f"**Training Accuracy (R²):** {train_r2:.2f}")
st.write(f"**Testing Accuracy (R²):** {test_r2:.2f}")
st.write(f"**Cross-Validation Accuracy (R²):** {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Predict for current conditions
current_conditions = X.copy()  # Assuming conditions are already in X
model_df['Predicted_Price'] = model.predict(current_conditions)
st.subheader("Model Predictions")
st.dataframe(model_df)
