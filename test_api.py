import requests

API_KEY = "d09dba68e0e177c17926b55ce89d739c"
city = "Kochi"

url = f"https://api.openweathermap.org/geo/1.0/direct?q={city},IN&limit=1&appid={API_KEY}"
response = requests.get(url)

print("Status code:", response.status_code)
print("Response:", response.json())