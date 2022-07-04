from geopy.geocoders import Nominatim
import geocoder

# initialize the Nominatim object
Nomi_locator = Nominatim(user_agent="My App")

my_location = geocoder.ip('me')
print(my_location.latlng)

# my latitude and longitude coordinates
latitude = my_location.geojson['features'][0]['properties']['lat']
longitude = my_location.geojson['features'][0]['properties']['lng']

# get the location
location = Nomi_locator.reverse(f"{latitude}, {longitude}")
print("Your Current IP location is", location)
