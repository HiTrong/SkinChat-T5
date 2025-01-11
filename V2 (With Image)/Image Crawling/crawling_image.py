# Import Library
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

import urllib
import time
import json
import os

# Load json file
with open("skin_sick.json", "r", encoding='utf-8') as file:
    sick_dict = json.load(file)
    


# crawling
for sick_name in sick_dict:
    sick_config = sick_dict[sick_name]
    folder_path = sick_config["img_folder"]
    if any(os.path.isfile(os.path.join(folder_path, file)) for file in os.listdir(folder_path)):
        continue
    
    url = str("https://www.google.com/search?q=dấu+hiệu+của+{}&hl=en&tbm=isch&sxsrf=APwXEdeMCCcn15mo1obWv-xVcr_tpnFYQg%3A1684476865544&source=hp&biw=1737&bih=1032&ei=wRNnZNX-HtX4kPIP9umT2AY&iflsig=AOEireoAAAAAZGch0VXQnHgSIAIKBwcg5h0gf-nJjQvD&oq=toyota+supr&gs_lcp=CgNpbWcQAxgAMgQIIxAnMgQIIxAnMggIABCABBCxAzIICAAQgAQQsQMyBQgAEIAEMggIABCABBCxAzIICAAQgAQQsQMyCAgAEIAEELEDMggIABCABBCxAzIICAAQgAQQsQM6BwgjEOoCECc6CAgAELEDEIMBOgQIABADOgkIABAYEIAEEApQlglY7SNgpSxoB3AAeAGAAZIBiAHkDJIBBDEwLjeYAQCgAQGqAQtnd3Mtd2l6LWltZ7ABCg&sclient=img".format(sick_name.replace(" ","+")))
    # Configure driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    # Điều hướng đến trang web
    driver.get(url)
            
    # Lấy chiều cao ban đầu của trang
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        
    img_results = driver.find_elements(By.XPATH, "//img[contains(@class, 'YQ4gaf') and not(contains(@class, 'zr758c'))]")
    image_urls = []
    for img in img_results:
        image_urls.append(img.get_attribute('src'))
    folder_path = sick_config["img_folder"]
    
    counter = 0
    for i in range(len(image_urls)):
        counter += 1
        try:
            urllib.request.urlretrieve(str(image_urls[i]), folder_path + "{}.jpg".format(counter))
        except:
            counter -= 1
            continue
    driver.quit()


