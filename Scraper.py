import pandas as pd
import numpy as np
import requests
import re
from concurrent.futures import ThreadPoolExecutor
from time import sleep

div_regex = re.compile('<div')
div_close_regex = re.compile('</div>')
span_regex = re.compile('<span>.*?</span>')
div_class_info_regex = re.compile('<div class="info">\n.*?\n</div>')
div_price_regex = re.compile(
    '(<div class="price ">\n.*?</div>|<div class="price DOWN">\n.*?</div>)')
div_model_regex = re.compile('<a href=\".*?\" class="title saveSlink">.*?</a>')
div_location_regex = re.compile('<div class="location">.*?</div>')
href_regex = re.compile('www.mobile.bg/obiava.*?"')
mp_label_regex = re.compile('<div class="mpLabel">.*?</div>')
mp_info_regex = re.compile('<div class="mpInfo">.*?</div>')
div_start_plus_end_regex = re.compile('<div>.*?</div>')
div_remover_regex = re.compile('</{0,1}(div|span|a).*?>')
span_title_regex = re.compile('<span class="Title">.*?</span>')
item_div_regex = re.compile('<div.*?>.+')
picture_regex = re.compile('src=".*"')
big_picture_regex = re.compile('data-src=".*"')


def first(a):
    return '' if len(a) == 0 else a[0]


def combine_labels(source):
    result = source[0]
    for labels in source[1:]:
        result += labels
    return list(set(result))


def get_as_dicts(keys, values):
    result = []
    for labels, data in zip(keys, values):
        current = {}
        for label, value in zip(labels, data):
            current[label] = value
        result.append(current)
    return result


def clean_nested(source):
    source_patched = []
    for nested1_source in source:
        nested1_source_patched = []
        for nested2 in nested1_source:
            nested1_source_patched.append(
                div_remover_regex.sub('', nested2).strip())
        source_patched.append(nested1_source_patched)
    return source_patched


def data_cleaner(func):
    def clean_cars_data(self, page_url, sleep_time):
        cars_models, cars_description, cars_prices, cars_params, individual_urls, infos_labels, infos_details, tech_labels, tech_details, locations, all_titles, all_items, status, small_pictures, big_pictures = func(
            self, page_url, sleep_time)
        cars_models_patched = [div_remover_regex.sub(
            '', i).strip() for i in cars_models]
        cars_description_patched = [div_remover_regex.sub(
            '', i).strip() for i in cars_description]
        cars_prices_patched = [div_remover_regex.sub(
            '', i).strip() for i in cars_prices]
        locations_patched = [div_remover_regex.sub(
            '', i).strip() for i in locations]

        cars_params_patched = clean_nested(cars_params)
        infos_labels_patched = clean_nested(infos_labels)
        infos_details_patched = clean_nested(infos_details)
        tech_labels_patched = clean_nested(tech_labels)
        tech_details_patched = clean_nested(tech_details)

        all_titles_patched = clean_nested(all_titles)
        all_items_patched = [clean_nested(i) for i in all_items]
        small_pictures_patched = clean_nested(small_pictures)
        big_pictures_patched = clean_nested(big_pictures)
        return [cars_models_patched, cars_description_patched,
                cars_prices_patched, cars_params_patched,
                individual_urls, infos_labels_patched,
                infos_details_patched, tech_labels_patched,
                tech_details_patched, locations_patched, all_titles_patched, all_items_patched, status, small_pictures_patched, big_pictures_patched]
    return clean_cars_data


class Scraper:
    def __init__(self, url) -> None:
        self.main_url = url

    def first(self, a):
        return '' if len(a) == 0 else a[0]

    def extract_divs(self, text, searched_div):
        line_parsing = False
        result = ''
        results = []
        count = 0
        for line in text.split('\n'):
            line = line.strip()
            # print(line,count)
            if line_parsing:
                if div_regex.search(line):
                    count += 1
                if div_close_regex.search(line):
                    count -= 1

                result += f"{line}\n"
                if count == 0:
                    line_parsing = False
                    results.append(result)
                    result = ''

            else:
                if not line.startswith(searched_div):
                    continue

                line_parsing = True
                result += f"{line}\n"
                count = 1
        return results

    def get_info_of_cars_on_page(self, r):
        cars_top = self.extract_divs(r.text, '<div class="item TOP " id="')
        cars_default = self.extract_divs(r.text, '<div class="item  " id="')
        cars_short = self.extract_divs(r.text, '<div id="shortList6"')
        cars_vip = self.extract_divs(r.text, '<div class="item VIP " id="')
        cars_beset = self.extract_divs(r.text, '<div class="item BEST " id="')
        cars = cars_top+cars_default+cars_short+cars_vip+cars_beset
        cars_params = []
        cars_info = []
        cars_prices = []
        cars_models = []
        individual_urls = []
        locations = []
        status = ['TOP']*len(cars_top)+['Default'] * \
            len(cars_default)+['SHORT']*len(cars_short) + \
            ['VIP']*len(cars_vip) + ['BEST']*len(cars_beset)
        for car in cars:
            cars_params.append(span_regex.findall(car))
            cars_info.append(first(div_class_info_regex.findall(car)))
            cars_prices.append(first(div_price_regex.findall(car)))
            cars_models.append(first(div_model_regex.findall(car)))
            locations.append(first(div_location_regex.findall(car)))

        for car_model in cars_models:
            individual_urls.append(href_regex.findall(car_model)[0][:-1])

        individual_urls
        return (cars_models, cars_info, cars_prices, cars_params, individual_urls, locations, status)

    def get_individual_car_cards_info(self, car_url):
        print("Extracting car:", car_url)
        individual = requests.get(f'https://{car_url}')
        individual.encoding = individual.apparent_encoding

        info_html = self.extract_divs(
            individual.text, '<div class="borderBox carParams">')[0]
        info_labels = mp_label_regex.findall(info_html)
        infos_details = mp_info_regex.findall(info_html)

        tech_html = self.extract_divs(
            individual.text, '<div class="techData">')[0]
        data = div_start_plus_end_regex.findall(tech_html)
        tech_labels = data[0::2]
        tech_details = data[1::2]

        additional = first(self.extract_divs(
            individual.text, '<div class="carExtri">'))
        titles = span_title_regex.findall(additional)
        items_htmls = self.extract_divs(div_close_regex.sub(
            '\n</div>\n', div_regex.sub('\n<div', additional)), '<div class="items">')
        items = []
        for item in items_htmls:
            items.append(item_div_regex.findall(item))

        small_picture_divs = self.extract_divs(
            individual.text, '<div class="smallPicturesGallery')
        small_pictures = []
        for picture in small_picture_divs:
            img = picture_regex.findall(picture)[0][7:-1]
            small_pictures.append(img)

        big_pictures = []
        big_pictures_divs = self.extract_divs(
            individual.text, '<div class="owl-carousel')

        for picture in big_pictures_divs:
            images = picture_regex.findall(picture)
            big_pictures += [i[5:-1] for i in images]
        return (info_labels, infos_details, tech_labels, tech_details, titles, items, small_pictures, big_pictures)

    @data_cleaner
    def get_cars_info(self, page_url, sleep_time=2):
        r = requests.get(page_url)
        r.encoding = r.apparent_encoding
        cars_models, cars_description, cars_prices, cars_params, individual_urls, locations, status = self.get_info_of_cars_on_page(
            r)
        infos_labels = []
        infos_details = []
        tech_labels = []
        tech_details = []
        all_titles = []
        all_items = []
        all_small_pictures = []
        all_big_pictures = []
        for url in individual_urls:
            sleep(sleep_time)
            try:
                info_label, info_detail, tech_label, tech_detail, titles, items, small_pictures, big_pictures = self.get_individual_car_cards_info(
                    url)
                infos_labels.append(info_label)
                infos_details.append(info_detail)
                tech_labels.append(tech_label)
                tech_details.append(tech_detail)
                all_titles.append(titles)
                all_items.append(items)
                all_small_pictures.append(small_pictures)
                all_big_pictures.append(big_pictures)
            except Exception as e:
                print("Error extracting individual car data:", e)
                infos_labels.append([])
                infos_details.append([])
                tech_labels.append([])
                tech_details.append([])
                all_titles.append([])
                all_items.append([])
                all_small_pictures.append([])
                all_big_pictures.append([])
                continue
        return [cars_models, cars_description, cars_prices, cars_params, individual_urls, infos_labels, infos_details, tech_labels, tech_details, locations, all_titles, all_items, status, all_small_pictures, all_big_pictures]

    def get_cars_in_pages(self, pages, url, sleep_time=2, low_price=0, high_price=0):
        # cars_data=get_cars_info(URL)
        cars_data = self.get_cars_info(
            f'{url}/p-{int(pages[0])}' + f'?price={low_price}&price1={high_price}' if low_price > 0 or high_price > 0 else '', sleep_time)
        for i in pages[1:]:
            print("Extracting page:", i)
            cars = self.get_cars_info(
                f'{url}/p-{int(i)}' + f'?price={low_price}&price1={high_price}' if low_price > 0 or high_price > 0 else '', sleep_time)
            for j in range(len(cars_data)):
                cars_data[j] = cars_data[j] + cars[j]
        return cars_data

    def get_cars_in_pages_threaded(self, pages, threads, sleep_time=2):
        with ThreadPoolExecutor(threads) as pool:
            pages = np.array_split(pages, threads)
            result = [pool.submit(self.get_cars_in_pages,
                                  i, sleep_time) for i in pages]
            cars_data = result[0].result()
            for i in result[1:]:
                cars = i.result()
                for j in range(len(cars_data)):
                    cars_data[j] = cars_data[j] + cars[j]
        return cars_data

    def get_brands(self):
        with open("brands.txt", "r", encoding="utf8") as f:
            text = '\n'.join(f.readlines())
        divs = self.extract_divs(text, '<div class="a" ')
        brands = [(
            span_regex.findall(div)[0].replace(
                '<span>', '').replace('</span>', ''),
            span_regex.findall(div)[1].replace(
                '<span>', '').replace('</span>', '')) for div in divs]
        return brands

    def scrape_pages(self, start=1, end=10, url=None, sleep_time=2, low_price=0, high_price=0, results_file="scraping_results/latest.csv"):
        # cars_data=get_cars_in_pages(np.linspace(2,150,size,dtype=np.int64))
        # cars_data=get_cars_in_pages_threaded(np.arange(2,10),4)
        print("Scraping url:", url)
        if url is None:
            url = self.main_url
        cars_data = self.get_cars_in_pages(
            np.arange(start, end), url, sleep_time, low_price=low_price, high_price=high_price)

        all_info_labels = combine_labels(cars_data[-10])
        infos_as_dicts = get_as_dicts(cars_data[5], cars_data[6])
        tech_as_dict = get_as_dicts(cars_data[7], cars_data[8])
        additional_data_for_car = get_as_dicts(cars_data[10], cars_data[11])
        df = pd.DataFrame({'CarModel': cars_data[0], 'CarDescription': cars_data[1], 'CarPrice': cars_data[2],
                          'Params': cars_data[3], 'URL': cars_data[4], "Location": cars_data[9], 'Status': cars_data[12], "SmallPictures": cars_data[13], "BigPictures": cars_data[14]})
        df = df.join(pd.DataFrame(tech_as_dict), how='outer').join(pd.DataFrame(
            infos_as_dicts), lsuffix='Tech', rsuffix='Info').join(pd.DataFrame(additional_data_for_car))
        for label in all_info_labels:
            df_copy = df.dropna(subset=[f"{label}Tech", f"{label}Info"])
            if (df_copy[f"{label}Tech"] != df_copy[f"{label}Info"]).sum():
                print(label)
        # df.to_csv(results_file)
        return df

    def scrape_main_pages(self, start=1, end=10, sleep_time=2, results_file="scraping_results/latest.csv", low_price=0, high_price=0):
        print('Extracting Cars')
        df = self.scrape_pages(
            start, end, self.main_url, sleep_time, low_price=low_price, high_price=high_price)
        print("Cars Extracted")
        df.to_csv(results_file)

    def scrape_by_brands(self, brands=None, sleep_time=2, results_file="scraping_results/latest.csv"):
        if brands is None:
            brands = self.get_brands()
        print('Extracting Cars for brand')

        df = pd.concat([self.scrape_pages(start=1, end=min(count/20+1, 150),
                                          url=self.main_url + "/" + brand.lower().replace(' ', '-'), sleep_time=sleep_time)
                        for brand, count in brands])
        print("Cars Extracted")
        df.to_csv(results_file)

    def scrape_by_brands_and_page(self, brands=None, start=1, end=150, sleep_time=2, results_file="scraping_results/latest_bmw3.csv"):
        if brands is None:
            brands = self.get_brands()
        print('Extracting Cars for brand')

        df = pd.concat([self.scrape_pages(start=start, end=end,
                                          url=self.main_url + "/" + brand.lower().replace(' ', '-'), sleep_time=sleep_time)
                        for brand in brands])
        print("Cars Extracted")
        df.to_csv(results_file)


if __name__ == "__main__":
    scraper = Scraper('https://www.mobile.bg/obiavi/avtomobili-dzhipove')
    print(scraper.get_brands())
    # scraper.scrape_by_brands(brands=[
    #                        ('BMW', 20), ('Mercedes-Benz', 19), ('Audi', 18)], sleep_time=5)

    # scraper.scrape_by_brands_and_page(
    #    brands=['BMW'], start=1, end=100, sleep_time=2, results_file="scraping_results/latest_bmw_big.csv")

    scraper.scrape_main_pages(start=1, end=150, sleep_time=2,
                              results_file="scraping_results/latest_1000000_10000000.csv", low_price=1000000, high_price=10000000)
