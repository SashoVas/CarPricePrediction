import pandas as pd
import numpy as np
import requests
import re
from concurrent.futures import ThreadPoolExecutor


div_regex=re.compile('<div')
div_close_regex=re.compile('</div>')
span_regex=re.compile('<span>.*?</span>')
div_class_info_regex=re.compile('<div class="info">\n.*?\n</div>')
div_price_regex=re.compile('<div>[0-9 ]+ лв.</div>')
div_model_regex=re.compile('<a href=\".*?\" class="title saveSlink">.*?</a>')
div_location_regex=re.compile('<div class="location">.*?</div>')
href_regex=re.compile('www.mobile.bg/obiava.*?"')
mp_label_regex=re.compile('<div class="mpLabel">.*?</div>')
mp_info_regex=re.compile('<div class="mpInfo">.*?</div>')
div_start_plus_end_regex=re.compile('<div>.*?</div>')
div_remover_regex=re.compile('</{0,1}(div|span|a).*?>')
span_title_regex=re.compile('<span class="Title">.*?</span>')
item_div_regex=re.compile('<div.*?>.+')

def first(a):
    return '' if len(a)==0 else a[0]

def combine_labels(source):
    result=source[0]
    for labels in source[1:]:
        result+=labels
    return list(set(result))

def get_as_dicts(keys,values):
    result=[]
    for labels,data in zip(keys,values):
        current={}
        for label,value in zip(labels,data):
            current[label]=value
        result.append(current)
    return result

def clean_nested(source):
    source_patched=[]
    for nested1_source in source:
        nested1_source_patched=[]
        for nested2 in nested1_source:
            nested1_source_patched.append(div_remover_regex.sub('',nested2).strip())
        source_patched.append(nested1_source_patched)   
    return source_patched

def data_cleaner(func):
    def clean_cars_data(self,page_url):
        cars_models,cars_description,cars_prices,cars_params,individual_urls,infos_labels,infos_details,tech_labels,tech_details,locations,all_titles,all_items,status=func(self,page_url)
        cars_models_patched=[div_remover_regex.sub('',i).strip() for i in cars_models]
        cars_description_patched=[div_remover_regex.sub('',i).strip() for i in cars_description]
        cars_prices_patched=[div_remover_regex.sub('',i).strip() for i in cars_prices]
        locations_patched=[div_remover_regex.sub('',i).strip() for i in locations]

        cars_params_patched=clean_nested(cars_params)
        infos_labels_patched=clean_nested(infos_labels)
        infos_details_patched=clean_nested(infos_details)
        tech_labels_patched=clean_nested(tech_labels)
        tech_details_patched=clean_nested(tech_details)

        all_titles_patched=clean_nested(all_titles)
        all_items_patched=[clean_nested(i) for i in all_items]
        return [cars_models_patched,cars_description_patched,
                cars_prices_patched,cars_params_patched,
                individual_urls,infos_labels_patched,
                infos_details_patched,tech_labels_patched,
                tech_details_patched,locations_patched
                ,all_titles_patched,all_items_patched,status]
    return clean_cars_data


class Scraper:
    def __init__(self,url) -> None:
        self.main_url=url

    def first(self,a):
        return '' if len(a)==0 else a[0]

    def extract_divs(self,text,searched_div):
        line_parsing=False
        result=''
        results=[]
        count=0
        for line in text.split('\n'):
            line=line.strip()
            #print(line,count)
            if line_parsing:
                if div_regex.search(line):
                    count+=1
                if div_close_regex.search(line):
                    count-=1

                result+=f"{line}\n"
                if count==0:
                    line_parsing=False
                    results.append(result)
                    result=''

            else:
                if not line.startswith(searched_div) :
                    continue

                line_parsing=True
                result+=f"{line}\n"
                count=1
        return results

    def get_info_of_cars_on_page(self,r):
        cars_top=self.extract_divs(r.text,'<div class="item TOP " id="')
        cars_default=self.extract_divs(r.text,'<div class="item  " id="')
        cars_short=self.extract_divs(r.text,'<div id="shortList6"')
        cars=cars_top+cars_default+cars_short
        cars_params=[]
        cars_info=[]
        cars_prices=[]
        cars_models=[]
        individual_urls=[]
        locations=[]
        status=['TOP']*len(cars_top)+['Default']*len(cars_default)+['SHORT']*len(cars_short)
        for car in cars:
            cars_params.append(span_regex.findall(car))
            cars_info.append(first(div_class_info_regex.findall(car)))
            cars_prices.append(first(div_price_regex.findall(car)))
            cars_models.append(first(div_model_regex.findall(car)))
            locations.append(first(div_location_regex.findall(car)))

        for car_model in cars_models:
            individual_urls.append(href_regex.findall(car_model)[0][:-1])

        individual_urls
        return (cars_models,cars_info,cars_prices,cars_params,individual_urls,locations,status)
    
    def get_individual_car_cards_info(self,car_url):
        individual=requests.get(f'https://{car_url}')
        individual.encoding = individual.apparent_encoding

        info_html=self.extract_divs(individual.text,'<div class="borderBox carParams">')[0]
        info_labels=mp_label_regex.findall(info_html)
        infos_details=mp_info_regex.findall(info_html)

        tech_html=self.extract_divs(individual.text,'<div class="techData">')[0]
        data=div_start_plus_end_regex.findall(tech_html)
        tech_labels=data[0::2]
        tech_details=data[1::2]

        additional=first(self.extract_divs(individual.text,'<div class="carExtri">'))
        titles=span_title_regex.findall(additional)
        items_htmls=self.extract_divs(div_close_regex.sub('\n</div>\n',div_regex.sub('\n<div',additional)),'<div class="items">')
        items=[]
        for item in items_htmls:
            items.append(item_div_regex.findall(item))
        return (info_labels,infos_details,tech_labels,tech_details,titles,items)
    
    @data_cleaner
    def get_cars_info(self,page_url):
        r=requests.get(page_url)
        r.encoding = r.apparent_encoding
        cars_models,cars_description,cars_prices,cars_params,individual_urls,locations,status=self.get_info_of_cars_on_page(r)
        infos_labels=[]
        infos_details=[]
        tech_labels=[]
        tech_details=[]
        all_titles=[]
        all_items=[]
        for url in individual_urls:
            info_label,info_detail,tech_label,tech_detail,titles,items=self.get_individual_car_cards_info(url)
            infos_labels.append(info_label)
            infos_details.append(info_detail)
            tech_labels.append(tech_label)
            tech_details.append(tech_detail)
            all_titles.append(titles)
            all_items.append(items)
        return [cars_models,cars_description,cars_prices,cars_params,individual_urls,infos_labels,infos_details,tech_labels,tech_details,locations,all_titles,all_items,status]
    
    def get_cars_in_pages(self,pages):
        #cars_data=get_cars_info(URL)
        cars_data=self.get_cars_info(f'{self.main_url}/p-{int(pages[0])}')
        for i in pages[1:]:
            cars=self.get_cars_info(f'{self.main_url}/p-{int(i)}')
            for j in range(len(cars_data)):
                cars_data[j]=cars_data[j]  + cars[j]
        return cars_data
    
    def get_cars_in_pages_threaded(self,pages,threads):
        with ThreadPoolExecutor(threads) as pool:
            pages=np.array_split(pages,threads)
            result=[pool.submit(self.get_cars_in_pages,i) for i in pages]
            cars_data=result[0].result()
            for i in result[1:]:
                cars=i.result()
                for j in range(len(cars_data)):
                    cars_data[j]=cars_data[j]  + cars[j]
        return cars_data
    
    def scraper_main_pages(self,start=2,end=10):
        #cars_data=get_cars_in_pages(np.linspace(2,150,size,dtype=np.int64))
        #cars_data=get_cars_in_pages_threaded(np.arange(2,10),4)
        cars_data=self.get_cars_in_pages(np.arange(start,end))

        all_info_labels=combine_labels(cars_data[-8])
        infos_as_dicts=get_as_dicts(cars_data[5],cars_data[6])
        tech_as_dict=get_as_dicts(cars_data[7],cars_data[8])
        additional_data_for_car=get_as_dicts(cars_data[10],cars_data[11])
        df=pd.DataFrame({'CarModel':cars_data[0],'CarDescription':cars_data[1],'CarPrice':cars_data[2],'Params':cars_data[3],'URL':cars_data[4],"Location":cars_data[9],'Status':cars_data[12]})
        df=df.join(pd.DataFrame(tech_as_dict),how='outer').join(pd.DataFrame(infos_as_dicts),lsuffix='Tech',rsuffix='Info').join(pd.DataFrame(additional_data_for_car))
        for label in all_info_labels:
            df_copy=df.dropna(subset=[f"{label}Tech",f"{label}Info"])
            if (df_copy[f"{label}Tech"]!=df_copy[f"{label}Info"]).sum():
                print(label)
        df.to_csv('scraping_results/latest.csv')
        print("Scraping Done")

if __name__=="__main__":
    scraper=Scraper('https://www.mobile.bg/obiavi/avtomobili-dzhipove')
    scraper.scraper_main_pages()
    