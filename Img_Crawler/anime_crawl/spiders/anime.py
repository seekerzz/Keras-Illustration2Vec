# -*- coding: utf-8 -*-
import scrapy
from anime_crawl.items import AnimeCrawlItem
from scrapy import Request
import re
import json

class Animespider(scrapy.Spider):
    name = 'anime'
    allowed_domains = ['safebooru.org']

    def __init__(self, frac, maxfrac, *args, **kwargs):
        super(Animespider, self).__init__(*args, **kwargs)
        self.frac = int(frac)
        self.maxfrac = int(maxfrac)
        self.url_head = "http://safebooru.org/index.php?page=post&s=view&id="

    def start_requests(self):
        maxpage=2724300
        for i in range(self.frac*maxpage//self.maxfrac,(self.frac+1)*maxpage//self.maxfrac):
            url = self.url_head+str(i)
            yield Request(url, self.parse)

    def parse(self, response):
        item = AnimeCrawlItem()
        if (response.url.find("id")>0):
            id = response.url[response.url.find("id=")+len("id="):]
            tags = response.xpath('//*[@id="tag-sidebar"]/li/a/text()').extract()
            types = response.xpath('//*[@id="tag-sidebar"]/li/@class').extract()
            img_src = response.xpath('//*[@id="image"]//attribute::src').extract()
            item["id"] = id
            item["tags"] = tags
            item["types"] = types
            item["img_src"] = img_src
            return item

