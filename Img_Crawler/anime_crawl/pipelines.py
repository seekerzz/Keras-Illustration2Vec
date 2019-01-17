# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from scrapy import Request
from scrapy.pipelines.images import ImagesPipeline
from scrapy.utils.project import get_project_settings
import os
class ImagePipeline(ImagesPipeline):
    def file_path(self, request, response=None, info=None):
        url = request.url
        id = url[url.find("?")+len("?"):]
        file_type=url[url.rfind("."):url.find("?")]
        return id+file_type

    def get_media_requests(self, item, info):
        yield Request("http:"+item['img_src'][0])

class AnimeCrawlPipeline(object):
    def process_item(self, item, spider):
        settings = get_project_settings()
        saved_path = settings.get("IMAGES_STORE")
        with open(os.path.join(saved_path,item["id"]+".txt"),"w") as f:
            for t in item["tags"]:
                f.write(t+"\n")
        return item
