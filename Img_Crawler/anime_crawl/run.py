from scrapy import cmdline
import sys

frac = sys.argv[1]
maxfrac = sys.argv[2]
cmdline.execute(("scrapy crawl anime -o tags_%s.csv -a frac=%s -a maxfrac=%s" %(frac,frac,maxfrac)).split())
