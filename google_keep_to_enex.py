#!/usr/bin/env python3

import sys
import re
from datetime import datetime
import time
import json

def mungefile(fn):
    fp = open(fn, 'r')
    data = fp.read()
    my_json = json.loads(data)

    title = my_json["title"].replace(u"\u200e", "").strip()
    tags = ''

    if my_json["isArchived"] == True :
        tags = '<tag>archived</tag>'
    date = datetime.fromtimestamp(my_json["userEditedTimestampUsec"]/1000000)

    iso = date.strftime('%Y%m%dT%H%M%SZ')

    content = my_json["textContent"].replace(u"\u200e", "").replace("\n", "<br>").strip()

    fp.close()

    content = ('''
  <note>
    <title>{title}</title>
    <content><![CDATA[<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE en-note SYSTEM "http://xml.evernote.com/pub/enml2.dtd"><en-note style="word-wrap: break-word; -webkit-nbsp-mode: space; -webkit-line-break: after-white-space;">{content}</en-note>]]></content>
    <created>{iso}</created>
    <updated>{iso}</updated>
    {tags}
    <note-attributes>
      <latitude>0</latitude>
      <longitude>0</longitude>
      <source>google-keep</source>
      <reminder-order>0</reminder-order>
    </note-attributes>
  </note>
'''.format(**locals()))


    #exit()
    print(content)

print ('''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE en-export SYSTEM "http://xml.evernote.com/pub/evernote-export3.dtd">
<en-export export-date="20180502T065115Z" application="Evernote" version="Evernote Mac 6.10 (454269)">''')
for arg in sys.argv[1:]:
    mungefile(arg)
print ('''</en-export>''')
