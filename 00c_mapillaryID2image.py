import pandas as pd
import subprocess
import json,urllib.request,requests

df=pd.DataFrame(pd.read_csv("Piacenza/piacenza.csv"))
path="Piacenza/img/original/"
subprocess.run(["mkdir",path])
pre_url="https://graph.mapillary.com/"
suff_url="?access_token=MLY|4463150933761310|5995ca3757fc4f9a9c8f5e96b2efaa03&fields=thumb_original_url"
list_id=list(df["id"])

for i in range(len(list_id)):
    _id=list_id[i]
    url=pre_url+str(_id)+suff_url
    with urllib.request.urlopen(url) as _u:
        img_url=json.loads(_u.read().decode())["thumb_original_url"]
        print(img_url)
    with open(path+'{}.jpg'.format(str(_id)), 'wb') as handler:
        image_data = requests.get(img_url, stream=True).content
        handler.write(image_data)
    print(str(i) + " image(s) download in total " + str(len(list_id)))



