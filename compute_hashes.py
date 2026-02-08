import hashlib
for p in ['uploads\\img1.jpg','test_case\\img1.jpg']:
    try:
        b=open(p,'rb').read()
        print(p, len(b), hashlib.md5(b).hexdigest())
    except Exception as e:
        print('err',p,e)
