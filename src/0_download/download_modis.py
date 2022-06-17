from __future__ import (division, print_function, absolute_import, unicode_literals)

import os
import os.path
import shutil
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import csv

try:
    from StringIO import StringIO   # python2
except ImportError:
    from io import StringIO         # python3

USERAGENT = 'tis/download.py_1.0--' + sys.version.replace('\n','').replace('\r','')


def geturl(url, token=None, out=None):
    headers = { 'user-agent' : USERAGENT }
    if not token is None:
        headers['Authorization'] = 'Bearer ' + token
    try:
        import ssl
        CTX = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        if sys.version_info.major == 2:
            import urllib2
            try:
                print(url)
                fh = urllib2.urlopen(urllib2.Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read()
                else:
                    shutil.copyfileobj(fh, out)
            except urllib2.HTTPError as e:
                # print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                # print('HTTP GET error message: %s' % e.message, file=sys.stderr)
                print("urllib2 http出错")
            except urllib2.URLError as e:
                # print('Failed to make request: %s' % e.reason, file=sys.stderr)
                print("urllib2 url出错")
            return None

        else:
            from urllib.request import urlopen, Request, URLError, HTTPError
            try:
                print(url)
                fh = urlopen(Request(url, headers=headers), context=CTX)
                if out is None:
                    return fh.read().decode('utf-8')
                else:
                    shutil.copyfileobj(fh, out)
            except HTTPError as e:
                # print('HTTP GET error code: %d' % e.code(), file=sys.stderr)
                # print('HTTP GET error message: %s' % e.message, file=sys.stderr)
                print("urllib.request http出错")
            except URLError as e:
                # print('Failed to make request: %s' % e.reason, file=sys.stderr)
                print("urllib.request url出错")
            return None

    except AttributeError:
        # OS X Python 2 and 3 don't support tlsv1.1+ therefore... curl
        import subprocess
        try:
            args = ['curl', '--fail', '-sS', '-L', '--get', url]
            for (k,v) in headers.items():
                args.extend(['-H', ': '.join([k, v])])
            if out is None:
                # python3's subprocess.check_output returns stdout as a byte string
                result = subprocess.check_output(args)
                return result.decode('utf-8') if isinstance(result, bytes) else result
            else:
                subprocess.call(args, stdout=out)
        except subprocess.CalledProcessError as e:
            print("出错")
            # print('curl GET error message: %' + (e.message if hasattr(e, 'message') else e.output), file=sys.stderr)
        return None



################################################################################
"""
"""

DESC = "This script will recursively download all files if they don't exist from a LAADS URL and stores them to the specified path"


def sync(ddate,src, dest, tok):
    '''synchronize src url with dest directory'''
    try:
        import csv
        files = [ f for f in csv.DictReader(StringIO(geturl('%s.csv' % src, tok)), skipinitialspace=True) ]
    except ImportError:
        import json
        files = json.loads(geturl(src + '.json', tok))

    exit()
    # use os.path since python 2/3 both support it while pathlib is 3.4+
    items = getItems(r'E:\Code\python\cloudclassfication\data\MODIS\LAADS_query.2022-05-21T05_30.csv')
    for f in files:
        # currently we use filesize of 0 to indicate directory
        filesize = int(f['size'])
        path1 = os.path.join(dest, ddate)
        path = os.path.join(dest,ddate, f['name'])
        url = src + '/' + f['name']
        if filesize == 0:
            try:
                print('creating dir:', path)
                os.mkdir(path)
                sync(src + '/' + f['name'], path, tok)
            except IOError as e:
                print("mkdir `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                sys.exit(-1)
        else:
            try:
                if not os.path.exists(path) and f['name'] in items:
                    print('creating dir:', path)
                    os.mkdir(path1)
                    print('downloading: ', path)
                    with open(path, 'w+b') as fh:
                        geturl(url, tok, fh)
                else:
                    print('skipping: ', path)
            except IOError as e:
                # print("open `%s': %s" % (e.filename, e.strerror), file=sys.stderr)
                # sys.exit(-1)
                print("出错了，但是我不知道原因")
                pass
    return 0

def getItems(filename):
    f = open(filename, 'r')
    f.readline()  #读掉表头
    items = []
    while True:
        itemstr = f.readline()
        if itemstr == '':
            break
        items.append(itemstr.split(',')[1].split('/')[7])
    return items

def date2julian(date):
    year, month, day = int(date[0:4]),int(date[4:6]),int(date[6:8])
    hour = 10
    JD0 = int(365.25 * (year - 1)) + int(30.6001 * (1 + 13)) + 1 + hour / 24 + 1720981.5
    if month <= 2:
        JD2 = int(365.25 * (year - 1)) + int(30.6001 * (month + 13)) + day + hour / 24 + 1720981.5
    else:
        JD2 = int(365.25 * year) + int(30.6001 * (month + 1)) + day + hour / 24 + 1720981.5
    DOY = JD2 - JD0 + 1
    return int(DOY)

def download(dest,files,flags,date_list,tok):
    try:
        index = 0
        for f in files:
            if int(flags[index]) == 1: # 已下载，跳过
                print(index,flags[index],"Have downloaded! skip.")
                index += 1
                continue
            print(f[76:120])
            fname = f[76:120]
            doy = int(f[72:75])
            path = dest+'\\'+date_list[doy]
            if not os.path.exists(path):
                os.mkdir(path)
            path = os.path.join(path,fname)
            print("下载 ",fname,'到 ',path)
            with open(path, 'w+b') as fh:
                geturl(f,tok,fh)
                pass
            # 更新已下载标记
            flags[index] = 1
            csv_path = 'E:\Code\python\cloudclassfication\data\MODIS\LAADS_query.2022-05-21T05_30.csv'
            df = np.array(pd.read_csv(csv_path, index_col=0))
            df[index][3] = 1
            pd.DataFrame(df).to_csv(csv_path)
            index += 1
            # print("子任务完成.")
            # exit()
    except:
        print("发生了错误，问题不大！继续run")

if __name__ == '__main__':
    print("开始下载")
    destination = r'E:\Code\python\cloudclassfication\data\MODIS'
    urlPsth_MOD06 = r'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD06_L2'

    startday = '20080101'
    endday = '20081232'
    startDoy = date2julian(startday)
    endDoy = date2julian(endday)
    date_list = pd.date_range(start=startday, periods=len(range(startDoy, endDoy))).strftime("%Y%m%d").tolist()
    date_list.insert(0,"temp")
    tok = 'Ymx1ZXJhYmJpdHM6YkdsemFHVnVaMTl5WTBBeE5qTXVZMjl0OjE2NTMxMTMzMTk6YWVjZjg0ZWFiMWYzMTM2NTdjMDhlNTg0YjU2NDRmNGViZWI1NmQ5Zg'

    # 待下载列表
    csv_path = 'E:\Code\python\cloudclassfication\data\MODIS\LAADS_query.2022-05-21T05_30.csv'
    df = np.array(pd.read_csv(csv_path))
    if(df[0][0]!=0):
        pd.DataFrame(df).to_csv(csv_path)
    flags = []
    files = []
    with open(r'E:\Code\python\cloudclassfication\data\MODIS\LAADS_query.2022-05-21T05_30.csv', 'r') as rf:
        reader = csv.reader(rf, delimiter=',')
        for row in reader:
            files.extend(['https://ladsweb.modaps.eosdis.nasa.gov' + row[2]])
            flags.extend([row[4]])
    files.remove(files[0])
    flags.remove(flags[0])

    download(destination,files,flags,date_list,tok)

    # for doy in range(startDoy, endDoy):
    #     sync(date_list[doy],urlPsth_MOD06 + '/2008/' + str(doy), destination, tok)
    print("Well done!")