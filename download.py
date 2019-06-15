from os.path import join
from requests import Session
from sys import argv, exit
from zipfile import ZipFile

DOC_ID = '1ICEfaZ2r_cnqd3FLNC5F_UOEUalgV7cv'

def download_file_from_google_drive(id, dst):
    print('* Downloading document with id %s to %s.' % (id, dst))
    url = "https://docs.google.com/uc?export=download"
    session = Session()
    response = session.get(url, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(url, params = params, stream = True)
    save_response_content(response, dst)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, dst):
    with open(dst, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

if __name__ == "__main__":
    if len(argv) != 2:
        print('Specify directory to download dataset to.')
        exit(1)
    file_path = join(argv[1], 'onsets.zip')
    download_file_from_google_drive(DOC_ID, file_path)
    print('* Extracting %s' % file_path)
    zf = ZipFile(file_path, 'r')
    zf.extractall(argv[1])
    zf.close()
