import ssl
import os

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='zlyzlyzly/CVSBench',
    repo_type='dataset',
    allow_patterns='cvusa/data/*',
    local_dir=r'D:\MRP Test for Remote Sensing Images\database\cvusa_download'
)

print('Download completed!')
