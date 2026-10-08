"""Start the repository implementation with authorized server-local credentials."""
from pathlib import Path
import os,sys
import json
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
os.environ['REWARD_AS_AGENT_PROVIDER']='doubao'
if not os.environ.get('ARK_API_KEY'):
    key=Path(os.environ['ARK_API_KEY_FILE'])
    os.environ['ARK_API_KEY']=key.read_text().strip()
os.environ['REWARD_AS_AGENT_API_KEY']=os.environ['ARK_API_KEY']
if os.environ.get('DOUBAO_PROXY_SECRET_FILE'):
    token=json.loads(Path(os.environ['DOUBAO_PROXY_SECRET_FILE']).read_text())['token']
    for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY']:
        os.environ[k]='http://download:'+token+'@127.0.0.1:'+os.environ.get('DOUBAO_PROXY_PORT','18889')
os.environ['NO_PROXY']=os.environ['no_proxy']='127.0.0.1,localhost'
from reward_as_agent.cli import main
main(['serve'])
