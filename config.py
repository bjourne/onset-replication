from os import name
from os.path import expanduser
from re import match
from socket import gethostname
import rnn
import cnn

########################################################################
# Evaluation settings
########################################################################
ONSET_EVAL_WINDOW = 0.025
# Combine onsets detected less than 30 ms apart.
ONSET_EVAL_COMBINE = 0.03

# If someone wants to use my code they will have to add their paths to
# the following list.
CONFIGS = [
    (('nt', r'^bjourne$'),
     {
         'data-dir' : r'c:\code\material\data\onsets_ISMIR_2012',
         'cache-dir' : r'c:\code\tmp',
         'model-dir' : r'c:\code\tmp',
         'cnn' : {
             'module' : cnn,
             'seed' : 3553105877,
             'digest' : '3c8639887a83b24e9c0c5618d66787d60ee261404df2715754ffbee02aaf2e60'
         },
         'rnn' : {
             'module' : rnn,
             'seed' : 1553106605,
             'digest' : 'd4cc07e09497b8e621b26925088c527acdac7781bf6a1189035331a277cf3ea7'
         }
     }),
    (('posix', r'^.*kth\.se$'),
     {
         'data-dir' : '/tmp/onsets_ISMIR_2012',
         'cache-dir' : '/tmp/cache',
         'model-dir' : expanduser('~/onset_models/eval_data'),
         'cnn' : {
             'module' : cnn,
             'seed' : 3553105877,
             'digest' : '617901ad291705439572e91a0af7438f845806c30eae865e778dcd7621425cd4'
         },
         'rnn' : {
             'module' : rnn,
             'seed' : 3553105877,
             'digest' : '2331eefa22bf15779cf0b6cd370c4aff858b9212e1116c3708b899ede851351b'
         }
     })
]

def get_config():
    hostname = gethostname()
    for (sys_name, pat), config in CONFIGS:
        if sys_name == name and match(pat, hostname):
            return config
    raise Exception('No matching config for %s, %s!' % (name, hostname))
