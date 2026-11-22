from common import *

def compression_path(case_id,method):
    selected=read(HERE/'compression_selection.json')
    return HERE/selected['folders'][method]/(case_id+'__'+method+'.json')
