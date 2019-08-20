#!/var/www/lexi/lexi-venv/bin/python

import sys, os
import site

VIRTUALENV = '/var/www/lexi/lexi-venv'

sys.path.insert(0, '/var/www/lexi/')

activate_this = VIRTUALENV+'/bin/activate_this.py'
with open(activate_this) as _f:
    exec(_f.read(), dict(__file__=activate_this))

site.addsitedir(VIRTUALENV+'/lib/python3.6/site-packages/')
#site.addsitedir(VIRTUALENV+'/lib64/python3.6/site-packages/')

from lexi.server.run_lexi_server import app as application
sys.stdout.write("Success!\n")
