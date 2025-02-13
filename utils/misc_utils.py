from datetime import datetime

getdate = lambda : str(datetime.datetime.now()).replace(' ', '_').replace(':', '.')

